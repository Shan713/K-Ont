"""30k generation, P6 + P7: per-system quota, cross-source duplicate check, release folders and manifest (Step 66).

Sources (manifests written by P2 / P3 / P4): K-Ont templates, mixed-site templates, CrystaLLM samples.
  select  (crystallm venv) per system, at most --target CIFs. Within a formula the order interleaves the knowledge
          sources best-first (template rank 1, mixed host-1 ordering 1, template rank 2, mixed next, ...), and formulas
          are taken round-robin, so every formula is represented before any formula gets its 2nd structure; CrystaLLM
          samples come after the knowledge-guided ones. P2 and P3 were not deduplicated against each other, so a
          structure that matches an already selected one of the same formula (StructureMatcher 0.2 / 0.3 / 5 deg) is
          skipped here. Each selected CIF gets its space group (SpacegroupAnalyzer, symprec 0.1).
  mpflag  (battGPT venv) tags every selected formula already in Materials Project (the key stays in battGPT/.env)
  release copies to data/gen30k/release/<system>/<id>.cif, writes manifest.csv + README.txt, one zip per system
          and one zip of everything

    .venv-crystallm/Scripts/python.exe abox/gen30k_package.py --stage select [--workers 3]
    ../battGPT/.venv/Scripts/python.exe abox/gen30k_package.py --stage mpflag
    .venv-crystallm/Scripts/python.exe abox/gen30k_package.py --stage release
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import warnings
import zipfile
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "data", "gen30k")
SEL = os.path.join(OUT, "selected.json")
MPF = os.path.join(OUT, "mp_flags.json")
REL = os.path.join(OUT, "release")
SYSTEM_ORDER = None  # filled from compositions.json (ternary -> septenary)


def load_rows():
    rows = []
    for name in ("templates_manifest.json", "mixed_manifest.json", "crystallm_manifest.json"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            rows += json.load(open(p))["rows"]
    return rows


def within_formula_order(r):
    """knowledge sources interleaved best-first, CrystaLLM after them"""
    if r["source"] == "kont_template":
        return (0, 2 * int(r["file"].split("__t")[1].split("__")[0]) - 1)
    if r["source"] == "mixed_site_template":
        return (0, 2 * ((r["kont_host_rank"] - 1) * 3 + r["ewald_rank"]))
    return (1, r["round"] * 100 + r["sample"])


def _sg(path):
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        s = Structure.from_file(path)
        a = SpacegroupAnalyzer(s, symprec=0.1)
        return a.get_space_group_symbol(), a.get_space_group_number()
    except Exception:
        return None, None


def stage_select(a):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    by_sys = defaultdict(lambda: defaultdict(list))
    for r in load_rows():
        by_sys[r["system"]][r["formula"]].append(r)
    systems = list(json.load(open(os.path.join(OUT, "compositions.json")))["systems"])
    selected, report, t0 = [], {}, time.time()
    for sysname in systems:
        forms = by_sys.get(sysname, {})
        queues = {f: sorted(rs, key=within_formula_order) for f, rs in forms.items()}
        kept, dup = defaultdict(list), 0
        picked = []
        # round-robin over formulas (fewest atoms first inside a round for determinism: sorted formula names)
        while len(picked) < a.target and any(queues.values()):
            for f in sorted(queues):
                if len(picked) >= a.target:
                    break
                while queues[f]:
                    r = queues[f].pop(0)
                    s = Structure.from_file(os.path.join(OUT, r["file"]))
                    # P2 vs P3 never compared; CrystaLLM rows were already checked against both
                    if r["source"] != "crystallm" and any(sm.fit(s, q) for q in kept[f]):
                        dup += 1
                        continue
                    kept[f].append(s)
                    picked.append(r)
                    break
        src = defaultdict(int)
        for r in picked:
            src[r["source"]] += 1
        report[sysname] = {"available": sum(len(v) for v in forms.values()), "selected": len(picked),
                           "formulas": len({r["formula"] for r in picked}), "cross_source_duplicates": dup, **src}
        selected += picked
        print(f"{sysname:20s} {report[sysname]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    paths = [os.path.join(OUT, r["file"]) for r in selected]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for r, (sym, num) in zip(selected, ex.map(_sg, paths, chunksize=64)):
            r["space_group"], r["space_group_number"] = sym, num
    tot = defaultdict(int)
    for v in report.values():
        for k in ("selected", "kont_template", "mixed_site_template", "crystallm"):
            tot[k] += v.get(k, 0)
    json.dump({"target_per_system": a.target, "report": report, "total": tot, "rows": selected}, open(SEL, "w"), indent=1)
    print(json.dumps(dict(tot)))


def stage_mpflag(a):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "battGPT"))
    from mp_api.client import MPRester
    from pipeline.config import PipelineConfig
    from pymatgen.core import Composition
    names = sorted({r["formula"] for r in json.load(open(SEL))["rows"]})
    in_mp = set()
    with MPRester(PipelineConfig().mp_api_key, mute_progress_bars=True) as mpr:
        for i in range(0, len(names), 100):
            docs = mpr.materials.summary.search(formula=names[i:i + 100], fields=["formula_pretty"])
            in_mp |= {Composition(x.formula_pretty).reduced_formula for x in docs}
    json.dump({f: f in in_mp for f in names}, open(MPF, "w"))
    print(f"{len(names)} formulas, {sum(f in in_mp for f in names)} already in Materials Project")


COLUMNS = ["id", "system", "n_elements", "formula", "formula_in_materials_project", "source", "space_group",
           "space_group_number", "n_atoms", "kont_rank", "kont_score", "template_id", "template_formula",
           "template_space_group", "template_kind", "host_id", "host_formula", "host_space_group", "ewald_rank",
           "supercell", "metal_counts", "crystallm_model", "crystallm_z", "path"]


def stage_release(a):
    from pymatgen.core import Structure
    d = json.load(open(SEL))
    mp = json.load(open(MPF)) if os.path.exists(MPF) else {}
    if os.path.exists(REL):
        shutil.rmtree(REL)
    os.makedirs(REL)
    rows, count = [], defaultdict(int)
    for r in d["rows"]:
        s = r["system"]
        count[s] += 1
        rid = f"G025-{s}-{count[s]:04d}"
        dst = os.path.join(REL, s, rid + ".cif")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(OUT, r["file"]), dst)
        n_atoms = r.get("n_atoms") or len(Structure.from_file(dst))
        rows.append({"id": rid, "system": s, "n_elements": s.count("-") + 1, "formula": r["formula"],
                     "formula_in_materials_project": mp.get(r["formula"], ""), "source": r["source"],
                     "space_group": r.get("space_group"), "space_group_number": r.get("space_group_number"),
                     "n_atoms": n_atoms, "kont_rank": r.get("kont_rank", r.get("kont_host_rank")),
                     "kont_score": r.get("kont_score"), "template_id": r.get("template_id"),
                     "template_formula": r.get("template_formula"), "template_space_group": r.get("template_space_group"),
                     "template_kind": r.get("template_kind"), "host_id": r.get("host_id"), "host_formula": r.get("host_formula"),
                     "host_space_group": r.get("host_space_group"), "ewald_rank": r.get("ewald_rank"),
                     "supercell": json.dumps(r["supercell"]) if r.get("supercell") else "",
                     "metal_counts": json.dumps(r["metal_counts"]) if r.get("metal_counts") else "",
                     "crystallm_model": r.get("model"), "crystallm_z": r.get("z"), "path": f"{s}/{rid}.cif"})
    with open(os.path.join(REL, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tot = d["total"]
    readme = f"""G025 knowledge-guided Li-oxide structures (K-Ont template generation)

{len(rows)} CIFs over {len(count)} chemical systems: Li + 1-5 of Al, Fe, Mn, Ni, V + O; at most {d['target_per_system']} per system.
Generated, NOT relaxed: positions come from known frameworks (templates) or from CrystaLLM. Relax before any use.

Sources (column `source`):
  kont_template        {tot.get('kont_template', 0):6d}  a known framework from the KG / Materials Project pool, chosen by the
                                K-Ont retriever for the formula, with the new elements substituted; cell scaled by atomic volume
  mixed_site_template  {tot.get('mixed_site_template', 0):6d}  a known single-metal Li oxide framework, enlarged, with the system's metals
                                ordered on its metal sites by lowest Ewald energy; host chosen by K-Ont
  crystallm            {tot.get('crystallm', 0):6d}  CrystaLLM (small, fine-tuned on the battery KG) sampled for the formula

Checks on every CIF: right composition, no two atoms closer than 0.5 A, no duplicate of the same formula
(StructureMatcher ltol 0.2, stol 0.3, angle 5). Space groups: SpacegroupAnalyzer, symprec 0.1, on the unrelaxed cell.
Oxidation states assumed for charge balance: Li+ Al3+ Fe2+/3+ Mn2+/3+/4+ Ni2+/3+ V3+/4+/5+ O2-.
manifest.csv: one row per CIF, with provenance (template / host Materials Project id, K-Ont rank, Ewald rank).
"""
    open(os.path.join(REL, "README.txt"), "w", encoding="utf-8").write(readme)
    zdir = os.path.join(OUT, "release_zips")
    os.makedirs(zdir, exist_ok=True)
    with zipfile.ZipFile(os.path.join(zdir, "G025_gen30k_all.zip"), "w", zipfile.ZIP_DEFLATED) as zall:
        for s in count:
            with zipfile.ZipFile(os.path.join(zdir, f"G025_gen30k_{s}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(os.path.join(REL, s))):
                    z.write(os.path.join(REL, s, fn), f"{s}/{fn}")
                    zall.write(os.path.join(REL, s, fn), f"{s}/{fn}")
                z.write(os.path.join(REL, "README.txt"), "README.txt")
        zall.write(os.path.join(REL, "manifest.csv"), "manifest.csv")
        zall.write(os.path.join(REL, "README.txt"), "README.txt")
    print(f"released {len(rows)} CIFs in {len(count)} systems -> {REL} and {zdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "mpflag", "release"])
    ap.add_argument("--target", type=int, default=970)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"select": stage_select, "mpflag": stage_mpflag, "release": stage_release}[a.stage](a)


if __name__ == "__main__":
    main()
