"""Novel-cathode pilot, step 4: KG/ontology steering applied to the generated novel structures.

Inputs (from the workstation pilot): data/crystallm/novel_results_*.json, novel_chgnet_*.json and the
generated CIFs in data/crystallm/novel_gen/; data/crystallm/novel_parents.json (the parent cathodes'
structures and ontology structure families, from the KG).

Per composition and prompt (composition only; parent's space group = KG-analogue requirement), for
every valid CIF with the right composition:
  kg       KG plausibility score (abox/kg_steer.py)
  keeps    the parent's framework: StructureMatcher.fit_anonymous against the parent's structure (same
           structure type with the elements anonymised -- a substituted cathode that keeps its
           parent's olivine/garnet/... framework). Non-circular: it compares geometry, not the prompt.
Then a shortlist: per composition, among the candidates CHGNet relaxed, the KG-plausible ones (all, if
none pass), lowest CHGNet energy; reported with framework, family, energy and capacity.

    python abox/novel_steer_analysis.py --workers 14
"""
import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import phaseC_generate as gen  # noqa: E402
import kg_steer  # noqa: E402

DATA = gen.DATA


def analyse(args):
    path, target, parent_cif = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
    except Exception:
        return path, {"valid": False}
    row = {"valid": True, "composition_ok": s.composition.reduced_formula == target}
    if not row["composition_ok"]:
        return path, row
    row.update(kg_steer.kg_score(s))
    try:
        p = Structure.from_str(parent_cif, fmt="cif")
        row["keeps_parent_framework"] = bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit_anonymous(s, p))
    except Exception:
        row["keeps_parent_framework"] = False
    return path, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--gen-dir", default="novel_gen")
    a = ap.parse_args()
    from pymatgen.core import Composition
    cands = {c["id"]: c for c in json.load(open(os.path.join(DATA, "novel_candidates.json"), encoding="utf-8"))}
    parents = json.load(open(os.path.join(DATA, "novel_parents.json"), encoding="utf-8"))
    results = [m for f in sorted(glob.glob(os.path.join(DATA, "novel_results_*.json"))) for m in json.load(open(f))]
    energies = {}
    for f in sorted(glob.glob(os.path.join(DATA, "novel_chgnet_*.json"))):
        for m in json.load(open(f))["materials"]:
            for cond, rows in m["conditions"].items():
                for r in rows:
                    if "e" in r:
                        energies[(m["id"], cond, r["k"])] = r
    jobs = []
    for m in results:
        c = cands[m["id"]]
        target = Composition(c["cell"]).reduced_formula
        pcif = parents[c["parent_id"]]["cif"]
        for cond, v in m["conditions"].items():
            for k in range(v["n"]):
                jobs.append((os.path.join(DATA, a.gen_dir, m["id"], f"{cond}_{k}.cif"), target, pcif))
    with ProcessPoolExecutor(a.workers) as ex:
        scored = dict(ex.map(analyse, jobs, chunksize=4))

    per = defaultdict(lambda: defaultdict(list))
    for (path, _, _) in jobs:
        mid, fname = path.split(os.sep)[-2], os.path.basename(path)
        cond, k = fname[:-4].rsplit("_", 1)
        per[mid][cond].append((int(k), scored[path]))
    summary = {}
    for cond in ("composition", "parent_sg"):
        rows = [r for mid in per for _, r in per[mid].get(cond, [])]
        ok = [r for r in rows if r.get("composition_ok") and "total" in r]
        fam_rows = [r for mid in per if parents[cands[mid]["parent_id"]]["structure_family"]
                    for _, r in per[mid].get(cond, []) if r.get("composition_ok") and "total" in r]
        summary[cond] = {
            "cifs": len(rows), "valid": round(sum(r["valid"] for r in rows) / max(len(rows), 1), 3),
            "right_composition": round(len(ok) / max(len(rows), 1), 3),
            "kg_plausible": round(sum(r["plausible"] for r in ok) / max(len(ok), 1), 3),
            "keeps_parent_framework": round(sum(r["keeps_parent_framework"] for r in ok) / max(len(ok), 1), 3),
            "keeps_framework_family_labelled_parents": round(sum(r["keeps_parent_framework"] for r in fam_rows) / max(len(fam_rows), 1), 3),
            "compositions_with_a_framework_keeping_candidate": sum(
                any(r.get("keeps_parent_framework") for _, r in per[mid].get(cond, [])) for mid in per)}
        print(cond, summary[cond], flush=True)

    shortlist = []
    for mid in per:
        c, p = cands[mid], parents[cands[mid]["parent_id"]]
        relaxed = []
        for cond in per[mid]:
            for k, r in per[mid][cond]:
                e = energies.get((mid, cond, k))
                if e and "total" in r:
                    relaxed.append((cond, k, r, e))
        if not relaxed:
            continue
        pool = [x for x in relaxed if x[2]["plausible"]] or relaxed
        cond, k, r, e = min(pool, key=lambda x: x[3]["e"])
        shortlist.append({"id": mid, "formula": c["formula"], "substitution": c["substitution"],
                          "parent": c["parent_formula"], "parent_family": p["structure_family"],
                          "prompt": cond, "k": k, "e_chgnet": e["e"], "sg_after_relax": e.get("sg_after_relax"),
                          "kg_plausible": r["plausible"], "kg_score": r["total"],
                          "keeps_parent_framework": r["keeps_parent_framework"],
                          "capacity_upper_bound_mAh_g": c["theoretical_capacity_mAh_g"]})
    shortlist.sort(key=lambda x: (not x["kg_plausible"], not x["keeps_parent_framework"], -x["capacity_upper_bound_mAh_g"]))
    out = {"summary": summary, "shortlist": shortlist,
           "per_cif": {f"{mid}/{cond}_{k}": r for mid in per for cond in per[mid] for k, r in per[mid][cond]}}
    json.dump(out, open(os.path.join(DATA, "novel_steer_analysis.json"), "w"), indent=1)
    lines = ["# Novel cathodes: KG-steered shortlist", "",
             "Selection: KG-plausible first, then keeps the parent's framework, then capacity (upper bound).",
             "Energies are CHGNet eV/atom and are NOT comparable across compositions (stability vs known",
             "phases is the next step).", "",
             "| formula | swap | parent | family | prompt | KG-plausible | keeps framework | E (eV/atom) | capacity |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {s['formula']} | {s['substitution']} | {s['parent']} | {s['parent_family'] or '-'} | {s['prompt']} | "
              f"{s['kg_plausible']} | {s['keeps_parent_framework']} | {s['e_chgnet']:.4f} | {s['capacity_upper_bound_mAh_g']} |"
              for s in shortlist]
    open(os.path.join(DATA, "novel_shortlist.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"shortlist: {len(shortlist)} compositions; KG-plausible {sum(s['kg_plausible'] for s in shortlist)}, "
          f"keep parent framework {sum(s['keeps_parent_framework'] for s in shortlist)}")


if __name__ == "__main__":
    main()
