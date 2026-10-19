"""A bigger KG for the template pool, built in chunks that fit the laptop (BUILD_LOG Step 56).

Templates can only come from the KG: for 15% of the 902 K-Ont-unseen targets no pool material has the true
framework (Step 40 oracle 85%). This adds more Materials Project materials to the pool, through the SAME battGPT
pipeline and K-Ont extraction/verbalisation as the 5,000 KG, without retraining K-Ont:
  select   the battGPT --grow candidate cache (Li/Na broad pool + ratio sweeps: one representative per formula +
           polymorph extras) minus the 5,000 KG's materials minus every evaluation composition (the 1,004
           reduced formulas of Step 45: all K-Ont-unseen KG materials and the 200 novel cathodes) -> chunks of
           --chunk ids (data/kg_ext/ids_chunk<i>.json)
  build    battGPT scripts/populate_kg_ont.py --ids-file per chunk (~1.4 s/material; MP key from battGPT's own
           config, never read here) -> ../battGPT/output/battgpt_kg_ext_<i>/battery_kg.ttl
  extract  abox/extract.py + abox/verbalize.py --no-labels per chunk (one chunk at a time: the 5,000 KG's
           extraction peaked at 7.3 GB) -> data/kg_ext/abox_<i>/
  pool     every chunk's materials as pool rows (CrystaLLM-layout CIF via kg_corpus.convert, full sentence)
           -> data/crystallm/kg_ext_pool.json.gz
Resumable: each stage skips chunks already done.

    .venv-crystallm/Scripts/python.exe abox/kg_extend.py --stage select
    .venv-crystallm/Scripts/python.exe abox/kg_extend.py --stage build      (calls battGPT's venv)
    .venv-crystallm/Scripts/python.exe abox/kg_extend.py --stage extract    (calls K-Ont's .venv)
    .venv-crystallm/Scripts/python.exe abox/kg_extend.py --stage pool
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATT = os.path.join(os.path.dirname(HERE), "battGPT")
sys.path.insert(0, os.path.join(HERE, "abox"))
EXT = os.path.join(HERE, "data", "kg_ext")
DATA = os.path.join(HERE, "data", "crystallm")
CACHE = os.path.join(BATT, "K-Ont", "data", "candidate_materials_grow.json")


def chunks():
    return sorted(glob.glob(os.path.join(EXT, "ids_chunk*.json")), key=lambda p: int(p.split("chunk")[-1].split(".")[0]))


def stage_select(a):
    from pymatgen.core import Composition
    os.makedirs(EXT, exist_ok=True)
    cache = json.load(open(CACHE))
    have = {m.split("/")[-1] for m in json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "entities.json"),
                                                     encoding="utf-8"))["materials"]}
    ban = set(json.load(open(os.path.join(DATA, "scratch", "banned_formulas.json"))))
    ids, banned, dup = [], 0, 0
    rows = sorted(cache.items(), key=lambda fe: (fe[1].get("energy_above_hull") if fe[1].get("energy_above_hull") is not None else 9, fe[0]))
    for f, e in rows:
        try:
            rf = Composition(f).reduced_formula
        except Exception:
            continue
        if rf in ban:
            banned += 1 + len(e.get("polymorph_extras", []))
            continue
        for mid in [e["material_id"]] + [x["material_id"] for x in e.get("polymorph_extras", [])]:
            if mid in have:
                dup += 1
            elif mid not in ids:
                ids.append(mid)
    ids = ids[:a.limit] if a.limit else ids
    for i in range(0, len(ids), a.chunk):
        json.dump(ids[i:i + a.chunk], open(os.path.join(EXT, f"ids_chunk{i // a.chunk}.json"), "w"))
    s = {"candidate_formulas": len(cache), "already_in_5000_kg": dup, "evaluation_compositions_skipped": banned,
         "new_materials": len(ids), "chunks": (len(ids) + a.chunk - 1) // a.chunk, "chunk_size": a.chunk}
    json.dump(s, open(os.path.join(EXT, "select_summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


def stage_build(a):
    py = os.path.join(BATT, ".venv", "Scripts", "python.exe")
    for p in chunks():
        i = p.split("chunk")[-1].split(".")[0]
        out = os.path.join(BATT, "output", f"battgpt_kg_ext_{i}")
        if os.path.exists(os.path.join(out, "battery_kg.ttl")):
            continue
        print(f"building chunk {i} ({len(json.load(open(p)))} materials) -> {out}", flush=True)
        with open(os.path.join(EXT, f"build_{i}.log"), "w") as f:
            r = subprocess.run([py, "scripts/populate_kg_ont.py", "--ids-file", p, "--out-dir", out], cwd=BATT,
                               stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"chunk {i} FAILED (see data/kg_ext/build_{i}.log); stopping", flush=True)
            sys.exit(1)


def stage_extract(a):
    py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
    for p in chunks():
        i = p.split("chunk")[-1].split(".")[0]
        ttl = os.path.join(BATT, "output", f"battgpt_kg_ext_{i}", "battery_kg.ttl")
        d = os.path.join(EXT, f"abox_{i}")
        if not os.path.exists(ttl) or os.path.exists(os.path.join(d, "verbalizations_materials_full.json")):
            continue
        os.makedirs(d, exist_ok=True)
        print(f"extracting chunk {i}", flush=True)
        subprocess.run([py, "abox/extract.py", "--ttl", ttl, "--out", os.path.join(d, "entities.json")], cwd=HERE, check=True)
        subprocess.run([py, "abox/verbalize.py", "--entities", os.path.join(d, "entities.json"), "--out-dir", d, "--no-labels"],
                       cwd=HERE, check=True)


def val(x):
    """entities store properties as {"value": v, "unit": u}; the corpus stores the number"""
    return x.get("value") if isinstance(x, dict) else x


def _convert(item):
    import kg_corpus
    return kg_corpus.convert(item)


def stage_pool(a):
    from concurrent.futures import ProcessPoolExecutor
    rows, sents = [], {}
    for d in sorted(glob.glob(os.path.join(EXT, "abox_*"))):
        if not os.path.exists(os.path.join(d, "verbalizations_materials_full.json")):
            continue
        ents = json.load(open(os.path.join(d, "entities.json"), encoding="utf-8"))
        vm = json.load(open(os.path.join(d, "verbalizations_materials_full.json"), encoding="utf-8"))
        for mkey, m in ents["materials"].items():
            c = ents["crystals"].get(m.get("structure")) or {}
            if not c.get("cif"):
                continue
            mid = mkey.split("/")[-1]
            sg = str(c.get("space_group") or "").split("/")[-1]  # entities: "spacegroup/74", corpus: 74
            cs = str(c.get("crystal_system") or "").split("/")[-1] or None  # "crystalsystem/orthorhombic" -> "orthorhombic"
            rows.append(({"id": mid, "formula": m["formula"], "split": "ext", "space_group": int(sg) if sg.isdigit() else None,
                          "crystal_system": cs, "structure_family": c.get("structure_family"),
                          "battery_role": m.get("battery_role"), "e_above_hull": val((m.get("properties") or {}).get("energy_above_hull")),
                          "band_gap": val((m.get("properties") or {}).get("band_gap")), "cif": c["cif"]}, 1016))
            sents[mid] = vm[mkey]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_convert, rows, chunksize=16))
    pool = [{**r, "sentence": sents[r["id"]]} for _, r, err in res if r]
    errs = {}
    for _, r, err in res:
        if not r:
            errs[err] = errs.get(err, 0) + 1
    json.dump(pool, gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "wt", encoding="utf-8"))
    print(f"{len(pool)} extension pool materials; dropped {errs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "build", "extract", "pool"])
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="cap on new materials (0 = all candidates)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"select": stage_select, "build": stage_build, "extract": stage_extract, "pool": stage_pool}[a.stage](a)


if __name__ == "__main__":
    main()
