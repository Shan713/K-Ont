"""Phase C near-miss analysis: how close are the generated structures that didn't match?

A strict StructureMatcher "no" can hide "right structure type, slightly off". For every valid
generated CIF this records: is the reduced composition right, is the space group right (generated
structure re-analysed with SpacegroupAnalyzer, symprec 0.1), is the atom count per formula unit
right, the volume ratio to the true structure, the strict match and RMS distance (CrystaLLM
benchmark tolerances), and a loose match (ltol 0.3, stol 0.5, angle_tol 10). CPU only.

    python abox/phaseC_nearmiss.py --results phaseC_results_large.json --gen-dir phaseC_gen_large \
        --out phaseC_nearmiss_large.json
"""
import argparse
import json
import os
import statistics
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

import phaseC_generate as gen  # noqa: E402  (postprocess + the pymatgen compatibility alias)

DATA = gen.DATA


def truth_for(rec, cands):
    return gen.load_truth(cands.get(rec["id"], {"id": rec["id"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results JSON under data/crystallm")
    ap.add_argument("--gen-dir", required=True, help="generated-CIF folder under data/crystallm")
    ap.add_argument("--candidates", default="phaseC_test_with_pred.json")
    ap.add_argument("--out", required=True, help="output JSON under data/crystallm")
    a = ap.parse_args()
    results = json.load(open(os.path.join(DATA, a.results)))
    cands = {c["id"]: c for c in json.load(open(os.path.join(DATA, a.candidates), encoding="utf-8"))}
    strict = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    out = {"results": a.results, "gen_dir": a.gen_dir, "materials": []}
    agg = {}
    for i, rec in enumerate(results):
        truth = truth_for(rec, cands)
        t_red = truth.composition.reduced_formula
        t_sg = rec["sg"]
        t_fu = len(truth) / truth.composition.get_reduced_composition_and_factor()[1]
        mrec = {"id": rec["id"], "formula": rec["formula"], "sg": t_sg, "cathode_like": rec.get("cathode_like"),
                "conditions": {}}
        for cond, v in rec["conditions"].items():
            rows = []
            for k in range(v["n"]):
                path = os.path.join(DATA, a.gen_dir, rec["id"], f"{cond}_{k}.cif")
                if not os.path.exists(path):
                    continue
                try:
                    s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
                except Exception:
                    continue
                row = {"k": k, "composition_ok": s.composition.reduced_formula == t_red}
                try:
                    row["sg"] = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
                except Exception:
                    row["sg"] = None
                row["sg_ok"] = row["sg"] == t_sg
                if row["composition_ok"]:
                    fu = len(s) / s.composition.get_reduced_composition_and_factor()[1]
                    row["atoms_per_fu_ok"] = abs(fu - t_fu) < 1e-6
                    row["volume_ratio_per_atom"] = round((s.volume / len(s)) / (truth.volume / len(truth)), 3)
                    try:
                        row["strict"] = bool(strict.fit(s, truth))
                        rms = strict.get_rms_dist(s, truth)
                        row["rms"] = round(rms[0], 4) if rms else None
                        row["loose"] = bool(loose.fit(s, truth))
                    except Exception:
                        row.update(strict=False, rms=None, loose=False)
                rows.append(row)
            mrec["conditions"][cond] = rows
            a_ = agg.setdefault((cond, bool(rec.get("cathode_like"))), {"valid": 0, "comp": 0, "sg": 0, "strict": 0, "loose": 0,
                                                                        "mat_loose": 0, "mat_sg": 0, "n_mat": 0, "vol": []})
            a_["n_mat"] += 1
            a_["valid"] += len(rows)
            a_["comp"] += sum(r["composition_ok"] for r in rows)
            a_["sg"] += sum(r["sg_ok"] for r in rows)
            a_["strict"] += sum(r.get("strict", False) for r in rows)
            a_["loose"] += sum(r.get("loose", False) for r in rows)
            a_["mat_loose"] += any(r.get("loose") for r in rows)
            a_["mat_sg"] += any(r["sg_ok"] for r in rows)
            a_["vol"] += [r["volume_ratio_per_atom"] for r in rows if "volume_ratio_per_atom" in r]
        out["materials"].append(mrec)
        print(f"{i + 1}/{len(results)} {rec['formula']}", flush=True)
    summary = {}
    for (cond, cath), a_ in sorted(agg.items()):
        n = max(a_["valid"], 1)
        summary[f"{cond} | {'cathode-like' if cath else 'other'}"] = {
            "materials": a_["n_mat"], "valid_cifs": a_["valid"],
            "right_composition": round(a_["comp"] / n, 3), "right_space_group": round(a_["sg"] / n, 3),
            "strict_match": round(a_["strict"] / n, 3), "loose_match": round(a_["loose"] / n, 3),
            "materials_with_loose_match": a_["mat_loose"], "materials_with_right_space_group": a_["mat_sg"],
            "median_volume_ratio": round(statistics.median(a_["vol"]), 3) if a_["vol"] else None}
    out["summary"] = summary
    json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
