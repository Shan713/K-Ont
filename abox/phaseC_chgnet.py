"""Phase C stability scoring with CHGNet (a machine-learned interatomic potential).

For each material: relax the true structure, then up to --max-per-cond generated candidates per
prompt (valid CIFs with the right reduced composition), all with CHGNet. Records per candidate:
energy per atom after relaxation, dE = E(candidate) - E(true) in eV/atom (0 = as stable as the known
structure; < 0 = CHGNet finds it MORE stable), whether the relaxed candidate matches the relaxed true
structure (CrystaLLM benchmark tolerances: relaxation can turn a near-miss into a match), and its
space group after relaxation. No Materials Project data or API key needed: the reference is the
known structure of the same composition. Saves after every material; --hours stops cleanly.

    python abox/phaseC_chgnet.py --results phaseC_results_large.json --gen-dir phaseC_gen_large \
        --out phaseC_chgnet_large.json --device cpu --max-per-cond 5
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

import phaseC_generate as gen  # noqa: E402
from phaseC_nearmiss import truth_for  # noqa: E402

DATA = gen.DATA


def relax(opt, model, s, fmax, steps):
    r = opt.relax(s, fmax=fmax, steps=steps, verbose=False)
    final = r["final_structure"]
    e = float(model.predict_structure(final)["e"])
    return final, e, len(r["trajectory"].energies)


def summarise(materials):
    summ = {}
    for cond in sorted({c for m in materials for c in m["conditions"]}):
        rows = [r for m in materials for r in m["conditions"].get(cond, []) if r.get("dE") is not None]
        best = [min(r["dE"] for r in m["conditions"][cond] if r.get("dE") is not None)
                for m in materials if any(r.get("dE") is not None for r in m["conditions"].get(cond, []))]
        if not rows:
            continue
        summ[cond] = {
            "candidates_relaxed": len(rows), "materials": len(best),
            "median_dE": round(statistics.median(r["dE"] for r in rows), 4),
            "within_0.05_eV": round(sum(r["dE"] <= 0.05 for r in rows) / len(rows), 3),
            "within_0.10_eV": round(sum(r["dE"] <= 0.10 for r in rows) / len(rows), 3),
            "materials_best_within_0.05_eV": sum(b <= 0.05 for b in best),
            "materials_best_below_true": sum(b < -0.005 for b in best),
            "match_after_relax": sum(r["match_after_relax"] for r in rows),
            "materials_match_after_relax": sum(any(r.get("match_after_relax") for r in m["conditions"].get(cond, []))
                                               for m in materials)}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--candidates", default="phaseC_test_with_pred.json")
    ap.add_argument("--conditions", default="composition,oracle")
    ap.add_argument("--max-per-cond", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--fmax", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hours", type=float, default=0, help="stop cleanly after this many hours (0 = no limit)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from chgnet.model import CHGNet, StructOptimizer
    model = CHGNet.load()
    opt = StructOptimizer(model=model, use_device=a.device)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    results = json.load(open(os.path.join(DATA, a.results)))
    cands = {c["id"]: c for c in json.load(open(os.path.join(DATA, a.candidates), encoding="utf-8"))}
    keep = a.conditions.split(",")
    out = {"results": a.results, "gen_dir": a.gen_dir, "settings": vars(a), "materials": []}
    t0 = time.time()
    for i, rec in enumerate(results):
        if a.hours and time.time() - t0 > a.hours * 3600:
            print("time budget reached, stopping", flush=True)
            break
        truth = truth_for(rec, cands)
        try:
            t_rel, e_true, _ = relax(opt, model, truth, a.fmax, a.steps)
        except Exception as e:
            print(f"{rec['formula']}: true structure failed to relax ({e})", flush=True)
            continue
        mrec = {"id": rec["id"], "formula": rec["formula"], "sg": rec["sg"], "cathode_like": rec.get("cathode_like"),
                "e_true": round(e_true, 5), "conditions": {}}
        for cond in keep:
            if cond not in rec["conditions"]:
                continue
            rows = []
            for k in range(rec["conditions"][cond]["n"]):
                if len(rows) >= a.max_per_cond:
                    break
                path = os.path.join(DATA, a.gen_dir, rec["id"], f"{cond}_{k}.cif")
                if not os.path.exists(path):
                    continue
                try:
                    s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
                except Exception:
                    continue
                if s.composition.reduced_formula != truth.composition.reduced_formula:
                    continue
                try:
                    s_rel, e, nsteps = relax(opt, model, s, a.fmax, a.steps)
                    try:
                        sg = SpacegroupAnalyzer(s_rel, symprec=0.1).get_space_group_number()
                    except Exception:
                        sg = None
                    rows.append({"k": k, "e": round(e, 5), "dE": round(e - e_true, 5), "steps": nsteps,
                                 "sg_after_relax": sg, "match_after_relax": bool(matcher.fit(s_rel, t_rel))})
                except Exception as e:
                    rows.append({"k": k, "error": str(e)[:200]})
            mrec["conditions"][cond] = rows
        out["materials"].append(mrec)
        out["summary"] = summarise(out["materials"])
        json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
        best = {c: min((r["dE"] for r in rows if "dE" in r), default=None) for c, rows in mrec["conditions"].items()}
        print(f"{i + 1}/{len(results)} {rec['formula']}: best dE " +
              " | ".join(f"{c} {v:+.3f}" if v is not None else f"{c} -" for c, v in best.items()) +
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for k, v in out.get("summary", {}).items():
        print(k, v)


if __name__ == "__main__":
    main()
