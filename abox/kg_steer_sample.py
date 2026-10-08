"""KG-steered sampling: keep generating until enough candidates pass the KG plausibility check.

For each composition: generate --batch candidates at a time (composition-only prompt), score every
valid CIF with the KG plausibility score (abox/kg_steer.py), stop once --target candidates are
KG-plausible or --max-samples were drawn. Compared, on the same compositions and samples:
  unsteered  the first --target valid, right-composition candidates in generation order
  steered    the first --target KG-plausible candidates
  KG top-1   the most KG-plausible candidate among everything drawn
Reports, where the true structure is known (test set), the share of delivered candidates that match it,
plus the cost: samples drawn per composition. Seeds are fixed per composition and round.

    python -u abox/kg_steer_sample.py --model crystallm_v1_large --n-materials 55 --out kg_steer_sample_large.json
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

import phaseC_generate as gen  # noqa: E402
import kg_steer  # noqa: E402

DATA = gen.DATA


def score_text(args):
    """Worker: parse a generated CIF, check composition, KG-score it, compare with the reference."""
    raw, target_formula, truth_cif = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(raw), fmt="cif")
    except Exception:
        return {"valid": False}
    row = {"valid": True, "composition_ok": s.composition.reduced_formula == target_formula}
    if not row["composition_ok"]:
        return row
    row.update(kg_steer.kg_score(s))
    if truth_cif:
        t = Structure.from_str(truth_cif, fmt="cif")
        row["strict"] = bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(s, t))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="phaseC_test_with_pred.json")
    ap.add_argument("--n-materials", type=int, default=55)
    ap.add_argument("--model", default="crystallm_v1_large")
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--gen-dir", default="kg_steer_gen")
    ap.add_argument("--out", default="kg_steer_sample_large.json")
    a = ap.parse_args()
    cands = json.load(open(os.path.join(DATA, a.candidates), encoding="utf-8"))
    cath = [t for t in cands if t.get("cathode_like")]
    chosen = (cath + [t for t in cands if not t.get("cathode_like")])[: a.n_materials]
    model, tok = gen.load_model("cuda", a.model), gen.CIFTokenizer()
    from pymatgen.core import Composition
    out, t0 = {"settings": vars(a), "materials": []}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in enumerate(chosen):
            target_formula = Composition(t["cell"]).reduced_formula
            truth_cif = None
            if not t.get("novel"):
                truth_cif = gen.load_truth(t).to(fmt="cif")
            rows, rnd = [], 0
            os.makedirs(os.path.join(DATA, a.gen_dir, t["id"]), exist_ok=True)
            while len(rows) < a.max_samples:
                torch.manual_seed(a.seed + rnd * 7919 + sum(map(ord, t["id"])))
                cifs = gen.sample_batch(model, tok, gen.prompt_for(t["cell"]), a.batch, "cuda")
                for j, c in enumerate(cifs):
                    with open(os.path.join(DATA, a.gen_dir, t["id"], f"steer_{len(rows) + j}.cif"), "w") as f:
                        f.write(c)
                rows += list(ex.map(score_text, [(c, target_formula, truth_cif) for c in cifs]))
                rnd += 1
                if sum(r.get("plausible", False) for r in rows) >= a.target:
                    break
            ok = [r for r in rows if r.get("composition_ok") and "total" in r]
            plaus = [r for r in ok if r["plausible"]]
            rec = {"id": t["id"], "formula": t["formula"], "samples": len(rows),
                   "valid": sum(r["valid"] for r in rows), "right_composition": len(ok), "plausible": len(plaus),
                   "rows": rows}
            if truth_cif:
                unst = ok[: a.target]
                st = plaus[: a.target]
                rec.update({
                    "unsteered_match": round(sum(r["strict"] for r in unst) / len(unst), 3) if unst else None,
                    "steered_match": round(sum(r["strict"] for r in st) / len(st), 3) if st else None,
                    "kg_top1_match": min(ok, key=lambda r: r["total"])["strict"] if ok else None,
                    "any_match": any(r["strict"] for r in ok)})
            out["materials"].append(rec)
            json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
            print(f"{i + 1}/{len(chosen)} {t['formula']}: {len(rows)} samples, {len(plaus)} plausible"
                  + (f", match unsteered {rec['unsteered_match']} steered {rec['steered_match']}" if truth_cif else "")
                  + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ms = [m for m in out["materials"] if m.get("unsteered_match") is not None]
    both = [m for m in ms if m.get("steered_match") is not None]
    out["summary"] = {
        "compositions": len(out["materials"]),
        "reached_target_plausible": sum(m["plausible"] >= a.target for m in out["materials"]),
        "mean_samples": round(sum(m["samples"] for m in out["materials"]) / len(out["materials"]), 1),
        "valid_share": round(sum(m["valid"] for m in out["materials"]) / sum(m["samples"] for m in out["materials"]), 3),
        "unsteered_match_mean": round(sum(m["unsteered_match"] for m in ms) / len(ms), 3) if ms else None,
        "steered_match_mean_same_compositions": round(sum(m["steered_match"] for m in both) / len(both), 3) if both else None,
        "unsteered_match_mean_same_compositions": round(sum(m["unsteered_match"] for m in both) / len(both), 3) if both else None,
        "kg_top1_match": round(sum(bool(m["kg_top1_match"]) for m in ms) / len(ms), 3) if ms else None,
        "any_match": round(sum(m["any_match"] for m in ms) / len(ms), 3) if ms else None}
    json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
    print(out["summary"])


if __name__ == "__main__":
    main()
