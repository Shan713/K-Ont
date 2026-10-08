"""Novel-cathode pilot, step 3: relax generated candidates with CHGNet and keep the most stable.

For each novel composition and prompt: relax up to --max-per-cond valid candidates with the right
composition, record energy per atom and space group after relaxation, and save the lowest-energy
relaxed structure overall to data/crystallm/novel_best/<id>.cif. Also compares the two prompts:
which one produced the lower-energy structure (CrystaLLM's own choice vs the parent cathode's space
group, a KG analogue). Saves after every composition; --start/--end select a chunk.

    python abox/novel_chgnet.py --results novel_results_000.json --gen-dir novel_gen --out novel_chgnet_000.json
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
from pymatgen.core import Composition, Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

import phaseC_generate as gen  # noqa: E402

DATA = gen.DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--conditions", default="composition,parent_sg")
    ap.add_argument("--max-per-cond", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fmax", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from chgnet.model import CHGNet, StructOptimizer
    model = CHGNet.load()
    opt = StructOptimizer(model=model, use_device="cpu")
    best_dir = os.path.join(DATA, "novel_best")
    os.makedirs(best_dir, exist_ok=True)
    results = json.load(open(os.path.join(DATA, a.results)))
    keep = a.conditions.split(",")
    out = {"results": a.results, "settings": vars(a), "materials": []}
    t0 = time.time()
    for i, rec in enumerate(results):
        target = Composition(rec["cell"]).reduced_formula
        mrec = {"id": rec["id"], "formula": rec["formula"], "conditions": {}}
        best = None
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
                if s.composition.reduced_formula != target:
                    continue
                try:
                    r = opt.relax(s, fmax=a.fmax, steps=a.steps, verbose=False)
                    final = r["final_structure"]
                    e = float(model.predict_structure(final)["e"])
                    try:
                        sg = SpacegroupAnalyzer(final, symprec=0.1).get_space_group_number()
                    except Exception:
                        sg = None
                    rows.append({"k": k, "e": round(e, 5), "sg_after_relax": sg,
                                 "steps": len(r["trajectory"].energies)})
                    if best is None or e < best[0]:
                        best = (e, cond, k, final, sg)
                except Exception as ex:
                    rows.append({"k": k, "error": str(ex)[:200]})
            mrec["conditions"][cond] = rows
        if best:
            e, cond, k, final, sg = best
            final.to(filename=os.path.join(best_dir, rec["id"] + ".cif"))
            mrec["best"] = {"e": round(e, 5), "condition": cond, "k": k, "sg_after_relax": sg}
        out["materials"].append(mrec)
        json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
        b = mrec.get("best")
        print(f"{i + 1}/{len(results)} {rec['formula']}: " +
              (f"best {b['e']:.4f} eV/atom from {b['condition']} (sg {b['sg_after_relax']})" if b else "no relaxed candidate") +
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # prompt comparison: lowest energy per prompt, per composition
    wins = {"composition": 0, "parent_sg": 0, "tie": 0}
    diffs = []
    for m in out["materials"]:
        bc = {c: min((r["e"] for r in rows if "e" in r), default=None) for c, rows in m["conditions"].items()}
        if bc.get("composition") is not None and bc.get("parent_sg") is not None:
            d = bc["parent_sg"] - bc["composition"]
            diffs.append(d)
            wins["tie" if abs(d) < 0.005 else ("parent_sg" if d < 0 else "composition")] += 1
    out["summary"] = {"compositions": len(out["materials"]),
                      "with_relaxed_structure": sum("best" in m for m in out["materials"]),
                      "lower_energy_prompt (5 meV/atom tie band)": wins,
                      "median_parent_minus_composition_eV": round(statistics.median(diffs), 4) if diffs else None}
    json.dump(out, open(os.path.join(DATA, a.out), "w"), indent=1)
    print(out["summary"])


if __name__ == "__main__":
    main()
