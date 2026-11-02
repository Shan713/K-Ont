"""30k generation, P8 pilot: CrystaLLM throughput and yield on two systems before the full run (Step 63).

For --per-system formulas of each pilot system (from data/gen30k/compositions.json, P1) the KG fine-tuned CrystaLLM
(Step 40 plain fine-tune, `const`: no K-Ont input; zero conditioning vector) writes --samples CIFs per formula. The
prompt is the formula times Z, with Z the largest of 1, 2, 4 that keeps the cell at <= 40 atoms. Measured:
  speed     seconds per sample, CIFs per hour (this GPU)
  yield     parses as a structure / right reduced composition / no two atoms closer than 0.5 A / unique within
            the formula (StructureMatcher 0.2 / 0.3 / 5 deg) -- the P5 checks, no relaxation
These numbers size the full run's CrystaLLM share (30k minus what the templates give) on the workstation.

    .venv-crystallm/Scripts/python.exe abox/gen30k_pilot.py [--systems Li-Fe-O,Li-Al-Fe-Mn-Ni-O]
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

import kg_condition2 as k2  # noqa: E402
import kg_condition3 as k3  # noqa: E402
import phaseC_generate as gen  # noqa: E402

OUT = os.path.join(HERE, "data", "gen30k")


def cell_for(formula, max_atoms=40):
    from pymatgen.core import Composition
    c = Composition(formula)
    z = max([z for z in (1, 2, 4) if c.num_atoms * z <= max_atoms] or [1])
    return (c * z).formula.replace(" ", ""), z


def check(cif, reduced):
    """(parsed, right composition, no clash, structure or None) -- the P5 checks"""
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(cif), fmt="cif")
    except Exception:
        return False, False, False, None
    right = s.composition.reduced_formula == reduced
    try:
        dm = s.distance_matrix
        clash = (dm + 10 * __import__("numpy").eye(len(s))).min() < 0.5
    except Exception:
        clash = True
    return True, right, not clash, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="Li-Fe-O,Li-Al-Fe-Mn-Ni-O")
    ap.add_argument("--per-system", type=int, default=10)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--model", default="kgcond3_const.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from pymatgen.analysis.structure_matcher import StructureMatcher
    comps = json.load(open(os.path.join(OUT, "compositions.json")))
    tok, dev = gen.CIFTokenizer(), "cuda"
    m = k2.Deep(gen.load_model(dev, "crystallm_v1_small"), 384, 8, gated=True).to(dev)
    m.load_state_dict(torch.load(os.path.join(gen.DATA, a.model), map_location=dev))
    m.eval()
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    report = {"params": vars(a), "gpu": torch.cuda.get_device_name(0), "systems": {}}
    gdir = os.path.join(OUT, "pilot_cifs")
    for sysname in a.systems.split(","):
        rows = comps["systems"][sysname]["selected"][:a.per_system]
        tot = {"samples": 0, "parsed": 0, "right_composition": 0, "no_clash": 0, "unique_valid": 0}
        t0 = time.time()
        for n, r in enumerate(rows):
            cell, z = cell_for(r["formula"])
            torch.manual_seed(a.seed + n)
            prompts = [gen.prompt_for(cell)] * a.samples
            cifs = k3.sample_group(m, tok, prompts, [torch.zeros(384)] * a.samples, dev)
            kept = []
            for k, cif in enumerate(cifs):
                p, right, ok, s = check(cif, r["reduced"])
                tot["samples"] += 1
                tot["parsed"] += p
                tot["right_composition"] += p and right
                tot["no_clash"] += p and right and ok
                if p and right and ok and not any(sm.fit(s, q) for q in kept):
                    kept.append(s)
                    d = os.path.join(gdir, sysname)
                    os.makedirs(d, exist_ok=True)
                    s.to(filename=os.path.join(d, f"{r['reduced']}_z{z}_s{k}.cif"))
            tot["unique_valid"] += len(kept)
        secs = time.time() - t0
        tot.update({"formulas": len(rows), "seconds": round(secs, 1), "sec_per_sample": round(secs / max(tot["samples"], 1), 2),
                    "valid_unique_per_formula": round(tot["unique_valid"] / len(rows), 2),
                    "valid_unique_cifs_per_hour": round(tot["unique_valid"] * 3600 / secs, 1)})
        report["systems"][sysname] = tot
        print(sysname, json.dumps(tot), flush=True)
    json.dump(report, open(os.path.join(OUT, "pilot_crystallm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
