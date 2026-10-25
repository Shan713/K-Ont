"""Relax candidate structures with MatterSim (runs in .venv-mattersim; called by kg_picker.py --stage refine).

MatterSim-v1.0.0-1M on the GPU, ASE BFGS on positions + cell (FrechetCellFilter), --fmax / --steps. Each job
also carries the reference structure: "right" = StructureMatcher(0.2, 0.3, 5) match of the relaxed OR the
unrelaxed candidate, as Step 48. Resumable: results are cached in the output JSON every 50 structures.

    .venv-mattersim/Scripts/python.exe abox/mattersim_relax.py jobs.json out.json [--steps 500 --fmax 0.05]
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--model", default="MatterSim-v1.0.0-1M.pth")
    a = ap.parse_args()
    import torch
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MatterSimCalculator(load_path=a.model, device=dev)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    jobs = json.load(open(a.jobs))
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [j for j in jobs if j["key"] not in out]
    print(f"{len(jobs)} structures, {len(todo)} to relax on {dev}", flush=True)
    t0 = time.time()
    for n, j in enumerate(todo):
        try:
            s0 = Structure.from_dict(j["structure"])
            atoms = AseAtomsAdaptor.get_atoms(s0)
            atoms.calc = calc
            BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=a.fmax, steps=a.steps)
            e = float(atoms.get_potential_energy()) / len(atoms)
            s1 = AseAtomsAdaptor.get_structure(atoms)
            ref = Structure.from_dict(j["ref"])
            right = bool(sm.fit(s1, ref) or sm.fit(s0, ref))
            out[j["key"]] = [round(e, 5), right]
        except Exception as ex:
            out[j["key"]] = None
            print(j["key"], "failed:", repr(ex)[:120], flush=True)
        if (n + 1) % 50 == 0 or n + 1 == len(todo):
            json.dump(out, open(a.out, "w"))
            print(f"  {n + 1}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
