"""30k generation, P4 (+ the P5 checks and P6 quota for CrystaLLM): fill every system up to its quota (Step 66).

Per system, quota = --target CIFs (default 970) minus the P2 templates and P3 mixed-site templates already written.
The KG fine-tuned CrystaLLM (Step 40 plain fine-tune `const`, no K-Ont input, as in the P8 pilot) then samples the
system's formulas -- P1's selected formulas and P3's host-derived formulas -- in rounds:
  - each round gives every still-active formula --samples prompts (formula x Z, Z rotating through 1, 2, 4 with
    <= 40 atoms, so later rounds ask for other cell sizes); formulas with the fewest kept CIFs go first;
  - prompts of equal token length share one batch (--batch sequences);
  - P5 checks on every sample: parses, right reduced composition, no two atoms < 0.5 A apart, and not a
    StructureMatcher duplicate (0.2 / 0.3 / 5 deg) of any template, mixed-site template or earlier sample of the
    same formula;
  - a formula is retired after --dry consecutive rounds with nothing new (CrystaLLM has run out of ideas for it).
Stops when the system reaches its quota or every formula is retired; the shortfall is logged.
Resumable: state after every batch in data/gen30k/crystallm/state.json; a restart continues where it stopped.

    .venv-crystallm/Scripts/python.exe -u abox/gen30k_crystallm.py [--systems Li-Fe-O,...] [--target 970]
Output: data/gen30k/crystallm/<system>/<formula>__z<Z>_r<round>_s<k>.cif + crystallm_manifest.json
"""
import argparse
import json
import os
import sys
import time
import warnings
import zlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "data", "gen30k")
CDIR = os.path.join(OUT, "crystallm")
STATE = os.path.join(CDIR, "state.json")


def z_options(formula, max_atoms=40):
    from pymatgen.core import Composition
    n = Composition(formula).num_atoms
    return [z for z in (1, 2, 4) if n * z <= max_atoms] or [1]


def check(cif, reduced):
    """P5 checks without the duplicate test: Structure or None"""
    import numpy as np
    import phaseC_generate as gen
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(cif), fmt="cif")
    except Exception:
        return None
    if s.composition.reduced_formula != reduced:
        return None
    if len(s) > 1 and (s.distance_matrix + 10 * np.eye(len(s))).min() < 0.5:
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--target", type=int, default=970)
    ap.add_argument("--samples", type=int, default=8, help="samples per formula per round")
    ap.add_argument("--batch", type=int, default=24, help="sequences per GPU batch (24 fits an 8 GB laptop GPU; 64 does not)")
    ap.add_argument("--dry", type=int, default=2, help="retire a formula after this many empty rounds")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--model", default="kgcond3_const.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    import kg_condition2 as k2
    import kg_condition3 as k3
    import phaseC_generate as gen

    comps = json.load(open(os.path.join(OUT, "compositions.json")))["systems"]
    tman = json.load(open(os.path.join(OUT, "templates_manifest.json")))["rows"]
    mman = json.load(open(os.path.join(OUT, "mixed_manifest.json")))["rows"] if os.path.exists(os.path.join(OUT, "mixed_manifest.json")) else []
    mplan = json.load(open(os.path.join(OUT, "mixed_plan.json"))) if os.path.exists(os.path.join(OUT, "mixed_plan.json")) else {}
    prior = {}
    for r in tman + mman:
        prior.setdefault((r["system"], r["formula"]), []).append(r["file"])
    systems = list(comps) if a.systems == "all" else a.systems.split(",")
    os.makedirs(CDIR, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}

    tok, dev = gen.CIFTokenizer(), "cuda"
    model = k2.Deep(gen.load_model(dev, "crystallm_v1_small"), 384, 8, gated=True).to(dev)
    model.load_state_dict(torch.load(os.path.join(gen.DATA, a.model), map_location=dev))
    model.eval()
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    zero = torch.zeros(384)

    for sysname in systems:
        forms = sorted({f["reduced"] for f in comps[sysname]["selected"]} | set(mplan.get(sysname, {})))
        st = state.setdefault(sysname, {"round": 0, "formulas": {f: {"kept": [], "sampled": 0, "dry": 0} for f in forms}})
        for f in forms:
            st["formulas"].setdefault(f, {"kept": [], "sampled": 0, "dry": 0})
        n_prior = sum(len(prior.get((sysname, f), [])) for f in st["formulas"])
        n_prior += sum(len(v) for (s_, f_), v in prior.items() if s_ == sysname and f_ not in st["formulas"])
        quota = max(0, a.target - n_prior)
        seen = {}  # formula -> structures to dedup against (loaded on first use)

        def known(f):
            if f not in seen:
                seen[f] = [Structure.from_file(os.path.join(OUT, p)) for p in prior.get((sysname, f), [])] + \
                          [Structure.from_file(os.path.join(CDIR, sysname, p)) for p in st["formulas"][f]["kept"]]
            return seen[f]

        kept = lambda: sum(len(v["kept"]) for v in st["formulas"].values())
        t0 = time.time()
        print(f"{sysname}: {n_prior} templates, quota for CrystaLLM {quota}, {kept()} already kept, {len(forms)} formulas",
              flush=True)
        while kept() < quota and st["round"] < a.max_rounds:
            active = [f for f, v in st["formulas"].items() if v["dry"] < a.dry]
            if not active:
                break
            rnd = st["round"]
            active.sort(key=lambda f: (len(st["formulas"][f]["kept"]), f))
            # only as many formulas as the remaining quota needs this round (avg yield ~0.6 unique per sample)
            need = quota - kept()
            active = active[:max(1, min(len(active), int(need / (0.6 * a.samples)) + 1))]
            groups = {}
            for f in active:
                zs = z_options(f)
                z = zs[rnd % len(zs)]
                from pymatgen.core import Composition
                cell = (Composition(f) * z).formula.replace(" ", "")
                p = gen.prompt_for(cell)
                groups.setdefault(len(tok.tokenize_cif(p)), []).append((f, z, p))
            per = max(1, a.batch // a.samples)
            got_round = {f: 0 for f in active}
            for length, items in sorted(groups.items()):
                for i in range(0, len(items), per):
                    b = items[i:i + per]
                    torch.manual_seed(a.seed + zlib.crc32(f"{sysname}|{rnd}|{length}|{i}".encode()))
                    cifs = k3.sample_group(model, tok, [p for _, _, p in b for _ in range(a.samples)],
                                           [zero] * (len(b) * a.samples), dev)
                    for j, (f, z, _) in enumerate(b):
                        v = st["formulas"][f]
                        for k in range(a.samples):
                            v["sampled"] += 1
                            s = check(cifs[j * a.samples + k], f)
                            if s is None or any(sm.fit(s, q) for q in known(f)):
                                continue
                            fn = f"{f}__z{z}_r{rnd}_s{k}.cif"
                            os.makedirs(os.path.join(CDIR, sysname), exist_ok=True)
                            s.to(filename=os.path.join(CDIR, sysname, fn))
                            v["kept"].append(fn)
                            seen[f].append(s)
                            got_round[f] += 1
                            if kept() >= quota:
                                break
                    json.dump(state, open(STATE, "w"))
                    if kept() >= quota:
                        break
                if kept() >= quota:
                    break
            for f, g in got_round.items():
                st["formulas"][f]["dry"] = 0 if g else st["formulas"][f]["dry"] + 1
            st["round"] += 1
            json.dump(state, open(STATE, "w"))
            print(f"  round {rnd}: {kept()}/{quota} kept, {len(active)} formulas sampled "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        st["quota"], st["shortfall"] = quota, max(0, quota - kept())
        json.dump(state, open(STATE, "w"))
        print(f"{sysname}: done, {kept()} CrystaLLM CIFs, shortfall {st['shortfall']}", flush=True)

    rows = []
    for sysname, st in state.items():
        for f, v in st["formulas"].items():
            for fn in v["kept"]:
                z, r, k = fn.rsplit("__", 1)[1][:-4].split("_")
                rows.append({"file": f"crystallm/{sysname}/{fn}", "system": sysname, "formula": f, "source": "crystallm",
                             "model": a.model, "z": int(z[1:]), "round": int(r[1:]), "sample": int(k[1:])})
    json.dump({"rows": rows, "per_system": {s: sum(len(v["kept"]) for v in st["formulas"].values()) for s, st in state.items()},
               "shortfall": {s: st.get("shortfall") for s, st in state.items()}},
              open(os.path.join(OUT, "crystallm_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
