"""30k generation, P1: charge-balanced Li-oxide formulas for the 31 chemical systems (Step 63).

The systems are Li + 1-5 of {Al, Fe, Mn, Ni, V} + O (5 ternary, 10 quaternary, 10 quinary, 5 senary, 1 septenary).
A formula is kept when
  - it contains every element of its system, in lowest terms (gcd of the counts = 1);
  - it has at most --max-atoms atoms per formula unit, Li count <= --max-li, every other metal count <= 8 / 6 / 4
    for ternary / quaternary / larger systems (ternary chemistry is small, so it is allowed larger cells);
  - it can be charge balanced with O2- and the cation states Li+, Al3+, Fe2+/3+, Mn2+/3+/4+, Ni2+/3+, V3+/4+/5+,
    mixed valence allowed (Mn3+/4+ as in LiMn2O4), i.e. 2*n(O) lies in [sum n*min state, sum n*max state];
  - Li is 15-75% of the cations (a Li-ion cathode-like range; pure Li oxides and Li-free oxides are out).
Each formula is tagged with its anonymous formula (ABC2 ...) and how many of the 10,267 template-pool materials
(5,000 KG + 5,267 Materials Project extension) share it ("exact_templates"), and how many of those are oxides with
O on the same-count site ("oxide_templates": LiFePO4 is an exact ABCD4 template for LiAlFeO4 but not an oxide one).

Selection per system (--per-system formulas): formulas with an oxide template first, then any exact template, then by fewer atoms, then by
a Li fraction near 1/3 - 1/2; ties broken by formula so the list is reproducible.

    .venv-crystallm/Scripts/python.exe abox/gen30k_compositions.py           -> data/gen30k/compositions.json
    ../battGPT/.venv/Scripts/python.exe abox/gen30k_compositions.py --stage mp-flag   (adds in_mp per formula)
"""
import argparse
import gzip
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from functools import reduce

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(HERE, "data", "gen30k")
METALS = ["Al", "Fe", "Mn", "Ni", "V"]
STATES = {"Li": (1, 1), "Al": (3, 3), "Fe": (2, 3), "Mn": (2, 4), "Ni": (2, 3), "V": (3, 5)}


def systems():
    out = []
    for k in range(1, 6):
        for combo in itertools.combinations(METALS, k):
            out.append(["Li", *combo, "O"])
    return out


def balanced(counts):
    lo = sum(n * STATES[e][0] for e, n in counts.items() if e != "O")
    hi = sum(n * STATES[e][1] for e, n in counts.items() if e != "O")
    return lo <= 2 * counts["O"] <= hi


def enumerate_system(els, a):
    ms = [e for e in els if e not in ("Li", "O")]
    max_metal = {1: 8, 2: 6}.get(len(ms), 4)
    found = []
    for li in range(1, a.max_li + 1):
        for mc in itertools.product(range(1, max_metal + 1), repeat=len(ms)):
            n_cat = li + sum(mc)
            if n_cat > a.max_atoms - 1 or not (0.15 <= li / n_cat <= 0.75):
                continue
            lo = li + sum(n * STATES[e][0] for e, n in zip(ms, mc))
            hi = li + sum(n * STATES[e][1] for e, n in zip(ms, mc))
            for o in range(math.ceil(lo / 2), hi // 2 + 1):
                if n_cat + o > a.max_atoms:
                    break
                counts = {"Li": li, **dict(zip(ms, mc)), "O": o}
                if reduce(math.gcd, counts.values()) != 1 or not balanced(counts):
                    continue
                found.append(counts)
    return found


def oxide_key(comp):
    """anonymous formula + the reduced amount of O, so an oxide template has O on the matching site"""
    red = comp.reduced_composition
    return f"{comp.anonymized_formula}|O{red['O']:g}"



# Step 67: shortest bonds in the 451 Materials Project Li-M-O oxides of the pool (cation-O >= 1.71 A in 95%,
# O-O >= 2.52, cation-cation >= 2.43); with these floors 92.5% of the real structures pass
FLOORS = {"cation-O": 1.70, "O-O": 2.30, "cation-cation": 2.30}


def physical(s, max_stretch=1.10):
    """(structure, stretch) when every shortest bond clears FLOORS, after an isotropic stretch of the cell of at most
    max_stretch (lengths) if needed; (None, needed stretch) otherwise. Unrelaxed cells are scaled by atomic volume
    only, so a few % is within that estimate's error; local distortions (B-O / P-O-like frameworks) are not."""
    import numpy as np
    d = s.distance_matrix + 99 * np.eye(len(s))
    o = np.array([sp.symbol == "O" for sp in s.species])
    c = ~o
    need = 0.0
    if c.any() and o.any():
        need = max(need, FLOORS["cation-O"] / d[np.ix_(c, o)].min())
    if o.sum() > 1:
        need = max(need, FLOORS["O-O"] / d[np.ix_(o, o)].min())
    if c.sum() > 1:
        need = max(need, FLOORS["cation-cation"] / d[np.ix_(c, c)].min())
    if need <= 1.0:
        return s, 1.0
    if need > max_stretch:
        return None, round(need, 3)
    s = s.copy()
    s.scale_lattice(s.volume * (need * 1.001) ** 3)
    return s, round(need * 1.001, 3)

def formula_str(counts):
    return "".join(f"{e}{'' if n == 1 else n}" for e, n in counts.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["enumerate", "mp-flag"], default="enumerate")
    ap.add_argument("--max-atoms", type=int, default=30)
    ap.add_argument("--max-li", type=int, default=8)
    ap.add_argument("--per-system", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "compositions.json")

    if a.stage == "mp-flag":  # battGPT venv: mp_api + the key from battGPT/.env (never printed)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "battGPT"))
        from mp_api.client import MPRester
        from pipeline.config import PipelineConfig
        from pymatgen.core import Composition
        d = json.load(open(path))
        names = sorted({f["reduced"] for s in d["systems"].values() for f in s["selected"]})
        in_mp = set()
        with MPRester(PipelineConfig().mp_api_key, mute_progress_bars=True) as mpr:
            for i in range(0, len(names), 100):
                docs = mpr.materials.summary.search(formula=names[i:i + 100], fields=["formula_pretty"])
                in_mp |= {Composition(x.formula_pretty).reduced_formula for x in docs}
        for s in d["systems"].values():
            for f in s["selected"]:
                f["in_mp"] = f["reduced"] in in_mp
            s["selected_in_mp"] = sum(f["in_mp"] for f in s["selected"])
        d["mp_flagged"] = True
        json.dump(d, open(path, "w"), indent=1)
        print({k: f'{s["selected_in_mp"]}/{len(s["selected"])}' for k, s in d["systems"].items()})
        return

    from pymatgen.core import Composition
    pool = [r["formula"] for s in ("train", "val", "test")
            for r in json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{s}.json.gz"), "rt", encoding="utf-8"))]
    pool += [r["formula"] for r in json.load(gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "rt", encoding="utf-8"))]
    pool_anon, pool_oxide = Counter(), Counter()
    for f in pool:
        c = Composition(f)
        pool_anon[c.anonymized_formula] += 1
        if "O" in c:
            pool_oxide[oxide_key(c)] += 1
    pool_red = {Composition(f).reduced_formula for f in pool}

    out = {"params": vars(a), "oxidation_states": STATES, "template_pool_size": len(pool), "systems": {}}
    for els in systems():
        name = "-".join(els)
        rows = []
        for counts in enumerate_system(els, a):
            comp = Composition(counts)
            anon = comp.anonymized_formula
            n_cat = sum(n for e, n in counts.items() if e != "O")
            rows.append({"formula": formula_str(counts), "reduced": comp.reduced_formula, "n_atoms": int(comp.num_atoms),
                         "li_fraction": round(counts["Li"] / n_cat, 3), "anonymous": anon,
                         "exact_templates": pool_anon.get(anon, 0), "oxide_templates": pool_oxide.get(oxide_key(comp), 0),
                         "in_template_pool": comp.reduced_formula in pool_red})
        rows.sort(key=lambda r: (r["oxide_templates"] == 0, r["exact_templates"] == 0, r["n_atoms"],
                                 abs(r["li_fraction"] - 0.42), r["formula"]))
        sel = rows[:a.per_system]
        out["systems"][name] = {
            "n_elements": len(els), "valid_formulas": len(rows),
            "with_exact_template": sum(r["exact_templates"] > 0 for r in rows),
            "with_oxide_template": sum(r["oxide_templates"] > 0 for r in rows),
            "selected": sel, "selected_with_oxide_template": sum(r["oxide_templates"] > 0 for r in sel)}
        S = out["systems"][name]
        print(f"{name:20s} valid {len(rows):5d}  exact-template {S['with_exact_template']:4d}  oxide-template "
              f"{S['with_oxide_template']:4d}  selected {len(sel)} ({S['selected_with_oxide_template']} with oxide template)")
    json.dump(out, open(path, "w"), indent=1)
    by_n = defaultdict(lambda: [0, 0])
    for s in out["systems"].values():
        by_n[s["n_elements"]][0] += len(s["selected"])
        by_n[s["n_elements"]][1] += s["selected_with_oxide_template"]
    print("selected / with oxide template, by number of elements:", dict(by_n))


if __name__ == "__main__":
    main()
