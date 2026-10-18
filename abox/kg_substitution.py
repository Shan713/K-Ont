"""Substitution baselines and a better element placement for the template generator (BUILD_LOG Step 47).

Reviewer question for Step 41: "isn't this just element substitution (Hautier et al.)?". Hautier's data-mined
substitution model (pymatgen SubstitutionProbability, Hautier et al. 2011) gives the probability that the ions
of a known structure can be replaced by the target's ions. Two uses:
  #1 baseline   rank the same KG pool frameworks by that probability instead of K-Ont (the standard way to pick
                a prototype for substitution); template generator otherwise identical
  #3 placement  for any chosen framework, put each target element on the candidate element's sites by the
                most probable stoichiometry-preserving mapping (oxidation states from Composition.
                oxi_state_guesses), instead of Step 41's order (amount, then electronegativity)
Rankers x placements, found@1/2/3/5/8 on the 902 K-Ont-unseen targets, same matcher; paired bootstrap.

    .venv-crystallm/Scripts/python.exe abox/kg_substitution.py [--workers 2]
"""
import argparse
import itertools
import json
import os
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(DATA, "kg_substitution.json")
KS = (1, 2, 3, 5, 8)


def _oxi(formula):
    from pymatgen.core import Composition
    try:
        g = Composition(formula).oxi_state_guesses(max_sites=-20)
        return formula, ({k: float(v) for k, v in g[0].items()} if g else None)
    except Exception:
        return formula, None


def _species(formula, oxi):
    from pymatgen.core import Composition, Species
    c = Composition(formula).reduced_composition
    return [(e.symbol, c[e], Species(e.symbol, oxi[e.symbol]) if oxi else None) for e in c.elements]


def best_mapping(t_formula, c_formula, oxi, sp):
    """the stoichiometry-preserving element mapping candidate -> target with the highest Hautier probability;
    returns (probability, {candidate element: target element}); probability 0 when no oxidation guess"""
    ts, cs = _species(t_formula, oxi.get(t_formula)), _species(c_formula, oxi.get(c_formula))
    by_amt = {}
    for sym, amt, s in ts:
        by_amt.setdefault(round(amt, 6), []).append((sym, s))
    best = (-1.0, None)
    groups = {}
    for sym, amt, s in cs:
        groups.setdefault(round(amt, 6), []).append((sym, s))
    if sorted(map(len, by_amt.values())) != sorted(map(len, groups.values())) or set(by_amt) != set(groups):
        return 0.0, None
    keys = sorted(groups)
    for perms in itertools.product(*[itertools.permutations(by_amt[k]) for k in keys]):
        mapping, tl, cl = {}, [], []
        for k, perm in zip(keys, perms):
            for (csym, cspec), (tsym, tspec) in zip(groups[k], perm):
                mapping[csym] = tsym
                tl.append(tspec)
                cl.append(cspec)
        try:  # a species absent from Hautier's table (rare oxidation state) has no data: probability 0
            p = sp.cond_prob_list(tl, cl) if None not in tl and None not in cl else 0.0
        except (ZeroDivisionError, KeyError, ValueError):
            p = 0.0
        if p > best[0]:
            best = (p, mapping)
    return best


def _match(args):
    t, c, mapping = args
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element
    try:
        ref = gen.reference_from_text(t["cif"])
        s = gen.reference_from_text(c["cif"])
        old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        if mapping is None:
            mapping = {a.symbol: b.symbol for a, b in zip(k2._ordered(c["formula"]), k2._ordered(t["formula"]))}
        s.replace_species(mapping)
        new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.scale_lattice(s.volume * new / old)
        ok = s.composition.reduced_formula == ref.composition.reduced_formula and \
            StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(s, ref)
        return (t["id"], c["id"], "hautier" if args[2] else "order"), bool(ok)
    except Exception:
        return (t["id"], c["id"], "hautier" if args[2] else "order"), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import kg_pool
    from kg_condition2 import similarity
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    formulas = sorted({rows[i]["formula"] for t in targets for i in [t] + pool["candidates"][t]})
    oxi_p = os.path.join(DATA, "kg_oxi_guesses.json")
    oxi = json.load(open(oxi_p)) if os.path.exists(oxi_p) else {}
    todo = [f for f in formulas if f not in oxi]
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            oxi.update(dict(ex.map(_oxi, todo, chunksize=32)))
        json.dump(oxi, open(oxi_p, "w"))
    print(f"oxidation guesses: {sum(v is not None for v in oxi.values())}/{len(oxi)} formulas", flush=True)
    sp = SubstitutionProbability()
    hp, hmap = {}, {}
    for t in targets:
        for c in pool["candidates"][t]:
            p, m = best_mapping(rows[t]["formula"], rows[c]["formula"], oxi, sp)
            hp[(t, c)], hmap[(t, c)] = p, m
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))
    rank = {"K-Ont ranker": {t: [c for c, _ in top8["prompt+kg+ft_kont"][t]] for t in targets},
            "composition rule": {t: [c for c, _ in top8["composition rule"][t]] for t in targets},
            "Hautier substitution probability": {t: sorted(pool["candidates"][t], key=lambda c: (
                -hp[(t, c)], -similarity(rows[t]["formula"], rows[c]["formula"]), rows[c]["e_above_hull"] or 0))[:8]
                for t in targets}}
    jobs = sorted({(t, c, place) for d in rank.values() for t, cs in d.items() for c in cs[:8] for place in ("order", "hautier")})
    cache_p = os.path.join(DATA, "kg_substitution_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = [(rows[t], rows[c], hmap[(t, c)] if place == "hautier" else None) for t, c, place in jobs
            if f"{t}|{c}|{place}" not in cache and not (place == "hautier" and hmap[(t, c)] is None)]
    print(f"{len(jobs)} (target, framework, placement) combinations, {len(todo)} to match", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, ((t, c, place), ok) in enumerate(ex.map(_match, todo, chunksize=16)):
            cache[f"{t}|{c}|{place}"] = ok
            if (n + 1) % 2000 == 0:
                json.dump(cache, open(cache_p, "w"))
                print(f"  {n + 1}/{len(todo)}", flush=True)
    json.dump(cache, open(cache_p, "w"))

    def ok(t, c, place):
        if place == "hautier" and hmap[(t, c)] is None:
            place = "order"  # no oxidation guess: fall back to Step 41's placement
        return cache.get(f"{t}|{c}|{place}", False)
    found = {}
    for r, d in rank.items():
        for pl in ("order", "hautier"):
            found[f"{r} · {pl} placement"] = {k: {t: float(any(ok(t, c, pl) for c in d[t][:k])) for t in targets} for k in KS}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y, k):
        dd = {t: found[x][k][t] - found[y][k][t] for t in targets}
        bs = sorted(100 * sum(dd[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(dd.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n": len(targets), "oxidation_guess_coverage": round(100 * sum(oxi.get(rows[t]["formula"]) is not None for t in targets) / len(targets), 1),
           "found": {n: {f"@{k}": round(100 * sum(v.values()) / len(targets), 1) for k, v in f.items()} for n, f in found.items()}}
    kon, hau = "K-Ont ranker · order placement", "Hautier substitution probability · order placement"
    for k in KS:
        rep[f"K-Ont ranker - Hautier ranker (order placement) @{k}"] = diff(kon, hau, k)
        rep[f"K-Ont ranker - Hautier ranker (hautier placement) @{k}"] = diff("K-Ont ranker · hautier placement", "Hautier substitution probability · hautier placement", k)
        rep[f"K-Ont ranker: hautier - order placement @{k}"] = diff("K-Ont ranker · hautier placement", kon, k)
    json.dump({"report": rep, "found": found}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
