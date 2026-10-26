"""Oxidation-state aware site assignment for K-Ont templates (BUILD_LOG Step 59).

Step 41's templates put the target's elements on the template's sites by a fixed rule: elements sorted by amount,
then electronegativity (kg_condition2._ordered), cell scaled by neutral atomic radii. That can put an ion on a site
whose original occupant had a different charge or size. Here, per template:
  1. template oxidation states from its real structure (pymatgen BVAnalyzer, bond valence; fallback: the most
     probable charge-balanced guess of its composition), averaged per element;
  2. target oxidation states: the most probable charge-balanced guesses of its composition (up to 5);
  3. every legal element mapping: only elements with the same amount in the reduced formula may swap, so the
     stoichiometry stays exact (the anonymous formula is shared);
  4. cost of a mapping = 10 x cation<->anion sign mismatches + |charge difference| + 2 x relative ionic-radius
     mismatch, + 0.1 x rank of the target guess; the cheapest wins, ties keep Step 41's mapping;
  5. cell scaled by the assigned ionic radii (--scale ionic) or by atomic radii as before (--scale atomic).
Evaluation (both rules built by the same code here, per pair): the same 902 K-Ont-unseen targets, the same K-Ont ranker top-8 candidates as Step 41, the same matcher
(StructureMatcher 0.2 / 0.3 / 5, unrelaxed); found@k and paired bootstrap against Step 41's rule.

    .venv-crystallm/Scripts/python.exe abox/kg_os_template.py --workers 3
"""
import argparse
import itertools
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(DATA, "kg_os_template.json")
RANKER = "prompt+kg+ft_kont"


def _avg_os(structure_or_comp):
    """element symbol -> average oxidation state (bond valence on the structure, else composition guess)."""
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.core import Structure
    if isinstance(structure_or_comp, Structure):
        try:
            v = BVAnalyzer().get_valences(structure_or_comp)
            acc = {}
            for site, val in zip(structure_or_comp, v):
                acc.setdefault(site.specie.symbol, []).append(val)
            return {e: sum(x) / len(x) for e, x in acc.items()}
        except Exception:
            structure_or_comp = structure_or_comp.composition
    g = structure_or_comp.oxi_state_guesses(max_sites=-50)
    return dict(g[0]) if g else None


def _radius(sym, ox):
    from pymatgen.core import Element, Species
    try:
        r = Species(sym, int(round(ox))).ionic_radius
        if r:
            return float(r)
    except Exception:
        pass
    return float(Element(sym).atomic_radius or 1.5)


def os_mapping(template_structure, template_formula, target_formula, max_guesses=5):
    """-> (mapping {template element: target element}, target oxidation states, changed vs Step 41's rule)"""
    import kg_condition2 as k2
    from pymatgen.core import Composition
    rule = {a.symbol: b.symbol for a, b in zip(k2._ordered(template_formula), k2._ordered(target_formula))}
    tos = _avg_os(template_structure)
    ct, cg = Composition(template_formula).reduced_composition, Composition(target_formula).reduced_composition
    guesses = cg.oxi_state_guesses(max_sites=-50)[:max_guesses]
    if not tos or not guesses:
        return rule, None, False
    by_amount_t, by_amount_g = {}, {}
    for e, n in ct.items():
        by_amount_t.setdefault(round(n, 6), []).append(e.symbol)
    for e, n in cg.items():
        by_amount_g.setdefault(round(n, 6), []).append(e.symbol)
    if sorted(by_amount_t) != sorted(by_amount_g):
        return rule, None, False
    groups = [(by_amount_t[k], by_amount_g[k]) for k in sorted(by_amount_t)]
    best = None
    for gi, g in enumerate(guesses):
        for perms in itertools.product(*[itertools.permutations(tg) for _, tg in groups]):
            m = {a: b for (ta, _), p in zip(groups, perms) for a, b in zip(ta, p)}
            cost = 0.1 * gi
            for a, b in m.items():
                oa, ob = tos.get(a, 0.0), float(g[b])
                if oa * ob < 0:
                    cost += 10
                cost += abs(oa - ob)
                ra, rb = _radius(a, oa), _radius(b, ob)
                cost += 2 * abs(ra - rb) / ra
            key = (round(cost, 6), m != rule)  # ties keep the old rule
            if best is None or key < best[0]:
                best = (key, m, dict(g))
    m = best[1]
    return m, best[2], m != rule


def build(template_cif, template_formula, target_formula, scale, mapping=None, target_os=None):
    import phaseC_generate as gen
    from pymatgen.core import Element
    s = gen.reference_from_text(template_cif)
    if mapping is None:
        mapping, target_os, _ = os_mapping(s, template_formula, target_formula)
    if scale == "ionic" and target_os:
        tos = _avg_os(s) or {}
        old = sum(n * _radius(e.symbol, tos.get(e.symbol, 0)) ** 3 for e, n in s.composition.items())
        new = sum(n * _radius(mapping[e.symbol], target_os.get(mapping[e.symbol], 0)) ** 3 for e, n in s.composition.items())
    else:
        old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        new = sum(n * float(Element(mapping[e.symbol]).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.replace_species(mapping)
    s.scale_lattice(s.volume * new / old)
    return s


def _job(args):
    t, c = args
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    import kg_condition2 as k2
    out = {"changed": False, "rule": False, "atomic": False, "ionic": False}
    try:
        ref = gen.reference_from_text(t["cif"])
        tmpl = gen.reference_from_text(c["cif"])
        rule = {a.symbol: b.symbol for a, b in zip(k2._ordered(c["formula"]), k2._ordered(t["formula"]))}
        s = build(c["cif"], c["formula"], t["formula"], "atomic", rule, None)  # Step 41, same code path
        out["rule"] = bool(s.composition.reduced_formula == ref.composition.reduced_formula and sm.fit(s, ref))
        m, tos, changed = os_mapping(tmpl, c["formula"], t["formula"])
        out["changed"] = changed
        for scale in ("atomic", "ionic"):
            s = build(c["cif"], c["formula"], t["formula"], scale, m, tos)
            out[scale] = bool(s.composition.reduced_formula == ref.composition.reduced_formula and sm.fit(s, ref))
    except Exception:
        pass
    return f"{t['id']}|{c['id']}", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    import kg_pool
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    top = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))[RANKER]
    res = json.load(open(OUT))["pairs"] if os.path.exists(OUT) else {}
    jobs = [(rows[t], rows[c]) for t in targets for c, _ in top[t][:8] if "rule" not in res.get(f"{t}|{c}", {})]
    print(f"{len(jobs)} substitutions to build and match", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (k, v) in enumerate(ex.map(_job, jobs, chunksize=16)):
            res[k] = v
            if (n + 1) % 500 == 0:
                json.dump({"pairs": res}, open(OUT, "w"))
                print(f"  {n + 1}/{len(jobs)}", flush=True)
    keys = {t: [f"{t}|{c}" for c, _ in top[t][:8]] for t in targets}
    arm = {"Step 41 rule": lambda k: res[k]["rule"],
           "OS-aware, atomic-radius cell": lambda k: res[k]["atomic"],
           "OS-aware, ionic-radius cell": lambda k: res[k]["ionic"]}
    found = {n: {kk: {t: float(any(f(k) for k in keys[t][:kk])) for t in targets} for kk in (1, 2, 3, 5, 8)}
             for n, f in arm.items()}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y, kk):
        d = {t: found[x][kk][t] - found[y][kk][t] for t in targets}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    allk = [k for t in targets for k in keys[t]]
    ch = [k for k in allk if res[k]["changed"]]
    rep = {"n_targets": len(targets), "substitutions": len(allk), "mapping_changed": len(ch),
           "of changed: right before / OS-aware atomic / ionic": [sum(res[k]["rule"] for k in ch),
                                                                  sum(res[k]["atomic"] for k in ch),
                                                                  sum(res[k]["ionic"] for k in ch)],
           "found@k": {n: {f"@{kk}": round(100 * sum(v.values()) / len(targets), 1) for kk, v in f.items()}
                       for n, f in found.items()}}
    for n in ("OS-aware, atomic-radius cell", "OS-aware, ionic-radius cell"):
        rep[f"{n} - Step 41 rule"] = {f"@{kk}": diff(n, "Step 41 rule", kk) for kk in (1, 3, 8)}
    json.dump({"report": rep, "pairs": res}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
