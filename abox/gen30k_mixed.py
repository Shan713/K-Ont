"""30k generation, P3: mixed-site templates -- several metals ordered on one known framework (Step 65).

Design: docs/GEN30K_P3_MIXED_SITE.md. A host is a known single-metal Li-M-O oxide (M a transition metal or Al) from
the 10,267 template pool. Its cell is enlarged (Hermite-normal-form supercells, most compact shapes first), and the
system's metals are spread over the host's M sites; Li and O stay where they are. Orderings are ranked by Ewald
energy with one average oxidation state per element (E = q^T F q, F from unit charges, computed once per supercell),
searched by random starts + greedy pair swaps; the lowest-energy distinct orderings are kept. Cell volume is scaled
by atomic volume as in P2. No relaxation.

  hosts     pool -> data/gen30k/mixed_hosts.json (lowest-energy per reduced formula + space group, E_hull <= 0.05,
            primitive cell <= 24 atoms)
  plan      host-first formulas for every system: per host and supercell multiplicity (<= 48 atoms), every split of
            the metal sites among the system's metals (each >= 1) that charge-balances; per system --per-system
            formulas, most even split first, round-robin over host Li : M : O ratios. Ternary systems (one metal,
            Step 67): the host's metal is replaced by the system's metal, primitive cell only ("host_substitution")
  score     (OnT venv) P2's composition-only K-Ont retriever ranks the hosts that can carry each formula
  build     per formula, the top --hosts hosts by K-Ont (every host for ternaries), most compact supercell, --k
            lowest-Ewald distinct orderings that pass the bond floors of real Li-M-O oxides after at most a 10%
            stretch (gen30k_compositions.physical, Step 67) -> data/gen30k/mixed/<system>/*.cif + mixed_manifest.json
  backtest  rebuild the pool's known multi-metal Li-M-O oxides (<= 48 atoms) from single-metal hosts with the same
            Li : M : O ratio: found@k of Ewald-ranked orderings vs random orderings on the same host supercells
            (StructureMatcher 0.2 / 0.3 / 5 deg, unrelaxed). Acceptance, fixed before the run: Ewald > random at k = 3.
            Hosts are tried in composition-rule order here (no K-Ont); the K-Ont host ranking is a separate column
            once the score stage has run.

    .venv-crystallm/Scripts/python.exe abox/gen30k_mixed.py --stage hosts
    .venv-crystallm/Scripts/python.exe abox/gen30k_mixed.py --stage backtest [--limit N] [--workers 3]
    .venv-crystallm/Scripts/python.exe abox/gen30k_mixed.py --stage plan
    .venv/Scripts/python.exe           abox/gen30k_mixed.py --stage score
    .venv-crystallm/Scripts/python.exe abox/gen30k_mixed.py --stage build --hosts 3 --k 3 --workers 3
"""
import argparse
import itertools
import json
import math
import os
import random
import sys
import time
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "data", "gen30k")
HOSTS = os.path.join(OUT, "mixed_hosts.json")
STATES = {"Li": (1, 1), "Al": (3, 3), "Fe": (2, 3), "Mn": (2, 4), "Ni": (2, 3), "V": (3, 5),
          # backtest targets carry other metals too; common oxide states
          "Ti": (3, 4), "Cr": (2, 4), "Co": (2, 4), "Cu": (1, 2), "Zn": (2, 2), "Nb": (3, 5), "Mo": (3, 6),
          "Sc": (3, 3), "Y": (3, 3), "Zr": (4, 4), "Ta": (4, 5), "W": (4, 6), "Ru": (3, 5), "Rh": (3, 4),
          "Pd": (2, 4), "Ag": (1, 1), "Cd": (2, 2), "Hf": (4, 4), "Re": (4, 7), "Os": (4, 6), "Ir": (3, 5),
          "Pt": (2, 4), "Au": (1, 3), "Hg": (1, 2), "Tc": (4, 7)}


def is_metal(e):
    return e.symbol not in ("Li", "O") and (e.is_transition_metal or e.symbol == "Al")


def ratio_key(comp):
    """Li : M : O of a Li-M-O oxide, metals lumped, in lowest terms"""
    li = Fraction(comp["Li"]).limit_denominator(100)
    m = Fraction(sum(v for e, v in comp.items() if is_metal(e))).limit_denominator(100)
    o = Fraction(comp["O"]).limit_denominator(100)
    g = math.lcm(li.denominator, m.denominator, o.denominator)
    x = [int(li * g), int(m * g), int(o * g)]
    d = math.gcd(*x)
    return tuple(v // d for v in x)


# ------------------------------------------------------------------------------------------------ geometry
def hnf(m):
    for a in range(1, m + 1):
        if m % a:
            continue
        for d in range(1, m // a + 1):
            if (m // a) % d:
                continue
            f = m // (a * d)
            for b in range(d):
                for c in range(f):
                    for e in range(f):
                        yield [[a, 0, 0], [b, d, 0], [c, e, f]]


def supercells(prim, mult, n_shapes=3):
    """the n most compact supercells of multiplicity mult (ratio of longest to shortest reduced lattice vector)"""
    seen, out = set(), []
    for M in hnf(mult):
        s = prim.copy()
        s.make_supercell(M)
        lat = s.lattice.get_lll_reduced_lattice()
        abc = sorted(round(x, 2) for x in lat.abc)
        key = (tuple(abc), tuple(round(x, 1) for x in sorted(lat.angles)))
        if key in seen:
            continue
        seen.add(key)
        out.append((abc[2] / abc[0], M, s))
    out.sort(key=lambda t: t[0])
    return [(M, s) for _, M, s in out[:n_shapes]]


def ewald_matrix(s):
    """F such that the Ewald energy of charges q is q^T F q + sum_i p_i q_i^2 (neutral q)"""
    import numpy as np
    from pymatgen.analysis.ewald import EwaldSummation
    u = s.copy()
    u.add_oxidation_state_by_site([1.0] * len(u))
    ew = EwaldSummation(u, compute_forces=False)
    F = np.asarray(ew.real_space_energy_matrix) + np.asarray(ew.reciprocal_space_energy_matrix)
    p = np.asarray(ew.point_energy_matrix)
    return F + np.diag(p if p.ndim == 1 else np.diag(p))


def avg_charges(counts, fixed_charge):
    """one average oxidation state per metal, inside its allowed range, summing to fixed_charge (None if impossible)"""
    lo = sum(n * STATES[e][0] for e, n in counts.items())
    hi = sum(n * STATES[e][1] for e, n in counts.items())
    if not lo - 1e-9 <= fixed_charge <= hi + 1e-9:
        return None
    t = (fixed_charge - lo) / (hi - lo) if hi > lo else 0.0  # same fraction of each element's range
    return {e: STATES[e][0] + t * (STATES[e][1] - STATES[e][0]) for e in counts}


def order(host_sc, counts, n_keep, rng, starts=64, random_only=False):
    """decorate host_sc's metal sites with `counts` ({element: n}); returns [(energy, structure)] lowest first"""
    import numpy as np
    msites = [i for i, site in enumerate(host_sc) if is_metal(site.specie)]
    n_li = sum(1 for site in host_sc if site.specie.symbol == "Li")
    n_o = sum(1 for site in host_sc if site.specie.symbol == "O")
    if len(msites) != sum(counts.values()):
        return []
    q_avg = avg_charges(counts, 2 * n_o - n_li)
    if q_avg is None:
        return []
    els = sorted(counts)
    base = np.array([1.0 if site.specie.symbol == "Li" else -2.0 if site.specie.symbol == "O" else 0.0 for site in host_sc])
    F = None if random_only else ewald_matrix(host_sc)
    pool = [e for e in els for _ in range(counts[e])]

    def energy(dec):
        if F is None:
            return 0.0
        q = base.copy()
        q[msites] = [q_avg[e] for e in dec]
        return float(q @ F @ q)

    found = []
    for _ in range(starts if not random_only else n_keep * 4):
        dec = pool[:]
        rng.shuffle(dec)
        e = energy(dec)
        while F is not None:
            best = (0.0, None)
            for i, j in itertools.combinations(range(len(dec)), 2):
                if dec[i] == dec[j]:
                    continue
                dec[i], dec[j] = dec[j], dec[i]
                de = energy(dec) - e
                dec[i], dec[j] = dec[j], dec[i]
                if de < best[0] - 1e-9:
                    best = (de, (i, j))
            if best[1] is None:
                break
            i, j = best[1]
            dec[i], dec[j] = dec[j], dec[i]
            e += best[0]
        found.append((round(e, 6), tuple(dec)))
    uniq = sorted(set(found))
    out = []
    for e, dec in uniq:
        s = host_sc.copy()
        for k, el in zip(msites, dec):
            s.replace(k, el)
        out.append((e, s))
    return out


def scale_volume(s, host):
    from pymatgen.core import Element
    v = lambda st: sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in st.composition.items())
    s = s.copy()
    s.scale_lattice(s.volume * v(s) / v(host))
    return s


def dedup(cands, n_keep, sm):
    kept = []
    for e, s in cands:
        if not any(sm.fit(s, q) for _, q in kept):
            kept.append((e, s))
        if len(kept) >= n_keep:
            break
    return kept


# ------------------------------------------------------------------------------------------------ stages
def stage_hosts(a):
    from pymatgen.core import Composition, Structure
    from gen30k_templates import pool_rows
    import phaseC_generate as gen
    best = {}
    for i, r in pool_rows().items():
        c = Composition(r["formula"])
        if {e.symbol for e in c.elements} & {"Li", "O"} != {"Li", "O"}:
            continue
        ms = [e for e in c.elements if e.symbol not in ("Li", "O")]
        if len(ms) != 1 or not is_metal(ms[0]) or ms[0].symbol not in STATES:
            continue
        eh = r.get("e_above_hull")
        if eh is None or eh > a.max_ehull:
            continue
        key = (c.reduced_formula, r.get("space_group"))
        if key not in best or eh < best[key][0]:
            best[key] = (eh, i, r)
    hosts = []
    for (red, sg), (eh, i, r) in sorted(best.items()):
        try:
            prim = gen.reference_from_text(r["cif"]).get_primitive_structure()
        except Exception:
            continue
        if len(prim) > a.max_prim:
            continue
        comp = prim.composition
        hosts.append({"id": i, "formula": red, "space_group": sg, "e_above_hull": eh, "ratio": list(ratio_key(comp)),
                      "n_atoms": len(prim), "n_metal_sites": int(sum(v for e, v in comp.items() if is_metal(e))),
                      "metal": [e.symbol for e in comp.elements if is_metal(e)][0], "cif": prim.to(fmt="cif")})
    json.dump(hosts, open(HOSTS, "w"))
    from collections import Counter
    print(f"{len(hosts)} hosts; ratios: {Counter(tuple(h['ratio']) for h in hosts).most_common(10)}")


def _backtest_one(args):
    t, hosts, k_max, seed = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    import phaseC_generate as gen
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    try:
        ref = gen.reference_from_text(t["cif"])
    except Exception:
        return t["id"], None
    counts = {e.symbol: int(v) for e, v in ref.composition.items() if is_metal(e)}
    if any(e not in STATES for e in counts):
        return t["id"], None
    n_m = sum(counts.values())
    res = {"ewald": [], "random": []}
    rng = random.Random(seed)
    for h in hosts:
        prim = Structure.from_str(h["cif"], fmt="cif")
        if n_m % h["n_metal_sites"] or len(prim) * n_m // h["n_metal_sites"] != len(ref):
            continue
        for M, sc in supercells(prim, n_m // h["n_metal_sites"]):
            for how in ("ewald", "random"):
                cands = order(sc, counts, k_max, rng, random_only=(how == "random"))
                cands = dedup([(e, scale_volume(s, sc)) for e, s in cands], k_max, sm)
                res[how].extend((h["id"], e, bool(sm.fit(s, ref))) for e, s in cands)
    return t["id"], res


def stage_backtest(a):
    from collections import defaultdict
    from pymatgen.core import Composition
    from gen30k_templates import pool_rows
    hosts = json.load(open(HOSTS))
    by_ratio = defaultdict(list)
    for h in hosts:
        by_ratio[tuple(h["ratio"])].append(h)
    targets = []
    for i, r in pool_rows().items():
        c = Composition(r["formula"])
        if {e.symbol for e in c.elements} & {"Li", "O"} != {"Li", "O"}:
            continue
        ms = [e for e in c.elements if e.symbol not in ("Li", "O")]
        if len(ms) < 2 or not all(is_metal(e) for e in ms) or (r.get("n_atoms") or 99) > a.max_atoms:
            continue
        hs = sorted(by_ratio.get(ratio_key(c), []), key=lambda h: (h["e_above_hull"], h["id"]))[:a.hosts]
        if hs:
            targets.append((r, hs))
    random.Random(0).shuffle(targets)
    if a.limit:
        targets = targets[:a.limit]
    print(f"{len(targets)} backtest targets with a same-ratio host", flush=True)
    jobs = [({"id": r["id"], "cif": r["cif"]}, hs, a.k, n) for n, (r, hs) in enumerate(targets)]
    t0, results = time.time(), {}
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for n, (tid, res) in enumerate(ex.map(_backtest_one, jobs, chunksize=1)):
                results[tid] = res
                if (n + 1) % 25 == 0:
                    print(f"{n + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    else:
        for n, j in enumerate(jobs):
            tid, res = _backtest_one(j)
            results[tid] = res
            if (n + 1) % 10 == 0:
                print(f"{n + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rep = {}
    ok = {t: r for t, r in results.items() if r and (r["ewald"] or r["random"])}
    for how in ("ewald", "random"):
        for k in (1, 3, 10):
            # candidates in order: hosts in their given order, orderings by energy (random: sampling order)
            rep[f"{how}_found@{k}"] = round(100 * sum(any(x[2] for x in r[how][:k]) for r in ok.values()) / max(len(ok), 1), 1)
        rep[f"{how}_found_any"] = round(100 * sum(any(x[2] for x in r[how]) for r in ok.values()) / max(len(ok), 1), 1)
    rep["targets_scored"] = len(ok)
    rep["accept (ewald@3 > random@3)"] = rep["ewald_found@3"] > rep["random_found@3"]
    json.dump({"params": vars(a), "report": rep, "results": results}, open(os.path.join(OUT, "mixed_backtest.json"), "w"))
    print(json.dumps(rep, indent=1))


PLAN = os.path.join(OUT, "mixed_plan.json")
MSCORES = os.path.join(OUT, "mixed_scores.json")


def stage_plan(a):
    """host-first formulas for every multi-metal system: (host, multiplicity) -> metal splits that charge-balance"""
    from collections import defaultdict
    from pymatgen.core import Composition, Structure
    hosts = json.load(open(HOSTS))
    systems = list(json.load(open(os.path.join(OUT, "compositions.json")))["systems"])
    plan = {}
    for sysname in systems:
        metals = [e for e in sysname.split("-") if e not in ("Li", "O")]
        k = len(metals)
        forms = defaultdict(list)
        for h in hosts:
            h.setdefault("n_li", round(Structure.from_str(h["cif"], fmt="cif").composition["Li"]))
            for m in range(1, 9):
                n_at, n_m = h["n_atoms"] * m, h["n_metal_sites"] * m
                if n_at > a.max_atoms or (k == 1 and m > 1):  # one metal: a supercell adds nothing
                    break
                if n_m < k:
                    continue
                n_o = n_at - n_m - h["n_li"] * m
                for cut in itertools.combinations(range(1, n_m), k - 1):
                    parts = [b - a_ for a_, b in zip((0,) + cut, cut + (n_m,))]
                    counts = dict(zip(metals, parts))
                    if avg_charges(counts, 2 * n_o - h["n_li"] * m) is None:
                        continue
                    comp = Composition({"Li": h["n_li"] * m, **counts, "O": n_o})
                    forms[comp.reduced_formula].append({"host": h["id"], "mult": m, "counts": counts,
                                                        "evenness": max(parts) - min(parts), "n_atoms": n_at,
                                                        "ratio": h["ratio"]})
        # most even split first (high-entropy-like), then fewer atoms; round-robin over host ratios for variety
        by_ratio = defaultdict(list)
        for f, opts in forms.items():
            best = min(opts, key=lambda o: (o["evenness"], o["n_atoms"]))
            by_ratio[tuple(best["ratio"])].append((best["evenness"], best["n_atoms"], f))
        for v in by_ratio.values():
            v.sort()
        picked, ratios = [], sorted(by_ratio, key=lambda r: -len(by_ratio[r]))
        while len(picked) < a.per_system and any(by_ratio[r] for r in ratios):
            for r in ratios:
                if by_ratio[r] and len(picked) < a.per_system:
                    picked.append(by_ratio[r].pop(0)[2])
        plan[sysname] = {f: forms[f] for f in picked}
        print(f"{sysname:20s} {len(forms):6d} host-derived formulas, {len(picked)} picked", flush=True)
    json.dump(plan, open(PLAN, "w"))


def stage_score(a):
    """OnT venv: K-Ont composition-only retriever (as P2) scores every (formula, host) option"""
    import gzip
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from ont.hit import HierarchyTransformer
    from gen30k_templates import MODEL, DATA
    plan = json.load(open(PLAN))
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    full = {k.split("/")[-1]: v for k, v in vm.items()}
    for r in json.load(gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "rt", encoding="utf-8")):
        full.setdefault(r["id"], r["sentence"])
    m = HierarchyTransformer.from_pretrained(a.model or MODEL).to("cuda").eval()

    def emb(texts):
        out = []
        for i in range(0, len(texts), 64):
            f = m.tokenize(texts[i:i + 64])
            f = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in f.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out.append(F.normalize(m(f)["sentence_embedding"].float(), dim=-1))
        return torch.cat(out)
    hid = sorted({o["host"] for p in plan.values() for opts in p.values() for o in opts})
    he = dict(zip(hid, emb([full[h] for h in hid])))
    keys, texts = [], []
    for sysname, p in plan.items():
        for f, opts in p.items():
            o = min(opts, key=lambda o: (o["evenness"], o["n_atoms"]))
            keys.append(f"{sysname}|{f}")
            texts.append("\n".join([f"Material: {f} (gen30k)", "Type: substance", f"Formula: {f}",
                                    "Chemical system: " + "-".join(sorted(sysname.split("-"))),
                                    f"Sites: {o['n_atoms']}"]))
    qe = dict(zip(keys, emb(texts)))
    scores = {}
    for sysname, p in plan.items():
        for f, opts in p.items():
            k = f"{sysname}|{f}"
            scores[k] = {h: float(he[h] @ qe[k]) for h in {o["host"] for o in opts}}
    json.dump(scores, open(MSCORES, "w"))
    print(f"scored {len(scores)} formulas over {len(hid)} hosts")


def _build_one(args):
    sysname, f, opts, sc, n_hosts, n_ord, seed = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from gen30k_compositions import physical
    hosts = {h["id"]: h for h in json.load(open(HOSTS))}
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    rng = random.Random(seed)
    rows, kept = [], []
    ranked = sorted({o["host"] for o in opts}, key=lambda h: -sc[h])
    single = len(opts[0]["counts"]) == 1
    for hr, h in enumerate(ranked if single else ranked[:n_hosts], 1):
        o = min([o for o in opts if o["host"] == h], key=lambda o: (o["evenness"], o["n_atoms"]))
        prim = Structure.from_str(hosts[h]["cif"], fmt="cif")
        shapes = supercells(prim, o["mult"], n_shapes=1)
        if not shapes:
            continue
        M, scell = shapes[0]
        cands = order(scell, o["counts"], n_ord, rng, starts=a_starts(len(o["counts"])), random_only=single)
        cands = [(e, *physical(scale_volume(s, scell))) for e, s in cands]
        good = []
        for e, s, x in cands:
            if s is not None and not any(sm.fit(s, q) for _, q, _ in good):
                good.append((e, s, x))
            if len(good) >= n_ord:
                break
        for k, (e, s, x) in enumerate(good, 1):
            if any(sm.fit(s, q) for q in kept):
                continue
            kept.append(s)
            sub = os.path.join(OUT, "mixed", sysname)
            os.makedirs(sub, exist_ok=True)
            fn = f"{f}__h{hr}_o{k}__{h}.cif"
            CifWriter(s).write_file(os.path.join(sub, fn))
            rows.append({"file": f"mixed/{sysname}/{fn}", "system": sysname, "formula": f,
                         "source": "host_substitution" if single else "mixed_site_template",
                         "host_id": h, "host_formula": hosts[h]["formula"], "host_space_group": hosts[h]["space_group"],
                         "kont_host_rank": hr, "kont_score": round(sc[h], 4), "supercell": M, "ewald_rank": k,
                         "ewald_energy": e, "metal_counts": o["counts"], "cell_stretch": x, "n_atoms": len(s)})
    return rows


def a_starts(k):
    return 32 if k >= 4 else 64


def stage_build(a):
    plan, sc = json.load(open(PLAN)), json.load(open(MSCORES))
    only = set(a.systems.split(",")) if a.systems else None
    jobs = [(s, f, opts, sc[f"{s}|{f}"], a.hosts, a.k, n) for n, (s, p) in enumerate(sorted(plan.items()))
            for f, opts in sorted(p.items()) if not only or s in only]
    rows, t0 = [], time.time()
    man = os.path.join(OUT, "mixed_manifest.json")
    if only and os.path.exists(man):  # rebuild only these systems, keep the rest of the manifest
        rows = [r for r in json.load(open(man))["rows"] if r["system"] not in only]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(_build_one, jobs, chunksize=4)):
            rows.extend(r)
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(jobs)} formulas, {len(rows)} CIFs ({(time.time() - t0) / 60:.1f} min)", flush=True)
    per_sys = {}
    for x in rows:
        per_sys[x["system"]] = per_sys.get(x["system"], 0) + 1
    json.dump({"per_system": per_sys, "rows": rows}, open(man, "w"), indent=1)
    print(json.dumps({"mixed_cifs": len(rows), "per_system": per_sys}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["hosts", "backtest", "plan", "score", "build"])
    ap.add_argument("--per-system", type=int, default=200, help="plan: formulas per system")
    ap.add_argument("--model", default=None, help="score: retriever (default: P2's)")
    ap.add_argument("--max-ehull", type=float, default=0.05)
    ap.add_argument("--max-prim", type=int, default=24)
    ap.add_argument("--max-atoms", type=int, default=48)
    ap.add_argument("--hosts", type=int, default=5, help="backtest: hosts tried per target")
    ap.add_argument("--k", type=int, default=10, help="orderings kept per host supercell")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--systems", default="", help="build: only these systems (comma list), merged into the manifest")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"hosts": stage_hosts, "backtest": stage_backtest, "plan": stage_plan, "score": stage_score, "build": stage_build}[a.stage](a)


if __name__ == "__main__":
    main()
