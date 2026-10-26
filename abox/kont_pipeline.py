"""Staged K-Ont discovery pipeline with provenance and a stage-wise ablation (BUILD_LOG Step 61).

Stages for one composition, each measured on its own:
  1 retrieve   K-Ont ranker top-3 KG frameworks (kg_rerank_top8, the Step 41/48 templates)
  2 generate   3 element-substituted templates + 5 CrystaLLM samples (Step 40 plain fine-tune, `const`)
  3 validate   the same checks for every candidate, before any relaxation: parses; composition = target;
               no clash (every pair distance >= 0.6 x the sum of the two atomic radii); volume per atom within
               0.4-4 x the composition's mean atomic volume
  4 rank       candidate order: templates by K-Ont rank, CrystaLLM samples in sampling order; or the K-Ont checker
               (Step 60) score
  5 screen     CHGNet energy per atom after the Step 48 relaxation (40 steps, cache kg_pipeline_energies_s40);
               optional knowledge tie-break: among candidates within --tie meV/atom of the lowest energy, the
               best-ranked one wins
  6 output     ranked candidates with provenance: source (template from <KG id, formula>, K-Ont rank / score, or
               CrystaLLM sample k), validity checks, energy, pick
Ablation: the Step 48 subset (300 K-Ont-unseen targets with CHGNet energies), single answer right
(StructureMatcher 0.2/0.3/5, the cached Step 48 labels); paired bootstrap (5,000) vs the previous stage and vs
CrystaLLM x8 + CHGNet.

    .venv-crystallm/Scripts/python.exe abox/kont_pipeline.py --stage validate
    .venv-crystallm/Scripts/python.exe abox/kont_pipeline.py --stage ablation
    .venv-crystallm/Scripts/python.exe abox/kont_pipeline.py --stage report --target mp-10180 [--target ...]
"""
import argparse
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
GEN = os.path.join(DATA, "kgcond3_gen", "const")
ENERGY = os.path.join(DATA, "kg_pipeline_energies_s40.json")
VALID = os.path.join(DATA, "kont_pipeline_valid.json")
OUT = os.path.join(DATA, "kont_pipeline_ablation.json")
REPORTS = os.path.join(DATA, "kont_pipeline_reports")


def subset():
    return sorted({k.split("|")[0] for k in json.load(open(ENERGY))})


def candidates(t, top8, n_t=3, n_c=5):
    """Stage 1 + 2: candidate keys with provenance (the Step 48 key scheme)."""
    out = [{"key": f"{t}|T|{c}", "source": "template", "from": c, "kont_rank": i + 1, "kont_score": round(float(sc), 4)}
           for i, (c, sc) in enumerate(top8[t][:n_t])]
    out += [{"key": f"{t}|C|{k}", "source": "CrystaLLM", "sample": k} for k in range(n_c)]
    return out


def build(cand, t, rows):
    """The unrelaxed structure of a candidate, exactly as Step 48 builds it."""
    import kg_picker
    if cand["source"] == "template":
        return kg_picker.build("T", rows[t]["formula"], rows[cand["from"]])
    return kg_picker.build("C", rows[t]["formula"], os.path.join(GEN, t, f"s_{cand['sample']}.cif"))


def checks(s, target_formula):
    """Stage 3: validity checks on an unrelaxed structure."""
    import numpy as np
    from pymatgen.core import Composition, Element
    r = {"composition": s.composition.reduced_formula == Composition(target_formula).reduced_formula}
    rad = np.array([float(Element(site.specie.symbol).atomic_radius or 1.5) for site in s])
    d = s.distance_matrix + np.eye(len(s)) * 1e9
    r["no_clash"] = bool((d >= 0.6 * (rad[:, None] + rad[None, :])).all())
    v_atoms = float(np.mean(4 / 3 * np.pi * rad ** 3))
    ratio = s.volume / len(s) / v_atoms
    r["volume_ok"] = bool(0.4 <= ratio <= 4.0)
    r["valid"] = all(r.values())
    r["volume_ratio"] = round(ratio, 3)
    return r


def stage_validate(a):
    import kg_pool
    pool, rows = kg_pool.load()
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    out = json.load(open(VALID)) if os.path.exists(VALID) else {}
    T = subset()
    for n, t in enumerate(T):
        for cd in candidates(t, top8, 3, 8):
            if cd["key"] in out:
                continue
            try:
                out[cd["key"]] = checks(build(cd, t, rows), rows[t]["formula"])
            except Exception as e:
                out[cd["key"]] = {"composition": False, "no_clash": False, "volume_ok": False, "valid": False,
                                  "error": repr(e)[:80]}
        if n % 50 == 0:
            json.dump(out, open(VALID, "w"))
            print(f"{n + 1}/{len(T)}", flush=True)
    json.dump(out, open(VALID, "w"))
    print("validated", len(out), "candidates")


def pick(cands, E, V, use_valid=False, order=None, tie=0.0):
    """Stage 5: lowest CHGNet energy; with tie > 0, the best-ranked candidate within tie meV/atom of the lowest."""
    pool = [c for c in cands if E.get(c["key"]) and (not use_valid or V.get(c["key"], {}).get("valid"))]
    if not pool:
        return None
    e0 = min(E[c["key"]][0] for c in pool)
    if tie > 0:
        near = [c for c in pool if 1000 * (E[c["key"]][0] - e0) <= tie]
        rank = {c["key"]: i for i, c in enumerate(order or cands)}
        return min(near, key=lambda c: rank.get(c["key"], 99))
    return min(pool, key=lambda c: E[c["key"]][0])


def stage_ablation(a):
    E = json.load(open(ENERGY))
    V = json.load(open(VALID))
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    chk = json.load(open(os.path.join(DATA, "kont_checker_scores.json")))["kont"] if os.path.exists(
        os.path.join(DATA, "kont_checker_scores.json")) else {}
    T = subset()
    right = lambda c: float(bool(c and E.get(c["key"]) and E[c["key"]][1]))
    arms, ceil = {}, {}

    def C8(t):
        return [{"key": f"{t}|C|{k}", "source": "CrystaLLM", "sample": k} for k in range(8)]

    def kont_order(t, cs):
        """templates by K-Ont rank first, then CrystaLLM samples by the Step 60 checker score."""
        sc = chk.get(t, [None] * 8)
        tm = [c for c in cs if c["source"] == "template"]
        cl = sorted([c for c in cs if c["source"] == "CrystaLLM"], key=lambda c: -(sc[c["sample"]] or -9))
        return tm + cl
    stages = [
        ("A  CrystaLLM, first sample (no knowledge, no physics)", lambda t: (C8(t)[:1], lambda cs: cs[0])),
        ("B  CrystaLLM x8 + CHGNet pick", lambda t: (C8(t), lambda cs: pick(cs, E, V))),
        ("C  + K-Ont retrieval: 3 templates + 5 CrystaLLM, CHGNet pick", lambda t: (candidates(t, top8), lambda cs: pick(cs, E, V))),
        ("D  + validation filter before the pick", lambda t: (candidates(t, top8), lambda cs: pick(cs, E, V, use_valid=True))),
        ("E  + knowledge tie-break (K-Ont order within 5 meV/atom)",
         lambda t: (candidates(t, top8), lambda cs: pick(cs, E, V, True, kont_order(t, cs), 5.0))),
        ("E' + knowledge tie-break (K-Ont order within 10 meV/atom)",
         lambda t: (candidates(t, top8), lambda cs: pick(cs, E, V, True, kont_order(t, cs), 10.0))),
        ("K-Ont template top-1 alone (knowledge, no physics)", lambda t: (candidates(t, top8)[:1], lambda cs: cs[0])),
    ]
    for name, f in stages:
        got, cg = {}, {}
        for t in T:
            cs, chooser = f(t)
            got[t] = right(chooser(cs))
            cg[t] = float(any(E.get(c["key"]) and E[c["key"]][1] for c in cs))
        arms[name], ceil[name] = got, cg
    rng = random.Random(0)
    B = [[rng.choice(T) for _ in T] for _ in range(5000)]

    def diff(x, y):
        d = {t: arms[x][t] - arms[y][t] for t in T}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(T), 1), round(bs[125], 1), round(bs[4874], 1)]
    names = [n for n, _ in stages]
    base = names[1]
    table = []
    for i, n in enumerate(names):
        row = {"stage": n, "right_%": round(100 * sum(arms[n].values()) / len(T), 1),
               "any_candidate_right_%": round(100 * sum(ceil[n].values()) / len(T), 1)}
        if 0 < i < 6:
            row["vs previous"] = diff(n, names[i - 1]) if i < 5 else diff(n, names[3])
        if n != base:
            row["vs CrystaLLM x8 + CHGNet"] = diff(n, base)
        table.append(row)
    vk = [k for t in T for k in [c["key"] for c in candidates(t, top8)]]
    filt = {"candidates": len(vk), "rejected": sum(not V.get(k, {}).get("valid") for k in vk),
            "rejected_but_right": sum((not V.get(k, {}).get("valid")) and bool(E.get(k) and E[k][1]) for k in vk),
            "reject reasons": {r: sum(not V.get(k, {}).get(r, False) for k in vk) for r in ("composition", "no_clash", "volume_ok")}}
    rep = {"n_targets": len(T), "table": table, "validation filter": filt}
    json.dump({"report": rep}, open(OUT, "w"), indent=1)
    w = max(len(r["stage"]) for r in table)
    print(f"{'stage'.ljust(w)}  right   any-right  vs previous            vs CrystaLLM x8 + CHGNet")
    for r in table:
        print(f"{r['stage'].ljust(w)}  {r['right_%']:5.1f}  {r['any_candidate_right_%']:5.1f}      "
              f"{str(r.get('vs previous', '')):22} {r.get('vs CrystaLLM x8 + CHGNet', '')}")
    print(json.dumps(filt, indent=1))


def stage_report(a):
    """Stage 6: ranked candidates with provenance for given targets (markdown + JSON)."""
    import kg_pool
    pool, rows = kg_pool.load()
    E = json.load(open(ENERGY))
    V = json.load(open(VALID))
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    os.makedirs(REPORTS, exist_ok=True)
    for t in a.target:
        cs = candidates(t, top8)
        chosen = pick(cs, E, V, use_valid=True)
        ranked = sorted(cs, key=lambda c: (not V.get(c["key"], {}).get("valid"), E[c["key"]][0] if E.get(c["key"]) else 9e9))
        lines = [f"# {rows[t]['formula']} ({t})", "", "| # | source | provenance | valid | CHGNet eV/atom | matches reference |",
                 "|---|---|---|---|---|---|"]
        recs = []
        for i, c in enumerate(ranked, 1):
            v = V.get(c["key"], {})
            e = E.get(c["key"])
            prov = (f"template from {rows[c['from']]['formula']} ({c['from']}), K-Ont rank {c['kont_rank']}, score {c['kont_score']}"
                    if c["source"] == "template" else f"CrystaLLM sample {c['sample']}")
            why = "yes" if v.get("valid") else "no (" + ", ".join(k for k in ("composition", "no_clash", "volume_ok") if not v.get(k)) + ")"
            lines.append(f"| {i}{' (pick)' if c is chosen else ''} | {c['source']} | {prov} | {why} | "
                         f"{e[0] if e else 'n/a'} | {('yes' if e[1] else 'no') if e else 'n/a'} |")
            recs.append({**c, "provenance": prov, "checks": v, "energy": e[0] if e else None, "right": e[1] if e else None,
                         "pick": c is chosen})
        md = "\n".join(lines) + "\n"
        open(os.path.join(REPORTS, f"{t}.md"), "w", encoding="utf-8").write(md)
        json.dump({"target": t, "formula": rows[t]["formula"], "candidates": recs},
                  open(os.path.join(REPORTS, f"{t}.json"), "w"), indent=1)
        print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["validate", "ablation", "report"])
    ap.add_argument("--target", action="append", default=[])
    a = ap.parse_args()
    {"validate": stage_validate, "ablation": stage_ablation, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
