"""K-Ont knowledge injected into CrystaLLM as TEXT: template-seeded prompts (BUILD_LOG Step 53).

Steps 40 and 45: a K-Ont vector says WHICH framework, but the model must still write every lattice parameter
and coordinate (ceiling +2-3 points). Here the knowledge goes in as the start of the CIF itself: for each of
the K-Ont ranker's top 3 templates (Step 41), the target's elements are put into the template's framework
(Step 40 substitution, written in CrystaLLM's layout by kg_corpus.convert) and the CIF header -- formula,
atom-type block, space group, cell lengths and angles, volume, Z -- is the prompt; CrystaLLM (Step 40 plain
fine-tune `const`, no vector) writes the symmetry operators and atom sites. 3 samples per template.
Scored like Steps 40-41 on the 902 K-Ont-unseen targets (right = matches the reference). Compared at matched
budgets with CrystaLLM alone (Step 40 const samples) and the templates themselves (Step 41).

    .venv-crystallm/Scripts/python.exe -u abox/kg_seeded.py --stage gen
    .venv-crystallm/Scripts/python.exe -u abox/kg_seeded.py --stage score
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
RANKERS = {"kont": "prompt+kg+ft_kont", "plain": "prompt+kg+ft_base", "composition": "composition rule", "hautier": None}
# controls (--ranker plain|composition|hautier): the same seeding with templates chosen without K-Ont


def paths(r):
    sfx = "" if r == "kont" else f"_{r}"
    return (os.path.join(DATA, f"kgseed_gen{sfx}"), os.path.join(DATA, f"kgseed_prompts{sfx}.json"),
            os.path.join(DATA, f"kgseed_eval{sfx}.json"), os.path.join(DATA, f"kgseed_scores{sfx}.json"))


def tops(r, pool, rows):
    """top-3 template ids per target for a ranker; Hautier from Step 47's substitution probabilities"""
    if RANKERS[r]:
        t8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))[RANKERS[r]]
        return {t: [c for c, _ in t8[t][:3]] for t in pool["targets"]}
    from kg_substitution import best_mapping
    from kg_condition2 import similarity
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
    oxi, sp = json.load(open(os.path.join(DATA, "kg_oxi_guesses.json"))), SubstitutionProbability()
    out = {}
    for t in pool["targets"]:
        hp = {c: best_mapping(rows[t]["formula"], rows[c]["formula"], oxi, sp)[0] for c in pool["candidates"][t]}
        out[t] = sorted(pool["candidates"][t], key=lambda c: (-hp[c], -similarity(rows[t]["formula"], rows[c]["formula"]),
                                                               rows[c]["e_above_hull"] or 0))[:3]
    return out


def header(cif):
    """the CIF up to and including _cell_formula_units_Z: everything before the symmetry ops and atom sites"""
    out = []
    for line in cif.splitlines():
        out.append(line)
        if line.startswith("_cell_formula_units_Z"):
            return "\n".join(out) + "\n"
    return None


def stage_gen(a):
    import torch
    import kg_pool
    import kg_condition2 as k2
    import kg_condition3 as k3
    import phaseC_generate as gen
    pool, rows = kg_pool.load()
    GEN, PROMPTS, _, _ = paths(a.ranker)
    top = tops(a.ranker, pool, rows)
    targets = pool["targets"]
    if os.path.exists(PROMPTS):
        prompts = json.load(open(PROMPTS))
    else:
        jobs = [(rows[t], rows[c], 1024 - 8) for t in targets for c in top[t][:3]]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(k3._substitute, jobs, chunksize=8))
        prompts = {f"{t}|{c}": header(cif) for t, c, cif, _ in res if cif and header(cif)}
        json.dump(prompts, open(PROMPTS, "w"))
    print(f"{len(prompts)} template prompts for {len(targets)} targets", flush=True)
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = k2.crystal_vectors("grow5000_v2", "crystal_anon")
    dim = next(iter(vecs.values())).shape[0]
    m = k2.Deep(gen.load_model(dev, "crystallm_v1_small"), dim, 8, gated=True).to(dev)
    m.load_state_dict(torch.load(os.path.join(DATA, "kgcond3_const.pt"), map_location=dev))
    m.eval()
    todo = [(k, p) for k, p in sorted(prompts.items())
            if not os.path.exists(os.path.join(GEN, k.split("|")[0], f"{k.split('|')[1]}_{a.samples - 1}.cif"))]
    groups = {}
    for k, p in todo:
        groups.setdefault(len(tok.tokenize_cif(p)), []).append((k, p))
    per = max(1, a.batch_seqs // a.samples)
    batches = [g[i:i + per] for _, g in sorted(groups.items()) for i in range(0, len(g), per)]
    t0 = time.time()
    for bi, b in enumerate(batches):
        torch.manual_seed(a.seed + bi)
        ps = [p for _, p in b for _ in range(a.samples)]
        try:
            cifs = k3.sample_group(m, tok, ps, [torch.zeros(dim)] * len(ps), dev)
        except Exception as e:  # a prompt the tokenizer/model cannot take: skip this batch, keep going
            print(f"  batch {bi} failed: {e.__class__.__name__}", flush=True)
            continue
        for j, (k, _) in enumerate(b):
            t, c = k.split("|")
            os.makedirs(os.path.join(GEN, t), exist_ok=True)
            for s in range(a.samples):
                open(os.path.join(GEN, t, f"{c}_{s}.cif"), "w").write(cifs[j * a.samples + s])
        if (bi + 1) % 50 == 0:
            print(f"  batch {bi + 1}/{len(batches)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("generated", flush=True)


def _score(args):
    path, target, ref = args
    import dpo_pairs as b2
    if not os.path.exists(path):
        return path, {"valid": False}
    return b2._kg((path, target, ref))


def stage_score(a):
    import kg_pool
    import phaseC_generate as gen
    from pymatgen.core import Composition
    pool, rows = kg_pool.load()
    GEN, _, OUT, cache_p = paths(a.ranker)
    top = {t: [(c, None) for c in cs] for t, cs in tops(a.ranker, pool, rows).items()}
    targets = pool["targets"]
    sc = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    jobs = []
    for t in targets:
        ref, tf = gen.reference_from_text(rows[t]["cif"]).as_dict(), Composition(rows[t]["formula"]).reduced_formula
        for c, _ in top[t][:3]:
            for s in range(a.samples):
                p = os.path.join(GEN, t, f"{c}_{s}.cif")
                if p not in sc:
                    jobs.append((p, tf, ref))
    print(f"{len(jobs)} seeded samples to score", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (p, r) in enumerate(ex.map(_score, jobs, chunksize=8)):
            sc[p] = {k: r.get(k) for k in ("valid", "composition_ok", "strict")}
            if (n + 1) % 2000 == 0:
                json.dump(sc, open(cache_p, "w"))
    json.dump(sc, open(cache_p, "w"))
    ok = lambda p: bool(sc.get(p, {}).get("strict"))
    seeded = lambda t, ci, s: os.path.join(GEN, t, f"{top[t][ci][0]}_{s}.cif")
    tmpl = json.load(open(os.path.join(DATA, "kg_template_gen.json")))["found"]["prompt+kg+ft_kont"]
    cl = {m["id"]: m["samples"] for m in json.load(open(os.path.join(DATA, "kgcond3_eval_const.json")))}
    T = lambda t, k: tmpl[str(k)][t] > 0  # Step 41 template top-k found (k in 1, 2, 3, 5, 8)
    C = lambda t, k: any(s.get("strict") for s in cl[t][:k])
    S = lambda t, idx: any(ok(seeded(t, ci, s)) for ci, s in idx if ci < len(top[t]))
    arms = {
        "CrystaLLM x1": lambda t: C(t, 1), "CrystaLLM x3": lambda t: C(t, 3), "CrystaLLM x8": lambda t: C(t, 8),
        "template top-1": lambda t: T(t, 1), "templates top-3": lambda t: T(t, 3),
        "seeded: top-1 template x1": lambda t: S(t, [(0, 0)]),
        "seeded: top-1 template x3": lambda t: S(t, [(0, 0), (0, 1), (0, 2)]),
        "seeded: top-3 templates x1": lambda t: S(t, [(0, 0), (1, 0), (2, 0)]),
        "seeded: top-3 templates x3 (9)": lambda t: S(t, [(ci, s) for ci in range(3) for s in range(3)]),
        "budget 8: templates x3 + CrystaLLM x5": lambda t: T(t, 3) or C(t, 5),
        "budget 8: templates x3 + seeded top-3 x1 + CrystaLLM x2": lambda t: T(t, 3) or S(t, [(0, 0), (1, 0), (2, 0)]) or C(t, 2),
        "budget 8: seeded top-3 x2 + CrystaLLM x2": lambda t: S(t, [(ci, s) for ci in range(3) for s in range(2)]) or C(t, 2),
        "budget 8: templates x3 + seeded top-3 x1 + seeded top-1 x1 + CrystaLLM x1":
            lambda t: T(t, 3) or S(t, [(0, 0), (1, 0), (2, 0), (0, 1)]) or C(t, 1)}
    found = {n: {t: float(f(t)) for t in targets} for n, f in arms.items()}
    valid = sum(bool(v.get("valid")) for v in sc.values()) / max(1, len(sc))
    comp = sum(bool(v.get("composition_ok")) for v in sc.values()) / max(1, len(sc))
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y):
        d = {t: found[x][t] - found[y][t] for t in targets}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n": len(targets), "seeded_samples_valid": round(100 * valid, 1), "seeded_samples_right_composition": round(100 * comp, 1),
           "found": {n: round(100 * sum(v.values()) / len(targets), 1) for n, v in found.items()},
           "seeded top-1 x1 - template top-1": diff("seeded: top-1 template x1", "template top-1"),
           "seeded top-1 x1 - CrystaLLM x1": diff("seeded: top-1 template x1", "CrystaLLM x1"),
           "seeded top-3 x1 - templates top-3": diff("seeded: top-3 templates x1", "templates top-3"),
           "seeded top-3 x1 - CrystaLLM x3": diff("seeded: top-3 templates x1", "CrystaLLM x3"),
           "budget 8 (T3+S3+C2) - (T3+C5)": diff("budget 8: templates x3 + seeded top-3 x1 + CrystaLLM x2", "budget 8: templates x3 + CrystaLLM x5"),
           "budget 8 (T3+S3+C2) - CrystaLLM x8": diff("budget 8: templates x3 + seeded top-3 x1 + CrystaLLM x2", "CrystaLLM x8")}
    json.dump({"report": rep}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    ap.add_argument("--ranker", default="kont", choices=list(RANKERS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--batch-seqs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    {"gen": stage_gen, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
