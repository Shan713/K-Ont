"""Does a bigger KG give better templates? (BUILD_LOG Step 56)

Same 902 K-Ont-unseen targets, same retriever, two template pools:
  5,000 KG   Step 40's pool (K-Ont-seen KG materials, kg_pool.py)
  extended   + the kg_extend.py materials (no evaluation composition anywhere in them)
Candidates = pool materials with the same anonymous formula and a different formula. Retriever = the one-model
K-Ont of Step 52c (data/runs/kont_fw_retain_galen_anchor): cosine between the target's structure-free sentence
and each candidate's full sentence, as kg_retriever_ft.py scores; composition rule as the no-K-Ont reference.
Metrics: targets with any candidate, oracle (some candidate has the true space group), template found@1/3/8
(kg_template_gen substitution, unrelaxed); paired bootstrap extended - 5,000.

    .venv-crystallm/Scripts/python.exe abox/kg_ext_eval.py --stage cands
    .venv/Scripts/python.exe abox/kg_ext_eval.py --stage score
    .venv-crystallm/Scripts/python.exe abox/kg_ext_eval.py --stage eval
"""
import argparse
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
DATA = os.path.join(HERE, "data", "crystallm")
MODEL = os.path.join(HERE, "data", "runs", "kont_fw_retain_galen_anchor", "final")
CANDS = os.path.join(DATA, "kg_ext_candidates.json")
SCORES = os.path.join(DATA, "kg_ext_scores.json")
OUT = os.path.join(DATA, "kg_ext_eval.json")


def ext_pool():
    return {r["id"]: r for r in json.load(gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "rt", encoding="utf-8"))}


def stage_cands(a):
    """crystallm venv (pymatgen): candidate lists for both pools"""
    from pymatgen.core import Composition
    import kg_pool
    pool, rows = kg_pool.load()
    ext = ext_pool()
    by = {}
    for i, r in ext.items():
        by.setdefault(Composition(r["formula"]).anonymized_formula, []).append(i)
    out = {}
    for t in pool["targets"]:
        an, f = Composition(rows[t]["formula"]).anonymized_formula, rows[t]["formula"]
        extra = [c for c in by.get(an, []) if ext[c]["formula"] != f and c not in rows]
        out[t] = {"base": pool["candidates"][t], "ext": pool["candidates"][t] + extra}
    json.dump(out, open(CANDS, "w"))
    print(f"targets {len(out)}; mean candidates {sum(len(v['base']) for v in out.values()) / len(out):.1f} -> "
          f"{sum(len(v['ext']) for v in out.values()) / len(out):.1f}")


def stage_score(a):
    """OnT venv: one-model K-Ont cosine for every (target, candidate) of the extended pool"""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from ont.hit import HierarchyTransformer
    from kg_retriever_ft import target_text
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    full = {k.split("/")[-1]: v for k, v in vm.items()}
    full.update({i: r["sentence"] for i, r in ext_pool().items()})
    cands = json.load(open(CANDS))
    m = HierarchyTransformer.from_pretrained(MODEL).to("cuda").eval()

    def emb(texts):
        out = []
        for i in range(0, len(texts), 64):
            f = m.tokenize(texts[i:i + 64])
            f = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in f.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out.append(F.normalize(m(f)["sentence_embedding"].float(), dim=-1))
        return torch.cat(out)
    cs = sorted({c for v in cands.values() for c in v["ext"]})
    ce = dict(zip(cs, emb([full[c] for c in cs])))
    ts = sorted(cands)
    te = dict(zip(ts, emb([target_text(full[t]) for t in ts])))
    json.dump({t: {c: float(ce[c] @ te[t]) for c in cands[t]["ext"]} for t in ts}, open(SCORES, "w"))
    print(f"scored {len(ts)} targets over {len(cs)} candidates")


def stage_eval(a):
    from concurrent.futures import ProcessPoolExecutor
    import kg_pool
    from kg_condition2 import similarity
    from kg_template_gen import _job
    pool, rows = kg_pool.load()
    ext = ext_pool()
    allrows = {**ext, **rows}
    cands, sc = json.load(open(CANDS)), json.load(open(SCORES))
    targets = pool["targets"]
    rank = {}
    for name, key in (("5,000 KG", "base"), ("extended KG", "ext")):
        rank[f"K-Ont (52c) · {name}"] = {t: sorted(cands[t][key], key=lambda c: -sc[t][c])[:8] for t in targets}
        rank[f"composition rule · {name}"] = {t: sorted(cands[t][key], key=lambda c: (-similarity(rows[t]["formula"], allrows[c]["formula"]),
                                                                                     allrows[c]["e_above_hull"] or 0))[:8] for t in targets}
    cache_p = os.path.join(DATA, "kg_template_ok_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = sorted({(t, c) for d in rank.values() for t, cs in d.items() for c in cs if f"{t}|{c}" not in cache})
    print(f"{len(todo)} substitutions to match", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for (t, c), ok in ex.map(_job, [(allrows[t], allrows[c], False) for t, c in todo], chunksize=16):
            cache[f"{t}|{c}"] = ok
    json.dump(cache, open(cache_p, "w"))
    found = {n: {k: {t: float(any(cache[f"{t}|{c}"] for c in d[t][:k])) for t in targets} for k in (1, 3, 8)} for n, d in rank.items()}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y, k):
        d = {t: found[x][k][t] - found[y][k][t] for t in targets}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n_targets": len(targets), "extension_materials": len(ext)}
    for key, name in (("base", "5,000 KG"), ("ext", "extended KG")):
        rep[name] = {"targets_with_candidates": sum(bool(cands[t][key]) for t in targets),
                     "mean_candidates": round(sum(len(cands[t][key]) for t in targets) / len(targets), 1),
                     "oracle_true_space_group": round(100 * sum(any(allrows[c]["space_group"] == rows[t]["space_group"] for c in cands[t][key])
                                                                for t in targets) / len(targets), 1)}
    rep["found"] = {n: {f"@{k}": round(100 * sum(v.values()) / len(targets), 1) for k, v in f.items()} for n, f in found.items()}
    for r in ("K-Ont (52c)", "composition rule"):
        for k in (1, 3, 8):
            rep[f"{r}: extended - 5,000 @{k}"] = diff(f"{r} · extended KG", f"{r} · 5,000 KG", k)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cands", "score", "eval"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"cands": stage_cands, "score": stage_score, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
