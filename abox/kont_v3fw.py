"""K-Ont v3: the framework relation trained NATIVELY in K-Ont's hyperbolic objective (BUILD_LOG Step 46).

Step 43 added "has the framework of" by fine-tuning K-Ont afterwards with a separate retrieval loss. Here the
relation is one more existential role in K-Ont's own training data, learned with OnT's loss (role rotation in
the Poincare ball, in-batch negatives) alongside everything K-Ont already learns:
  data   Step 36's merged data (data/battgpt_merged_grow_5000_v2_full) + exist rows
         {Concept: structure-free sentence of a K-Ont-seen KG training material (kg_pool seen_train; name,
          formula, chemistry, properties, Sites), role: "has the framework of", con: full sentence of a
          K-Ont-seen pool material with the same anonymous formula and space group (up to 2 per material)}
         -> data/battgpt_merged_grow_5000_v3fw_full
  train  train_ont.py exactly as Step 36 (full, galen, in-batch negatives, batch 32, 3 epochs)
         -> data/runs/full_grow5000_v3fw
  eval   (a) phase5_checks (does K-Ont keep its ontology abilities?) (b) framework retrieval with K-Ont's own
         existential score -- score(target, rot_r(candidate)), as phase5 hasStructure -- on the 902 K-Ont-unseen
         targets: top-1 right space group and template found@1/3 (kg_template_gen substitution), against
         Step 40's fine-tuned K-Ont / plain retrievers and Step 43's retention model.

    .venv/Scripts/python.exe abox/kont_v3fw.py --stage data
    .venv/Scripts/python.exe train_ont.py --variant full --epochs 3 --batch-size 32 --in-batch-negs \
        --data-prefix data/battgpt_merged_grow_5000_v3fw --output data/runs/full_grow5000_v3fw
    .venv/Scripts/python.exe abox/kont_v3fw.py --stage score      (GPU: existential scores for the 902)
    .venv-crystallm/Scripts/python.exe abox/kont_v3fw.py --stage eval
"""
import argparse
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
SRC = os.path.join(HERE, "data", "battgpt_merged_grow_5000_v2_full")
DATA = os.path.join(HERE, "data", "crystallm")
ROLE = "has the framework of"
# v3fw (Step 46): framework rows, in-batch (random) negatives only.
# v4hn (Step 51): the same rows + a per-row HARD negative ("hard_neg": a pool candidate of the same formula
# type but another space group), contrasted inside K-Ont's own existential loss (OnT patch
# patches/ont_exist_hard_negatives.patch) -- Step 46 showed random in-batch negatives are too easy.
VARIANT = os.environ.get("KONT_VARIANT", "v3fw")
DST = os.path.join(HERE, "data", f"battgpt_merged_grow_5000_{VARIANT}_full")
RUN = os.path.join(HERE, "data", "runs", f"full_grow5000_{VARIANT}")
SCORES = os.path.join(DATA, f"kg_ft_scores_kont_{VARIANT}.json")


def full_sentences():
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    return {k.split("/")[-1]: v for k, v in vm.items()}


def stage_data(a):
    import kg_pool
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from kg_retriever_ft import target_text
    pool, rows = kg_pool.load()
    full = full_sentences()
    shutil.copytree(SRC, DST, dirs_exist_ok=True)
    rng = random.Random(a.seed)
    new = []
    for i in pool["seen_train"]:
        same = [c for c in pool["candidates"][i] if rows[c]["space_group"] == rows[i]["space_group"]]
        other = [c for c in pool["candidates"][i] if rows[c]["space_group"] != rows[i]["space_group"]]
        for c in rng.sample(same, min(a.per_material, len(same))):
            r = {"Concept": target_text(full[i]), "role": ROLE, "con": full[c]}
            if VARIANT != "v3fw":
                r["hard_neg"] = full[rng.choice(other)] if other else ""
            new.append(r)
    path = os.path.join(DST, "train_exist.jsonl")
    old = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    with open(path, "w", encoding="utf-8") as f:  # rewritten whole: every row gets the same keys (Dataset schema)
        for r in old + new:
            if VARIANT != "v3fw":
                r.setdefault("hard_neg", "")
            f.write(json.dumps(r) + "\n")
    roles = json.load(open(os.path.join(DST, "role_names.json")))
    if ROLE not in roles.values():
        roles[str(len(roles))] = ROLE
    json.dump(roles, open(os.path.join(DST, "role_names.json"), "w"), indent=1)
    s = json.load(open(os.path.join(DST, "merge_summary.json")))
    s.update({"step46_framework_rows": len(new), "step46_role": ROLE, "n_merged_exist": s["n_merged_exist"] + len(new)})
    json.dump(s, open(os.path.join(DST, "merge_summary.json"), "w"), indent=1)
    print(f"{VARIANT}: {len(new)} '{ROLE}' rows ({sum(bool(r.get('hard_neg')) for r in new)} with a hard negative) from {len(pool['seen_train'])} seen training materials -> {DST}")


def stage_score(a):
    """K-Ont's own existential score for every (target, candidate) of the 902 K-Ont-unseen targets"""
    import torch
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from ont.model import OntologyTransformer
    import kg_pool
    from kg_retriever_ft import target_text
    pool, rows = kg_pool.load()
    full = full_sentences()
    m = OntologyTransformer.from_pretrained(os.path.join(RUN, "final"))
    m.hit_model.eval()
    man, lam = m.manifold, m.best_lambda or 0.0
    targets = pool["targets"]
    cands = sorted({c for t in targets for c in pool["candidates"][t]})
    with torch.no_grad():
        T = m.encode([target_text(full[t]) for t in targets], batch_size=64, convert_to_tensor=True).float()
        R = torch.tensor(m.encode_existence([ROLE] * len(cands), [full[c] for c in cands]), device=T.device).float()
    ci = {c: k for k, c in enumerate(cands)}
    out = {}
    for k, t in enumerate(targets):
        cs = pool["candidates"][t]
        r = R[[ci[c] for c in cs]]
        d = man.dist(T[k].unsqueeze(0).expand(len(cs), -1), r)
        s = -(d + lam * (man.dist0(r) - man.dist0(T[k].unsqueeze(0))))
        out[t] = dict(zip(cs, s.tolist()))
    json.dump(out, open(SCORES, "w"))
    print(f"scored {len(out)} targets with K-Ont's existential score (lambda {lam})")


def stage_eval(a):
    import kg_pool
    from concurrent.futures import ProcessPoolExecutor
    from kg_template_gen import _job
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    files = {f"K-Ont {VARIANT}, native framework relation": SCORES,
             "K-Ont v3, native framework relation (Step 46)": os.path.join(DATA, "kg_ft_scores_kont_v3fw.json"),
             "K-Ont + retrieval fine-tune (Step 40)": os.path.join(DATA, "kg_ft_scores_kont.json"),
             "plain encoder + retrieval fine-tune (Step 40)": os.path.join(DATA, "kg_ft_scores_base.json"),
             "K-Ont + fine-tune + retention (Step 43)": os.path.join(DATA, "kg_ft_scores_kont_fw_retain.json"),
             "K-Ont v2, no framework training": None}
    top = {}
    for n, f in files.items():
        if f is None:
            continue
        s = json.load(open(f))
        top[n] = {t: sorted(s[t], key=s[t].get, reverse=True)[:3] for t in targets if s.get(t)}
    cache_p = os.path.join(DATA, "kg_template_ok_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = sorted({(t, c) for d in top.values() for t, cs in d.items() for c in cs if f"{t}|{c}" not in cache})
    with ProcessPoolExecutor(3) as ex:
        for (t, c), ok in ex.map(_job, [(rows[t], rows[c], False) for t, c in todo], chunksize=16):
            cache[f"{t}|{c}"] = ok
    json.dump(cache, open(cache_p, "w"))
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]
    per = {n: {"sg": {t: float(rows[d[t][0]]["space_group"] == rows[t]["space_group"]) for t in targets},
               **{f"found@{k}": {t: float(any(cache[f"{t}|{c}"] for c in d[t][:k])) for t in targets} for k in (1, 3)}}
           for n, d in top.items()}
    rep = {"retrieval": {n: {k: round(100 * sum(v.values()) / len(targets), 1) for k, v in p.items()} for n, p in per.items()}}
    ref = f"K-Ont {VARIANT}, native framework relation"
    for n in per:
        if n != ref:
            for k in ("sg", "found@1", "found@3"):
                d = {t: per[ref][k][t] - per[n][k][t] for t in targets}
                bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
                rep["retrieval"][f"{VARIANT} - {n} ({k})"] = [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    p5 = os.path.join(RUN, "phase5_checks.json")
    if os.path.exists(p5):
        import kont_framework as kf
        rep["checks"] = {"K-Ont v2 (Step 36)": kf.headline(os.path.join(HERE, "data", "runs", "full_grow5000_v2", "phase5_checks.json")),
                         f"K-Ont {VARIANT} (native framework relation)": kf.headline(p5)}
    json.dump(rep, open(os.path.join(DATA, f"kont_{VARIANT}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "score", "eval"])
    ap.add_argument("--per-material", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"data": stage_data, "score": stage_score, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
