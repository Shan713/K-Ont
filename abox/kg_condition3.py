"""K-Ont injection that the generator must follow: counterfactual framework training + K-Ont retrieval
(BUILD_LOG Step 40).

Why Step 38 failed, measured: K-Ont's framework vectors DO separate polymorphs (crystal_anon: cosine 0.996
same space group vs 0.77 different, within a formula type; nearest neighbour has the right space group
86% of the time), but the generator ignored them (compliance with the analogue 37.3% vs 37.4% without
K-Ont; true structure 70.2 vs 68.6 even when the analogue was right). CrystaLLM was pre-trained on
Materials Project, so for KG training materials it already "knows" the answer from the composition and
fine-tuning never has to read the vector; and when no same-prototype neighbour existed, v2 trained on a
composition-retrieved analogue of a possibly different framework, which teaches it to ignore the vector.

Fix, three parts:
  1 COUNTERFACTUALS (stage cf): for each training material t, its composition substituted into the
    framework of a pool material c of the same formula type but a DIFFERENT space group (elements mapped
    in kg_condition2._ordered order, volume scaled by atomic radii, written exactly like the corpus,
    kg_corpus.convert). Trained as (prompt t, vector of c's framework) -> t-in-c's-framework. Same prompt,
    different vector, different answer: the only way to lower the loss is to read the vector.
  2 CLEAN REAL ROWS: the vector of a same-prototype neighbour, else the material's own framework vector
    -- never a wrong framework. Condition dropout (zero vector) on real rows only.
  3 RETRIEVAL at test time: the analogue is picked by the learned ranker (kg_rerank.py) that uses the
    fine-tuned K-Ont retriever (kg_retriever_ft.py), from the K-Ont-seen pool (kg_pool.py).
Split (kg_pool.py): generators train on K-Ont-seen KG materials only; evaluation on all 902 KG materials
K-Ont never saw that have a candidate (vs 79 in Steps 37-38), 8 samples each, same seeds across arms.

Arms (w = 1, plain conditional sampling):
  const      plain fine-tune on the same real rows, no K-Ont                  (baseline)
  cf_none    counterfactual model, zero vector                                 (model without K-Ont input)
  cf_rr      K-Ont ranker's analogue (prompt+kg+ft_kont)                       (THE K-Ont arm)
  cf_rrnok   ranker without any K-Ont feature (prompt+kg)                      (K-Ont's share of retrieval)
  cf_comp    composition-rule analogue (Step 38's retrieval)
  cf_rand    a random candidate's framework                                    (vector-content control)
  cf_gate    K-Ont ranker's analogue if its score >= 0.5, else zero vector
  cf_mix     half the samples from the ranker's 1st analogue, half from its 2nd
  cf_oracle  a candidate with the true space group (ceiling, uses the answer -- not a method)
Scored per sample: right composition, true structure (StructureMatcher fit to the reference), framework
compliance with the arm's analogue (fit_anonymous). Paired bootstrap over targets.
Acceptance, fixed before the results: cf_rr - const on true-structure share, 95% CI low > 0; and
cf_rr - cf_rand on compliance CI low > 0 (the vector steers).

    .venv-crystallm/Scripts/python.exe -u abox/kg_condition3.py --stage cf
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition3.py --stage train --arm const
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition3.py --stage train --arm cf
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition3.py --stage eval --arms const,cf_rr,...
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition3.py --stage report
"""
import argparse
import gzip
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import phaseC_generate as gen  # noqa: E402
import dpo_pairs as b2  # noqa: E402
import kg_condition as k1  # noqa: E402
import kg_condition2 as k2  # noqa: E402
import kg_pool  # noqa: E402

DATA = gen.DATA
CF = os.path.join(DATA, "kg_cf.json.gz")
RERANK = os.path.join(DATA, "kg_rerank.json")
KONT_RANKER, NOKONT_RANKER = "prompt+kg+ft_kont", "prompt+kg"
GEN = os.path.join(DATA, "kgcond3_gen")
ALL_ARMS = "const,cf_rr,cf_rrnok,cf_comp,cf_rand,cf_none,cf_gate,cf_oracle,cf_mix"


# ---------------------------------------------------------------- 1 counterfactual rows
def _substitute(args):
    t, c, block = args
    import kg_corpus
    from pymatgen.core import Element, Structure
    from pymatgen.io.cif import CifWriter
    try:
        s = gen.reference_from_text(c["cif"])
        mapping = {a.symbol: b.symbol for a, b in zip(k2._ordered(c["formula"]), k2._ordered(t["formula"]))}
        s.replace_species(mapping)
        r = lambda e: Element(e).atomic_radius or 1.5
        comp_c = gen.reference_from_text(c["cif"]).composition
        old = sum(n * float(r(el.symbol)) ** 3 for el, n in comp_c.items())
        new = sum(n * float(r(el.symbol)) ** 3 for el, n in s.composition.items())
        s.scale_lattice(s.volume * new / old)
        x = {k: t[k] for k in ("formula", "structure_family", "battery_role", "e_above_hull", "band_gap")}
        x.update({"id": t["id"], "split": t["split"], "space_group": c["space_group"], "crystal_system": c["crystal_system"],
                  "cif": str(CifWriter(s))})
        _, row, err = kg_corpus.convert((x, block))
        return t["id"], c["id"], (row or {}).get("cif"), err
    except Exception as e:
        return t["id"], c["id"], None, e.__class__.__name__


def stage_cf(a):
    pool, rows = kg_pool.load()
    rng = random.Random(a.seed)
    jobs = []
    for i in pool["seen_train"] + pool["seen_val"]:
        t = rows[i]
        other = {}
        for c in pool["candidates"][i]:
            if rows[c]["space_group"] != t["space_group"]:
                other.setdefault(rows[c]["space_group"], []).append(c)
        sgs = list(other)
        rng.shuffle(sgs)
        for sg in sgs[:a.cf_per_material]:  # distinct frameworks
            jobs.append((t, rows[rng.choice(other[sg])], 1024 - a.prefix_len))
    print(f"{len(jobs)} counterfactual jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_substitute, jobs, chunksize=8))
    ok = [{"id": t, "analogue": c, "cif": cif} for t, c, cif, _ in res if cif]
    errs = {}
    for *_, cif, e in res:
        if not cif:
            errs[str(e)] = errs.get(str(e), 0) + 1
    json.dump(ok, gzip.open(CF, "wt", encoding="utf-8"))
    print(f"{len(ok)} counterfactual rows; failures {errs}", flush=True)


# ---------------------------------------------------------------- 2 training
def real_analogues(pool, rows):
    """same-prototype neighbour in the pool (most similar chemistry), else the material itself"""
    out = {}
    for i in pool["seen_train"] + pool["seen_val"]:
        t = rows[i]
        same = [c for c in pool["candidates"][i] if rows[c]["space_group"] == t["space_group"]]
        out[i] = max(same, key=lambda c: k2.similarity(t["formula"], rows[c]["formula"])) if same else i
    return out


def stage_train(a):
    torch.manual_seed(a.seed)
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = k2.crystal_vectors(a.export, "crystal_anon")
    pool, rows = kg_pool.load()
    ra = real_analogues(pool, rows)
    dim = next(iter(vecs.values())).shape[0]
    model = k2.Deep(gen.load_model(dev, "crystallm_v1_small"), dim, a.prefix_len, gated=True).to(dev).train()
    lim = model.gpt.config.block_size - a.prefix_len + 1
    enc = lambda cif: k1.encode(tok, cif)
    data = {s: [(ra[i], e, True) for i in pool[f"seen_{s}"] for e in [enc(rows[i]["cif"])] if len(e) <= lim]
            for s in ("train", "val")}
    cf = {"train": [], "val": []}
    if a.arm == "cf":
        val_ids = set(pool["seen_val"])
        for r in json.load(gzip.open(CF, "rt", encoding="utf-8")):
            e = enc(r["cif"])
            if len(e) <= lim:
                cf["val" if r["id"] in val_ids else "train"].append((r["analogue"], e, False))
    zero = torch.zeros(dim)
    rng = random.Random(a.seed)

    def loss_of(batch, drop):
        seqs = [s for _, s, _ in batch]
        T = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        y = torch.full((len(seqs), T), -1, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])
            y[i, :len(s) - 1] = torch.tensor(s[1:])
        if a.arm == "const":
            v = torch.zeros(len(batch), dim)
        else:  # condition dropout on real rows only: a counterfactual without its vector has no right answer
            v = torch.stack([zero if (real and drop and rng.random() < a.p_drop) else vecs[c] for c, _, real in batch])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(v.to(dev), x.to(dev))
        return F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.to(dev).reshape(-1), ignore_index=-1)

    opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": a.lr},
                             {"params": [p for n, p in model.named_parameters() if not n.startswith("gpt.")], "lr": a.lr_cond}],
                            weight_decay=0.1)

    def val_loss(rows_):
        if not rows_:
            return float("nan")
        model.eval()
        with torch.no_grad():
            ls = [loss_of(rows_[i:i + a.batch], False).item() for i in range(0, len(rows_), a.batch)]
        model.train()
        return sum(ls) / len(ls)

    out = os.path.join(DATA, f"kgcond3_{a.arm}.pt")
    log = [{"epoch": 0, "val_real": val_loss(data["val"]), "val_cf": val_loss(cf["val"])}]
    best, t0 = log[0]["val_real"] + (log[0]["val_cf"] if cf["val"] else 0), time.time()
    torch.save(model.state_dict(), out)
    print(f"{a.arm}: {len(data['train'])} real + {len(cf['train'])} counterfactual train rows; {log[0]}", flush=True)
    for ep in range(1, a.epochs + 1):
        rows_ = data["train"] + cf["train"]
        rng.shuffle(rows_)
        for i in range(0, len(rows_), a.batch):
            loss = loss_of(rows_[i:i + a.batch], True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rec = {"epoch": ep, "val_real": val_loss(data["val"]), "val_cf": val_loss(cf["val"]),
               "gates": [round(float(torch.tanh(g)), 3) for g in model.gates]}
        log.append(rec)
        print(f"{a.arm} epoch {ep}: {rec} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        c = rec["val_real"] + (rec["val_cf"] if cf["val"] else 0)
        if c < best:
            best = c
            torch.save(model.state_dict(), out)
    json.dump(log, open(os.path.join(DATA, f"kgcond3_{a.arm}_log.json"), "w"), indent=1)


# ---------------------------------------------------------------- 3 evaluation
def analogue_plan(arm, targets, rows, pool, n, seed):
    """per target: list of n analogue ids (None = zero vector) and the compliance reference id"""
    rr = json.load(open(RERANK))
    top = rr["top3"]
    rng = random.Random(seed)
    plan = {}
    for t in targets:
        i = t["id"]
        kont = top[KONT_RANKER][i]
        ref = kont[0][0]
        if arm in ("const", "cf_none"):
            an = [None] * n
        elif arm == "cf_rr":
            an = [kont[0][0]] * n
        elif arm == "cf_rrnok":
            an = [top[NOKONT_RANKER][i][0][0]] * n
        elif arm == "cf_comp":
            an = [top["composition rule"][i][0][0]] * n
        elif arm == "cf_rand":
            an = [rng.choice(pool["candidates"][i])] * n
        elif arm == "cf_gate":
            an = [kont[0][0] if kont[0][1] >= 0.5 else None] * n
        elif arm == "cf_mix":
            second = kont[1][0] if len(kont) > 1 else kont[0][0]
            an = [kont[0][0]] * (n // 2) + [second] * (n - n // 2)
        elif arm == "cf_oracle":
            right = [c for c in pool["candidates"][i] if rows[c]["space_group"] == rows[i]["space_group"]]
            an = [right[0] if right else kont[0][0]] * n
        else:
            raise ValueError(arm)
        plan[i] = (an, an[0] if an[0] else ref)
    return plan


@torch.no_grad()
def sample_group(model, tok, prompts, vecs, dev, max_new=1000, temperature=0.8, top_k=10):
    """one sequence per (prompt, vec); all prompts have the same token length, so the batch shares
    positions and one KV cache"""
    gpt = model.gpt
    ids = torch.tensor([tok.encode(tok.tokenize_cif(p)) for p in prompts], dtype=torch.long, device=dev)
    pre, adds = model.cond(torch.stack(vecs).to(dev))
    wte = gpt.transformer.wte
    h = torch.cat([pre, wte(ids)], 1)
    out = ids
    n = len(prompts)
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, device=dev)
    cache, pos = [None] * len(gpt.transformer.h), 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            if pos + h.size(1) > gpt.config.block_size:
                break
            logits = k2._run(gpt, h, cache, pos, adds).float() / temperature
            pos += h.size(1)
            kth, _ = torch.topk(logits, top_k)
            logits[logits < kth[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, nl), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= (prev == nl) & (nxt == nl)
            prev = nxt
            if done.all():
                break
            h = wte(nxt)[:, None]
    return [tok.decode(r.tolist()).split("\n\n")[0] + "\n" for r in out]


def eval_targets(pool, rows):
    return [{"id": i, "formula": rows[i]["formula"], "cell": b2.cell_of(rows[i]["cif"])} for i in pool["targets"]]


def stage_eval(a):
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = k2.crystal_vectors(a.export, "crystal_anon")
    dim = next(iter(vecs.values())).shape[0]
    pool, rows = kg_pool.load()
    targets = eval_targets(pool, rows)
    groups = {}
    for t in targets:
        groups.setdefault(len(tok.tokenize_cif(gen.prompt_for(t["cell"]))), []).append(t)
    per = max(1, a.batch_seqs // a.samples)
    batches = [g[i:i + per] for _, g in sorted(groups.items()) for i in range(0, len(g), per)]
    print(f"{len(targets)} targets in {len(batches)} batches", flush=True)
    for arm in a.arms.split(","):
        out = os.path.join(DATA, f"kgcond3_eval_{arm}.json")
        if os.path.exists(out):
            continue
        plan = analogue_plan(arm, targets, rows, pool, a.samples, a.seed)
        gdir = os.path.join(GEN, arm)
        if not os.path.exists(os.path.join(gdir, "_done")):
            m = k2.Deep(gen.load_model(dev, "crystallm_v1_small"), dim, a.prefix_len, gated=True).to(dev)
            m.load_state_dict(torch.load(os.path.join(DATA, f"kgcond3_{'const' if arm == 'const' else 'cf'}.pt"), map_location=dev))
            m.eval()
            t0 = time.time()
            for bi, b in enumerate(batches):
                torch.manual_seed(a.seed + bi)
                prompts, vs = [], []
                for t in b:
                    for c in plan[t["id"]][0]:
                        prompts.append(gen.prompt_for(t["cell"]))
                        vs.append(vecs[c] if c else torch.zeros(dim))
                cifs = sample_group(m, tok, prompts, vs, dev)
                for j, t in enumerate(b):
                    os.makedirs(os.path.join(gdir, t["id"]), exist_ok=True)
                    for k in range(a.samples):
                        open(os.path.join(gdir, t["id"], f"s_{k}.cif"), "w").write(cifs[j * a.samples + k])
                if (bi + 1) % 10 == 0:
                    print(f"{arm}: batch {bi + 1}/{len(batches)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            open(os.path.join(gdir, "_done"), "w").write("")
            del m
            torch.cuda.empty_cache()
        refs = {}

        def struct(i):
            if i not in refs:
                refs[i] = gen.reference_from_text(rows[i]["cif"]).as_dict()
            return refs[i]
        jobs = [(os.path.join(gdir, t["id"], f"s_{k}.cif"), gen_formula(t), struct(t["id"]),
                 struct(plan[t["id"]][0][k] or plan[t["id"]][1])) for t in targets for k in range(a.samples)]
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            sc = dict(ex.map(k2._score, jobs, chunksize=8))
        res = [{"id": t["id"], "formula": t["formula"], "analogues": plan[t["id"]][0],
                "samples": [sc[os.path.join(gdir, t["id"], f"s_{k}.cif")] for k in range(a.samples)]} for t in targets]
        json.dump(res, open(out, "w"), indent=1)
        print(f"{arm}: scored ({(time.time() - t0) / 60:.1f} min)", flush=True)


def gen_formula(t):
    from pymatgen.core import Composition
    return Composition(t["cell"]).reduced_formula


# ---------------------------------------------------------------- 4 report
def stage_report(a):
    arms = [x for x in ALL_ARMS.split(",") if os.path.exists(os.path.join(DATA, f"kgcond3_eval_{x}.json"))]
    data = {x: {m["id"]: k2.per_comp(m) for m in json.load(open(os.path.join(DATA, f"kgcond3_eval_{x}.json")))} for x in arms}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    pool, rows = kg_pool.load()
    old79 = {r["id"] for r in k1.corpus("test")} & set(ids)
    keys = ["valid", "composition", "strict", "found", "complies"]
    rng = random.Random(0)
    rep = {"arms": arms, "acceptance": "cf_rr - const strict CI low > 0; cf_rr - cf_rand complies CI low > 0"}
    pairs = [("cf_rr", "const"), ("cf_rr", "cf_rand"), ("cf_rr", "cf_rrnok"), ("cf_rr", "cf_comp"), ("cf_rr", "cf_none"),
             ("cf_gate", "const"), ("cf_mix", "const"), ("cf_mix", "cf_rr"), ("cf_none", "const"), ("cf_oracle", "const")]
    for name, sel in (("all", ids), ("old79", sorted(old79))):
        boots = [[rng.choice(sel) for _ in sel] for _ in range(10000)]
        r = {"n": len(sel), "means": {x: {k: round(100 * sum(data[x][i][k] for i in sel) / len(sel), 1) for k in keys} for x in arms},
             "paired": {}}
        for p, q in pairs:
            if p in arms and q in arms:
                row = {}
                for k in ("strict", "found", "complies", "composition"):
                    diff = {i: data[p][i][k] - data[q][i][k] for i in sel}
                    bs = sorted(100 * sum(diff[i] for i in b) / len(b) for b in boots)
                    row[k] = {"diff": round(100 * sum(diff.values()) / len(sel), 1), "ci95": [round(bs[250], 1), round(bs[9749], 1)],
                              "better": sum(v > 0 for v in diff.values()), "worse": sum(v < 0 for v in diff.values())}
                r["paired"][f"{p} - {q}"] = row
        rep[name] = r
    json.dump(rep, open(os.path.join(DATA, "kgcond3_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cf", "train", "eval", "report"])
    ap.add_argument("--arm", default="cf", choices=["cf", "const"])
    ap.add_argument("--arms", default=ALL_ARMS)
    ap.add_argument("--export", default="grow5000_v2")
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--cf-per-material", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-cond", type=float, default=3e-4)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch-seqs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    {"cf": stage_cf, "train": stage_train, "eval": stage_eval, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
