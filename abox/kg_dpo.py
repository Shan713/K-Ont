"""Task B3: KG-reward fine-tuning of CrystaLLM (small) -- SFT on the KG corpus, then DPO on the B2 pairs.

Stages (each resumable by its output; GPU unless noted):
  sft     CrystaLLM small, next-token loss on the rebuilt KG corpus (kg_corpus_train; val for early stop)
          -> data/crystallm/crystallm_kg_sft/ckpt.pt          (same format as the CrystaLLM checkpoints)
  dpo     policy and frozen reference both start from --init (default the SFT model); DPO loss on
          dpo_pairs_train.jsonl, val pairs for model selection; pairs whose id left the corpus are dropped
          -> data/crystallm/crystallm_kg_dpo/ckpt.pt (+ dpo_train_log.json)
  eval    --targets crystallm (primary): 100 Li/Na compositions of CrystaLLM's held-out test set, never seen
          by any model; --targets kg: 100 from the KG corpus test split (KG CIFs come from MP, which is in
          CrystaLLM's training data -- base loss on them is ~0.24, so this set is partly seen).
          --samples each, every model in --models, same seeds; scored like B2 (valid, composition, KG
          score, strict match to the reference, CHGNet dE) -> b3_eval_<targets>_<model>.json (CPU scoring)
  report  paired bootstrap over compositions, each model vs base and vs SFT -> b3_report_<targets>.json

Acceptance, fixed before any result: DPO counts as a gain if the KG-plausible share rises (CI low > 0)
while the strict-match share and validity do not fall (CI low > -5 points), vs base and vs SFT.

    python -u abox/kg_dpo.py --stage sft
    python -u abox/kg_dpo.py --stage dpo
    python -u abox/kg_dpo.py --stage eval --models crystallm_v1_small,crystallm_kg_sft,crystallm_kg_dpo
    python -u abox/kg_dpo.py --stage report
"""
import argparse
import copy
import gzip
import json
import math
import os
import random
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import phaseC_generate as gen  # noqa: E402
import dpo_pairs as b2  # noqa: E402  (scoring functions shared with B2)
import kg_corpus  # noqa: E402

DATA = gen.DATA


def corpus(split):
    return json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))


def encode(tok, cif):
    # CrystaLLM's own layout (kg_corpus.layout); a training CIF ends with a blank line, which is where
    # generation stops
    return tok.encode(tok.tokenize_cif(kg_corpus.layout(cif).strip() + "\n\n"))


def no_dropout(model):
    """CrystaLLM's attention applies dropout_p=config.dropout even in eval mode; zero it (and the rest) so
    log-probs are deterministic -- needed for DPO, where policy and reference must agree at the start."""
    for m in model.modules():
        if isinstance(getattr(m, "dropout", None), float):
            m.dropout = 0.0
        if m.__class__.__name__ == "Dropout":
            m.p = 0.0
    return model


def save(model, name, extra=None):
    os.makedirs(os.path.join(DATA, name), exist_ok=True)
    import torch
    torch.save({"model_args": {k: getattr(model.config, k) for k in
                               ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout")},
                "model": model.state_dict(), **(extra or {})}, os.path.join(DATA, name, "ckpt.pt"))


def batches(seqs, size, rng):
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [seqs[j] for j in idx[i:i + size]]


def pad(seqs, device, fill=0):
    import torch
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), fill, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def lm_loss(model, seqs, device):
    import torch
    x = pad([s[:-1] for s in seqs], device)
    y = pad([s[1:] for s in seqs], device, fill=-1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    return loss


def stage_sft(a):
    import torch
    torch.manual_seed(a.seed)
    tok, dev = gen.CIFTokenizer(), "cuda"
    model = gen.load_model(dev, "crystallm_v1_small").train()
    bs = model.config.block_size
    train = [s for s in (encode(tok, r["cif"]) for r in corpus("train")) if len(s) <= bs + 1][:a.limit]
    val = [s for s in (encode(tok, r["cif"]) for r in corpus("val")) if len(s) <= bs + 1][:a.limit]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr_sft, weight_decay=0.1, betas=(0.9, 0.95))

    def val_loss():
        model.eval()
        with torch.no_grad():
            ls = [lm_loss(model, b, dev).item() for b in batches(val, a.batch, random.Random(0))]
        model.train()
        return sum(ls) / len(ls)

    log, best, rng, t0 = [{"epoch": 0, "val_loss": val_loss()}], None, random.Random(a.seed), time.time()
    best = log[0]["val_loss"]
    print(f"SFT: {len(train)} train / {len(val)} val CIFs; base val loss {best:.4f}", flush=True)
    for ep in range(1, a.epochs_sft + 1):
        for b in batches(train, a.batch, rng):
            loss = lm_loss(model, b, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_loss()
        log.append({"epoch": ep, "val_loss": v})
        print(f"epoch {ep}: val loss {v:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if v < best:
            best = v
            save(model, a.sft_name, {"sft_log": log})
        elif ep > 1:
            break  # early stop at the first epoch that does not improve
    json.dump(log, open(os.path.join(DATA, "sft_train_log.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(DATA, a.sft_name, "ckpt.pt")):
        print("SFT never beat the base model's val loss; no SFT checkpoint written", flush=True)


def seq_logp(model, prompts, fulls, device):
    """Sum of log p(completion | prompt) per sequence (completion = tokens after the prompt)."""
    import torch
    x = pad([s[:-1] for s in fulls], device)
    y = pad([s[1:] for s in fulls], device, fill=-1)
    mask = torch.zeros_like(y, dtype=torch.float)
    for i, (p, s) in enumerate(zip(prompts, fulls)):
        mask[i, len(p) - 1: len(s) - 1] = 1.0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x, torch.zeros_like(y))  # targets given -> logits for every position
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def load_pairs(split, tok, bs):
    keep = {r["id"] for r in corpus(split)}
    out, dropped = [], 0
    for line in open(os.path.join(DATA, f"dpo_pairs_{split}.jsonl"), encoding="utf-8"):
        p = json.loads(line)
        if p["id"] not in keep:
            dropped += 1
            continue
        pr = tok.encode(tok.tokenize_cif(p["prompt"]))
        ch, rj = encode(tok, p["chosen"]), encode(tok, p["rejected"])
        if max(len(ch), len(rj)) <= bs + 1:
            out.append((pr, ch, rj))
    return out, dropped


def dpo_loss(policy, ref, batch, beta, dev):
    import torch
    import torch.nn.functional as F
    prompts = [p for p, _, _ in batch]
    pc, pr = seq_logp(policy, prompts, [c for _, c, _ in batch], dev), seq_logp(policy, prompts, [r for _, _, r in batch], dev)
    with torch.no_grad():
        rc, rr = seq_logp(ref, prompts, [c for _, c, _ in batch], dev), seq_logp(ref, prompts, [r for _, _, r in batch], dev)
    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean()


def stage_dpo(a):
    import torch
    torch.manual_seed(a.seed)
    tok, dev = gen.CIFTokenizer(), "cuda"
    # no dropout in either model, so policy and reference log-probs agree at the start (loss = ln 2)
    policy = no_dropout(gen.load_model(dev, a.init))
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    bs = policy.config.block_size
    train, d_tr = load_pairs("train", tok, bs)
    val, d_va = load_pairs("val", tok, bs)
    train, val = train[:a.limit], val[:a.limit]
    print(f"DPO from {a.init}: {len(train)} train / {len(val)} val pairs "
          f"(dropped, id not in rebuilt corpus: {d_tr} / {d_va})", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr_dpo, weight_decay=0.0)

    def evaluate():
        with torch.no_grad():
            r = [dpo_loss(policy, ref, b, a.beta, dev) for b in batches(val, a.batch_dpo, random.Random(0))]
        return sum(x[0].item() for x in r) / len(r), sum(x[1].item() for x in r) / len(r)

    l0, acc0 = evaluate()
    log, best, rng, t0, step = [{"epoch": 0, "val_loss": l0, "val_acc": acc0}], l0, random.Random(a.seed), time.time(), 0
    print(f"start: val DPO loss {l0:.4f} (ln2 = {math.log(2):.4f}), pref acc {acc0:.3f}", flush=True)
    for ep in range(1, a.epochs_dpo + 1):
        for b in batches(train, a.batch_dpo, rng):
            loss, _ = dpo_loss(policy, ref, b, a.beta, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
        l, acc = evaluate()
        log.append({"epoch": ep, "steps": step, "val_loss": l, "val_acc": acc})
        print(f"epoch {ep}: val DPO loss {l:.4f}, pref acc {acc:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if l < best:
            best = l
            save(policy, a.dpo_name, {"dpo_log": log, "init": a.init, "beta": a.beta})
    json.dump(log, open(os.path.join(DATA, f"{a.dpo_name}_train_log.json"), "w"), indent=1)
    if best == l0:
        print("DPO never improved the val loss; no DPO checkpoint written", flush=True)


def eval_targets(which, n, max_atoms, seed=23):
    """crystallm: Li/Na compositions of CrystaLLM's own held-out test set (phaseC_test_with_pred.json) --
    the primary set, since KG materials come from MP and CrystaLLM's training data contains MP (base loss
    on KG CIFs is only ~0.24: largely seen). kg: the KG corpus test split (never seen by SFT/DPO)."""
    if which == "crystallm":
        rows = [t for t in json.load(open(os.path.join(DATA, "phaseC_test_with_pred.json"), encoding="utf-8"))
                if t["n_atoms"] <= max_atoms]
        random.Random(seed).shuffle(rows)
        return [{"id": t["id"], "formula": t["formula"], "cell": t["cell"], "ref": gen.load_truth(t).as_dict()}
                for t in rows[:n]]
    rows = [r for r in corpus("test") if r["n_atoms"] <= max_atoms]
    random.Random(seed).shuffle(rows)
    return [{"id": r["id"], "formula": r["formula"], "cell": b2.cell_of(r["cif"]),
             "ref": gen.reference_from_text(r["cif"]).as_dict()} for r in rows[:n]]


def stage_eval(a):
    import torch
    targets = eval_targets(a.targets, a.n_eval, a.max_atoms)
    tok = gen.CIFTokenizer()
    for name in a.models.split(","):
        out = os.path.join(DATA, f"b3_eval_{a.targets}_{name}.json")
        if os.path.exists(out):
            print(f"{name}: exists, skipped", flush=True)
            continue
        if not os.path.exists(os.path.join(DATA, name, "ckpt.pt")):
            print(f"{name}: no checkpoint (training did not improve on its start), skipped", flush=True)
            continue
        gdir = os.path.join(DATA, "b3_gen", a.targets, name)
        model = gen.load_model("cuda", name)
        t0 = time.time()
        for i, r in enumerate(targets):
            os.makedirs(os.path.join(gdir, r["id"]), exist_ok=True)
            if os.path.exists(os.path.join(gdir, r["id"], f"s_{a.samples - 1}.cif")):
                continue
            torch.manual_seed(a.seed + sum(map(ord, r["id"])))  # same seed for every model
            cifs = []
            while len(cifs) < a.samples:
                cifs += gen.sample_batch(model, tok, gen.prompt_for(r["cell"]),
                                         min(a.batch_gen, a.samples - len(cifs)), "cuda")
            for k, s in enumerate(cifs):
                open(os.path.join(gdir, r["id"], f"s_{k}.cif"), "w").write(s)
            if (i + 1) % 20 == 0:
                print(f"{name}: {i + 1}/{len(targets)} generated ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()
        score(a, name, targets, gdir, out)


def score(a, name, targets, gdir, out):
    from pymatgen.core import Composition
    jobs, refs = [], {}
    for r in targets:
        target = Composition(r["cell"]).reduced_formula
        refs[r["id"]] = r["ref"]
        jobs += [(os.path.join(gdir, r["id"], f"s_{k}.cif"), target, refs[r["id"]]) for k in range(a.samples)]
    with ProcessPoolExecutor(a.workers) as ex:
        kg = dict(ex.map(b2._kg, jobs, chunksize=4))
    relax = [(f"{r['id']}|truth", refs[r["id"]], False) for r in targets]
    relax += [(f"{r['id']}|{k}", open(os.path.join(gdir, r["id"], f"s_{k}.cif")).read(), True)
              for r in targets for k in range(a.samples)
              if kg[os.path.join(gdir, r["id"], f"s_{k}.cif")].get("composition_ok")]
    t0, energies = time.time(), {}
    with ProcessPoolExecutor(a.chgnet_workers, initializer=b2._init_chgnet, initargs=(a.chgnet_threads,)) as ex:
        for i, (key, e) in enumerate(ex.map(b2._relax, relax, chunksize=2)):
            energies[key] = e
            if (i + 1) % 200 == 0:
                print(f"{name}: relaxed {i + 1}/{len(relax)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = []
    for r in targets:
        et = energies.get(f"{r['id']}|truth")
        samples = []
        for k in range(a.samples):
            s = dict(kg[os.path.join(gdir, r["id"], f"s_{k}.cif")])
            e = energies.get(f"{r['id']}|{k}")
            if e is not None and et is not None:
                s["dE"] = round(e - et, 5)
            samples.append({"k": k, **s})
        res.append({"id": r["id"], "formula": r["formula"], "e_true": et, "samples": samples})
    json.dump(res, open(out, "w"), indent=1)
    print(f"{name}: scored -> {out}", flush=True)


def per_comp(m):
    s = m["samples"]
    n = len(s)
    ok = [x for x in s if x.get("composition_ok") and "total" in x]
    return {"valid": sum(bool(x.get("valid")) for x in s) / n,
            "composition": len(ok) / n,
            "kg_plausible": sum(bool(x.get("plausible")) for x in ok) / n,   # share of ALL samples
            "strict": sum(bool(x.get("strict")) for x in ok) / n,
            "found": float(any(x.get("strict") for x in ok)),
            "stable_vs_kg": sum(x.get("dE", 9) <= 0.05 for x in ok) / n}


def stage_report(a):
    f = lambda n: os.path.join(DATA, f"b3_eval_{a.targets}_{n}.json")
    names = [n for n in a.models.split(",") if os.path.exists(f(n))]
    data = {n: {m["id"]: per_comp(m) for m in json.load(open(f(n)))} for n in names}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    metrics = list(next(iter(data[names[0]].values())).keys())
    rng = random.Random(0)
    boots = [[rng.choice(ids) for _ in ids] for _ in range(10000)]
    rep = {"compositions": len(ids), "means": {}, "vs": {}}
    for n in names:
        rep["means"][n] = {k: round(100 * sum(data[n][i][k] for i in ids) / len(ids), 1) for k in metrics}
    for base in (names[0], "crystallm_kg_sft"):
        if base not in names:
            continue
        for n in names:
            if n == base or (base != names[0] and n == names[0]):
                continue
            row = {}
            for k in metrics:
                d = {i: data[n][i][k] - data[base][i][k] for i in ids}
                bs = sorted(100 * sum(d[i] for i in b) / len(b) for b in boots)
                row[k] = {"diff": round(100 * sum(d.values()) / len(ids), 1),
                          "ci95": [round(bs[250], 1), round(bs[9749], 1)],
                          "better": sum(v > 0 for v in d.values()), "worse": sum(v < 0 for v in d.values())}
            rep["vs"][f"{n} - {base}"] = row
    json.dump(rep, open(os.path.join(DATA, f"b3_report_{a.targets}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sft", "dpo", "eval", "report"])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--batch", type=int, default=8, help="SFT batch (CIFs)")
    ap.add_argument("--lr-sft", type=float, default=3e-5)
    ap.add_argument("--epochs-sft", type=int, default=3)
    ap.add_argument("--sft-name", default="crystallm_kg_sft")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap CIFs / pairs per split")
    ap.add_argument("--init", default="crystallm_kg_sft", help="DPO start + reference model")
    ap.add_argument("--dpo-name", default="crystallm_kg_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr-dpo", type=float, default=1e-6)
    ap.add_argument("--epochs-dpo", type=int, default=2)
    ap.add_argument("--batch-dpo", type=int, default=4, help="pairs per step")
    ap.add_argument("--models", default="crystallm_v1_small,crystallm_kg_sft,crystallm_kg_dpo")
    ap.add_argument("--targets", default="crystallm", choices=["crystallm", "kg"])
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-atoms", type=int, default=40)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch-gen", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chgnet-workers", type=int, default=4)
    ap.add_argument("--chgnet-threads", type=int, default=4)
    a = ap.parse_args()
    {"sft": stage_sft, "dpo": stage_dpo, "eval": stage_eval, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
