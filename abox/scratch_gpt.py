"""A CrystaLLM-architecture GPT pre-trained from scratch WITHOUT the evaluation compositions (BUILD_LOG Step 45).

Step 40 found that injecting K-Ont into CrystaLLM can add at most ~2 points (true-framework ceiling 49.6 vs
47.5): CrystaLLM was pre-trained on Materials Project and already recalls the held-out structures. Here a
model of the same architecture (CrystaLLM small: 8 layers, 512 wide, 1,024 context, 25M parameters) is
pre-trained on CrystaLLM's own training split with every structure of every evaluation composition removed,
so it cannot recall them; Step 40's arms are then fine-tuned from it (kg_condition3.py --base scratch_small).

data      cifs_v1_train.pkl.gz (CrystaLLM authors' Zenodo record 10642388, md5-checked, read with
          safe_pickle: plain data only) -> drop every CIF whose reduced formula is that of a K-Ont-unseen
          KG material (all 1,018, so every Step 40-44 target) or a Step 28 novel composition -> random
          300,000 train + 5,000 val (seed 0), CrystaLLM layout (kg_corpus.layout), tokenized, <= 1,024
          tokens -> data/crystallm/scratch/{train,val}.bin (uint16) + meta.json
pretrain  packed token stream, random 1,024-token windows (as CrystaLLM), batch 16, AdamW 6e-4, 1,000-step
          warm-up, cosine to 6e-5, bf16; checkpoint every 1,000 steps (resumes after a restart or power cut);
          final weights in CrystaLLM's checkpoint format -> data/crystallm/scratch_small/ckpt.pt

    .venv-crystallm/Scripts/python.exe -u abox/scratch_gpt.py --stage data
    .venv-crystallm/Scripts/python.exe -u abox/scratch_gpt.py --stage pretrain
"""
import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))

import numpy as np  # noqa: E402

DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(DATA, "scratch")
SRC = os.path.join(DATA, "cifs_v1_download", "cifs_v1_train.pkl.gz")
FINAL = os.path.join(DATA, "scratch_small")


def banned_formulas():
    from pymatgen.core import Composition
    import kg_pool
    unseen = {m.split("/")[-1] for m in json.load(open(os.path.join(HERE, "data", "split_grow_5000.json")))["test_materials"]}
    _, rows = kg_pool.load()
    ban = {Composition(r["formula"]).reduced_formula for i, r in rows.items() if i in unseen}
    ban |= {Composition(n["formula"]).reduced_formula for n in json.load(open(os.path.join(DATA, "novel_candidates.json")))}
    return ban


def _prep(args):
    cid, cif = args
    import re
    import phaseC_generate as gen
    import kg_condition as k1
    from pymatgen.core import Composition
    try:
        f = Composition(re.search(r"^data_(\S+)", cif, re.M).group(1)).reduced_formula
        toks = k1.encode(_TOK, cif)
        return cid, f, toks if len(toks) <= 1024 else None
    except Exception:
        return cid, None, None


def _init():
    global _TOK
    import phaseC_generate as gen
    _TOK = gen.CIFTokenizer()


def stage_data(a):
    import safe_pickle
    os.makedirs(OUT, exist_ok=True)
    ban = banned_formulas()
    d = safe_pickle.load(SRC)
    order = list(range(len(d)))
    random.Random(0).shuffle(order)
    want = a.n_train + a.n_val
    kept, dropped, too_long, bad = [], 0, 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for start in range(0, len(order), 20000):
            for cid, f, toks in ex.map(_prep, [d[i] for i in order[start:start + 20000]], chunksize=200):
                if f is None:
                    bad += 1
                elif f in ban:
                    dropped += 1
                elif toks is None:
                    too_long += 1
                else:
                    kept.append((cid, toks))
            print(f"  scanned {start + 20000:,}: kept {len(kept):,}, dropped (eval formula) {dropped:,} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if len(kept) >= want:
                break
    kept = kept[:want]
    val, train = kept[:a.n_val], kept[a.n_val:]
    for name, part in (("train", train), ("val", val)):
        arr = np.array([t for _, toks in part for t in toks], dtype=np.uint16)
        arr.tofile(os.path.join(OUT, f"{name}.bin"))
        print(name, len(part), "CIFs", len(arr), "tokens", flush=True)
    meta = {"source": "cifs_v1_train.pkl.gz (Zenodo 10642388, md5 54f695b7862b79ce653338f4161a120d)",
            "banned_reduced_formulas": len(ban), "scanned": start + 20000, "dropped_eval_formula": dropped,
            "too_long": too_long, "unparsable": bad, "train_cifs": len(train), "val_cifs": len(val),
            "train_ids_sample": [c for c, _ in train[:20]]}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    json.dump(sorted(ban), open(os.path.join(OUT, "banned_formulas.json"), "w"))
    print(json.dumps(meta, indent=1))


def stage_pretrain(a):
    import torch
    import phaseC_generate as gen
    from crystallm import GPT, GPTConfig
    torch.manual_seed(a.seed)
    args = dict(torch.load(os.path.join(DATA, "crystallm_v1_small", "ckpt.pt"), map_location="cpu")["model_args"])
    args["dropout"] = 0.0
    train = np.fromfile(os.path.join(OUT, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(OUT, "val.bin"), dtype=np.uint16)
    T, B = args["block_size"], a.batch
    steps = int(a.epochs * len(train) / (B * T))
    print(f"{len(train):,} train tokens; {steps:,} steps of {B}x{T} ({a.epochs} epochs)", flush=True)
    dev = "cuda"
    model = GPT(GPTConfig(**args)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ck_path = os.path.join(OUT, "ckpt_train.pt")
    step, log = 0, []
    if os.path.exists(ck_path):  # resume
        ck = torch.load(ck_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed at step {step:,}", flush=True)
    gen_ = np.random.default_rng(a.seed + step)

    def batch(src):
        ix = gen_.integers(0, len(src) - T - 1, B)
        x = torch.from_numpy(np.stack([src[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([src[i + 1:i + T + 1] for i in ix]).astype(np.int64))
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr / 10 + 0.5 * (a.lr - a.lr / 10) * (1 + math.cos(math.pi * min(1.0, p)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = []
        for _ in range(50):
            x, y = batch(val)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls.append(model(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "log": log,
                    "rng": torch.get_rng_state()}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    model.train()
    t0, tl = time.time(), 0.0
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(train)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
        step += 1
        if step % 1000 == 0 or step == steps:
            rec = {"step": step, "train_loss": round(tl / 1000, 4), "val_loss": round(val_loss(), 4), "lr": lr_at(step),
                   "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(rec, flush=True)
            tl = 0.0
            save()
    os.makedirs(FINAL, exist_ok=True)
    torch.save({"model_args": args, "model": model.state_dict(), "iter_num": step, "log": log,
                "note": "scratch pre-training without evaluation compositions (abox/scratch_gpt.py)"},
               os.path.join(FINAL, "ckpt.pt"))
    print("saved", FINAL, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "pretrain"])
    ap.add_argument("--n-train", type=int, default=300000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"data": stage_data, "pretrain": stage_pretrain}[a.stage](a)


if __name__ == "__main__":
    main()
