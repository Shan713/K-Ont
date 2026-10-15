"""K-Ont-conditioned crystal generation: CrystaLLM (small) with the material's K-Ont vector as a soft prompt.

The exported K-Ont vector of a material (abox/export_embeddings.py, `material_cond_tan`: the material
sentence WITHOUT structure lines -- formula, chemistry, properties -- encoded by the trained K-Ont model and
mapped to the tangent space) is projected to --prefix-len token embeddings and placed in front of the CIF
tokens. CrystaLLM and the projection are fine-tuned together on the KG corpus (CrystaLLM layout).

Arms, identical except for what the prefix sees (same data, steps, seeds, lr):
  kg      prefix = proj(K-Ont vector of this material)
  const   prefix = proj(0) -- a learned constant prefix: the same architecture and fine-tuning, no K-Ont information
At evaluation the kg model is also run with
  kg_shuffled   another material's vector -- if kg beats this, the gain comes from the vector's content
and the untouched base model is the reference.

Evaluation set: KG-corpus test materials that are ALSO held out from K-Ont training (split_grow_5000.json),
so neither the generator nor K-Ont saw them. Composition prompt only (data_<cell>). Per sample: valid,
right composition, strict match to the KG structure (StructureMatcher, CrystaLLM tolerances), KG
plausibility. Paired bootstrap over compositions. Acceptance (fixed now): kg > const on strict-match share
with CI low > 0, and kg > kg_shuffled.

    .venv-crystallm/Scripts/python.exe -u abox/kg_condition.py --stage train --arm kg
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition.py --stage train --arm const
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition.py --stage eval
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition.py --stage report
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import phaseC_generate as gen  # noqa: E402
import dpo_pairs as b2  # noqa: E402
import kg_corpus  # noqa: E402

DATA = gen.DATA
EXPORT = os.path.join(DATA, "..", "kont_export")


class Conditioned(nn.Module):
    """CrystaLLM GPT + a prefix projection. forward(prefix_vec, idx) -> logits for the idx positions."""

    def __init__(self, gpt, in_dim, prefix_len):
        super().__init__()
        self.gpt, self.k = gpt, prefix_len
        d = gpt.config.n_embd
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, prefix_len * d))

    def forward(self, vec, idx):
        g = self.gpt.transformer
        b, t = idx.shape
        pre = self.proj(vec).view(b, self.k, -1)
        x = torch.cat([pre, g.wte(idx)], dim=1)
        pos = torch.arange(0, t + self.k, device=idx.device)
        x = g.drop(x + g.wpe(pos)[None])
        for block in g.h:
            x = block(x)
        return self.gpt.lm_head(g.ln_f(x))[:, self.k:]


def vectors(name):
    z = np.load(os.path.join(EXPORT, name, "embeddings.npz"))["material_cond_tan"]
    ids = json.load(open(os.path.join(EXPORT, name, "id_maps.json")))["material"]
    z = z / np.median(np.linalg.norm(z, axis=1))
    return {i: torch.tensor(v, dtype=torch.float32) for i, v in zip(ids, z)}


def corpus(split):
    return json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))


def encode(tok, cif):
    return tok.encode(tok.tokenize_cif(kg_corpus.layout(cif).strip() + "\n\n"))


def batch_loss(model, rows, vecs, arm, dev):
    seqs = [r[1] for r in rows]
    T = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    y = torch.full((len(seqs), T), -1, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    v = torch.stack([vecs[r[0]] for r in rows]) if arm == "kg" else torch.zeros(len(rows), next(iter(vecs.values())).shape[0])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(v.to(dev), x.to(dev))
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.to(dev).reshape(-1), ignore_index=-1)


def stage_train(a):
    torch.manual_seed(a.seed)
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = vectors(a.export)
    gpt = gen.load_model(dev, "crystallm_v1_small")
    model = Conditioned(gpt, next(iter(vecs.values())).shape[0], a.prefix_len).to(dev).train()
    lim = gpt.config.block_size - a.prefix_len + 1
    data = {s: [(r["id"], e) for r in corpus(s) for e in [encode(tok, r["cif"])] if len(e) <= lim and r["id"] in vecs]
            for s in ("train", "val")}
    opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": a.lr},
                             {"params": model.proj.parameters(), "lr": a.lr_proj}], weight_decay=0.1)

    def val_loss():
        model.eval()
        with torch.no_grad():
            ls = [batch_loss(model, data["val"][i:i + a.batch], vecs, a.arm, dev).item()
                  for i in range(0, len(data["val"]), a.batch)]
        model.train()
        return sum(ls) / len(ls)

    rng, t0 = random.Random(a.seed), time.time()
    best, log = val_loss(), []
    log.append({"epoch": 0, "val_loss": best})
    print(f"{a.arm}: {len(data['train'])} train / {len(data['val'])} val; start val loss {best:.4f}", flush=True)
    out = os.path.join(DATA, f"kgcond_{a.arm}.pt")
    torch.save(model.state_dict(), out)
    for ep in range(1, a.epochs + 1):
        rows = data["train"][:]
        rng.shuffle(rows)
        for i in range(0, len(rows), a.batch):
            loss = batch_loss(model, rows[i:i + a.batch], vecs, a.arm, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_loss()
        log.append({"epoch": ep, "val_loss": v})
        print(f"{a.arm} epoch {ep}: val loss {v:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if v < best:
            best = v
            torch.save(model.state_dict(), out)
    json.dump(log, open(os.path.join(DATA, f"kgcond_{a.arm}_log.json"), "w"), indent=1)


@torch.no_grad()
def sample(model, tok, prompt, vec, n, dev, max_new=1000, temperature=0.8, top_k=10):
    """gen.sample_batch with a prefix vector (same temperature / top-k / blank-line stop)."""
    ids = torch.tensor(tok.encode(tok.tokenize_cif(prompt)), dtype=torch.long, device=dev)
    x = ids[None].repeat(n, 1)
    v = vec[None].repeat(n, 1).to(dev)
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, device=dev)
    lim = model.gpt.config.block_size - model.k
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            if x.size(1) >= lim:
                break
            logits = model(v, x)[:, -1, :].float() / temperature
            kth, _ = torch.topk(logits, top_k)
            logits[logits < kth[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, nl), nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            done |= (prev == nl) & (nxt == nl)
            prev = nxt
            if done.all():
                break
    return [tok.decode(r.tolist()).split("\n\n")[0] + "\n" for r in x]


def _cached_run(gpt, h, cache, start):
    """One forward over new positions h (embeddings, no positions yet) reusing per-layer keys/values.
    Same arithmetic as CrystaLLM's Block/CausalSelfAttention, including its attention dropout, which it
    applies even in eval mode (dropout_p=config.dropout) -- kept so sampling matches gen.sample_batch."""
    g = gpt.transformer
    h = h + g.wpe(torch.arange(start, start + h.size(1), device=h.device))[None]
    h = g.drop(h)
    for li, blk in enumerate(g.h):
        at = blk.attn
        B, T, C = h.shape
        q, k, v = at.c_attn(blk.ln_1(h)).split(at.n_embd, dim=2)
        q, k, v = (z.view(B, T, at.n_head, C // at.n_head).transpose(1, 2) for z in (q, k, v))
        if cache[li] is not None:
            k, v = torch.cat([cache[li][0], k], 2), torch.cat([cache[li][1], v], 2)
        cache[li] = (k, v)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=at.dropout, is_causal=T > 1)
        h = h + at.resid_dropout(at.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        h = h + blk.mlp(blk.ln_2(h))
    return gpt.lm_head(g.ln_f(h[:, -1]))


@torch.no_grad()
def sample_cached(gpt, tok, prompt, n, dev, prefix=None, max_new=1000, temperature=0.8, top_k=10):
    """sample() / gen.sample_batch with a key-value cache: identical sampling (temperature, top-k, stop at a
    blank line), each new token costs one position instead of the whole sequence. prefix: [k, n_embd]
    soft-prompt embeddings (None for the base model)."""
    ids = torch.tensor(tok.encode(tok.tokenize_cif(prompt)), dtype=torch.long, device=dev)
    wte = gpt.transformer.wte
    h = wte(ids)[None].repeat(n, 1, 1)
    if prefix is not None:
        h = torch.cat([prefix[None].repeat(n, 1, 1), h], 1)
    out = ids[None].repeat(n, 1)
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, device=dev)
    cache, pos = [None] * len(gpt.transformer.h), 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            if pos + h.size(1) > gpt.config.block_size:
                break
            logits = _cached_run(gpt, h, cache, pos).float() / temperature
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


def stage_check_cache(a):
    """The cached forward must give the same next-token logits as the full forward (dropout off)."""
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = vectors(a.export)
    m = Conditioned(gen.load_model(dev, "crystallm_v1_small"), next(iter(vecs.values())).shape[0], a.prefix_len).to(dev)
    m.load_state_dict(torch.load(os.path.join(DATA, "kgcond_kg.pt"), map_location=dev))
    m.eval()
    for blk in m.gpt.transformer.h:
        blk.attn.dropout = 0.0
    r = corpus("test")[0]
    seq = torch.tensor(encode(tok, r["cif"])[:200], device=dev)[None]
    v = vecs[r["id"]][None].to(dev)
    with torch.no_grad():
        full = m(v, seq)[0].float()
        pre = m.proj(v).view(1, m.k, -1)
        cache = [None] * len(m.gpt.transformer.h)
        wte = m.gpt.transformer.wte
        step = [_cached_run(m.gpt, torch.cat([pre, wte(seq[:, :1])], 1), cache, 0).float()]
        for t in range(1, seq.size(1)):
            step.append(_cached_run(m.gpt, wte(seq[:, t:t + 1]), cache, m.k + t).float())
        step = torch.cat(step)
    diff = (full - step).abs().max().item()
    agree = (full.argmax(-1) == step.argmax(-1)).float().mean().item()
    print(f"cached vs full forward over 200 tokens: max |logit diff| {diff:.2e}, same top token {agree:.3f}")


def eval_targets(a):
    kont_test = {m.split("/")[-1] for m in json.load(open(os.path.join(HERE, "data", "split_grow_5000.json")))["test_materials"]}
    rows = [r for r in corpus("test") if r["id"] in kont_test]
    return [{"id": r["id"], "formula": r["formula"], "cell": b2.cell_of(r["cif"]),
             "ref": gen.reference_from_text(r["cif"]).as_dict()} for r in rows]


def stage_eval(a):
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = vectors(a.export)
    targets = eval_targets(a)
    ids = [t["id"] for t in targets]
    shuffled = dict(zip(ids, ids[1:] + ids[:1]))  # each target gets the next target's vector
    print(f"{len(targets)} targets (unseen by K-Ont and by the generator)", flush=True)
    dim = next(iter(vecs.values())).shape[0]
    for arm in a.arms.split(","):
        out = os.path.join(DATA, f"kgcond_eval_{arm}.json")
        if os.path.exists(out):
            continue
        gdir = os.path.join(DATA, "kgcond_gen", arm)
        if arm == "base":
            m = gen.load_model(dev, "crystallm_v1_small")
        else:
            m = Conditioned(gen.load_model(dev, "crystallm_v1_small"), dim, a.prefix_len).to(dev)
            m.load_state_dict(torch.load(os.path.join(DATA, f"kgcond_{'kg' if arm == 'kg_shuffled' else arm}.pt"), map_location=dev))
        m.eval()
        t0 = time.time()
        for i, t in enumerate(targets):
            os.makedirs(os.path.join(gdir, t["id"]), exist_ok=True)
            torch.manual_seed(a.seed + sum(map(ord, t["id"])))
            prompt = gen.prompt_for(t["cell"])
            cifs = []
            while len(cifs) < a.samples:
                n = min(a.batch_gen, a.samples - len(cifs))
                # one sampler (key-value cached) for every arm
                if arm == "base":
                    cifs += sample_cached(m, tok, prompt, n, dev)
                else:
                    v = vecs[shuffled[t["id"]]] if arm == "kg_shuffled" else (vecs[t["id"]] if arm == "kg" else torch.zeros(dim))
                    with torch.no_grad():
                        pre = m.proj(v[None].to(dev)).view(m.k, -1)
                    cifs += sample_cached(m.gpt, tok, prompt, n, dev, prefix=pre)
            for k, s in enumerate(cifs):
                open(os.path.join(gdir, t["id"], f"s_{k}.cif"), "w").write(s)
            if (i + 1) % 20 == 0:
                print(f"{arm}: {i + 1}/{len(targets)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del m
        torch.cuda.empty_cache()
        from pymatgen.core import Composition
        jobs = [(os.path.join(gdir, t["id"], f"s_{k}.cif"), Composition(t["cell"]).reduced_formula, t["ref"])
                for t in targets for k in range(a.samples)]
        with ProcessPoolExecutor(a.workers) as ex:
            kg = dict(ex.map(b2._kg, jobs, chunksize=4))
        res = [{"id": t["id"], "formula": t["formula"],
                "samples": [kg[os.path.join(gdir, t["id"], f"s_{k}.cif")] for k in range(a.samples)]} for t in targets]
        json.dump(res, open(out, "w"), indent=1)
        print(f"{arm}: scored", flush=True)


def per_comp(m):
    s = m["samples"]
    n = len(s)
    ok = [x for x in s if x.get("composition_ok") and "total" in x]
    return {"valid": sum(bool(x.get("valid")) for x in s) / n, "composition": len(ok) / n,
            "strict": sum(bool(x.get("strict")) for x in ok) / n, "found": float(any(x.get("strict") for x in ok)),
            "kg_plausible": sum(bool(x.get("plausible")) for x in ok) / n}


def stage_report(a):
    arms = [x for x in a.arms.split(",") if os.path.exists(os.path.join(DATA, f"kgcond_eval_{x}.json"))]
    data = {x: {m["id"]: per_comp(m) for m in json.load(open(os.path.join(DATA, f"kgcond_eval_{x}.json")))} for x in arms}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    keys = list(next(iter(data[arms[0]].values())).keys())
    rng = random.Random(0)
    boots = [[rng.choice(ids) for _ in ids] for _ in range(10000)]
    rep = {"compositions": len(ids), "samples_per_composition": a.samples,
           "means": {x: {k: round(100 * sum(data[x][i][k] for i in ids) / len(ids), 1) for k in keys} for x in arms},
           "kg_vs": {}}
    for other in [x for x in arms if x != "kg"]:
        if "kg" not in arms:
            break
        row = {}
        for k in keys:
            d = {i: data["kg"][i][k] - data[other][i][k] for i in ids}
            bs = sorted(100 * sum(d[i] for i in b) / len(b) for b in boots)
            row[k] = {"diff": round(100 * sum(d.values()) / len(ids), 1), "ci95": [round(bs[250], 1), round(bs[9749], 1)],
                      "better": sum(v > 0 for v in d.values()), "worse": sum(v < 0 for v in d.values())}
        rep["kg_vs"][other] = row
    json.dump(rep, open(os.path.join(DATA, "kgcond_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval", "report", "check_cache"])
    ap.add_argument("--arm", default="kg", choices=["kg", "const"])
    ap.add_argument("--arms", default="base,const,kg,kg_shuffled")
    ap.add_argument("--export", default="grow5000_v2")
    ap.add_argument("--prefix-len", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--batch-gen", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    {"train": stage_train, "eval": stage_eval, "report": stage_report, "check_cache": stage_check_cache}[a.stage](a)


if __name__ == "__main__":
    main()
