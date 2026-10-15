"""K-Ont as a trained analogue retriever (BUILD_LOG Step 40).

K-Ont's material vector, as trained, does not say which framework an unseen composition takes (Step 39,
Step 40 ablation). Here the K-Ont encoder is fine-tuned for exactly that question, on the KG: for a
target material, given its structure-free sentence (formula, chemistry, properties, and the cell's site
counts, which the generation prompt also gives) and the full sentences of KG candidates with the same
anonymous formula, rank highest the candidates with the target's space group (listwise softmax,
multi-positive). Trained only on materials K-Ont saw, with only K-Ont-seen candidates (kg_pool.py); scored on every
K-Ont-unseen target, so nothing evaluated was ever trained on.
Two inits, same data and steps: the trained K-Ont model (data/runs/full_grow5000_v2/final) and the
plain OnT galen encoder it started from -- the difference is what K-Ont's KG training contributes.

    .venv/Scripts/python.exe -u abox/kg_retriever_ft.py --init kont
    .venv/Scripts/python.exe -u abox/kg_retriever_ft.py --init base
Output: data/crystallm/kg_ft_scores_<init>.json  {target: {candidate: score}}
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
sys.path.insert(0, os.path.join(HERE, "abox"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ont.hit import HierarchyTransformer  # noqa: E402
from phase5_checks import STRUCT_LINES  # noqa: E402

DATA = os.path.join(HERE, "data", "crystallm")
INITS = {"kont": os.path.join(HERE, "data", "runs", "full_grow5000_v2", "final"), "base": "Hui97/OnT-MiniLM-L12-galen"}


def target_text(full):
    return "\n".join(l for l in full.split("\n") if not l.startswith(STRUCT_LINES) or l.startswith("Sites:"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=list(INITS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--targets-per-step", type=int, default=4)
    ap.add_argument("--cands", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    full = {k.split("/")[-1]: v for k, v in vm.items()}
    import kg_pool
    pool, byid = kg_pool.load()
    cands_of = lambda t: [byid[c] for c in pool["candidates"][t["id"]]]
    tr = []
    for i in pool["seen_train"]:  # Step 40 pool: train on K-Ont-seen materials only, candidates K-Ont-seen
        t = byid[i]
        cs = cands_of(t)
        pos = [c for c in cs if c["space_group"] == t["space_group"]]
        if pos and len(pos) < len(cs):
            tr.append((t, cs))
    print(f"{a.init}: {len(tr)} training targets (K-Ont-seen, mixed candidates)", flush=True)

    dev = "cuda"
    model = HierarchyTransformer.from_pretrained(INITS[a.init]).to(dev)
    model.max_seq_length = 256
    model.train()
    logit_scale = torch.nn.Parameter(torch.tensor(3.0, device=dev))  # exp(3) = 20 = 1 / 0.05
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": a.lr}, {"params": [logit_scale], "lr": 1e-2}], weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.targets_per_step - 1) // a.targets_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))

    def emb(texts):
        f = model.tokenize(texts)
        f = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in f.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            e = model(f)["sentence_embedding"]
        return F.normalize(e.float(), dim=-1)

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        random.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), a.targets_per_step):
            batch = tr[i:i + a.targets_per_step]
            texts_t, texts_c, groups = [], [], []
            for t, cs in batch:
                pos = [c for c in cs if c["space_group"] == t["space_group"]]
                neg = [c for c in cs if c["space_group"] != t["space_group"]]
                k = min(len(pos), max(1, a.cands // 3))
                pick = random.sample(pos, k) + random.sample(neg, min(len(neg), a.cands - k))
                texts_t.append(target_text(full[t["id"]]))
                groups.append((len(texts_c), [c["space_group"] == t["space_group"] for c in pick]))
                texts_c += [full[c["id"]] for c in pick]
            et, ec = emb(texts_t), emb(texts_c)
            loss = 0
            for j, (off, lab) in enumerate(groups):
                s = logit_scale.exp() * (ec[off:off + len(lab)] @ et[j])
                lab = torch.tensor(lab, device=dev)
                loss = loss + torch.logsumexp(s, 0) - torch.logsumexp(s[lab], 0)
            loss = loss / len(groups)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps} loss {tot / ((i // a.targets_per_step) + 1):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"{a.init} epoch {ep + 1}: mean loss {tot / ((len(tr) + a.targets_per_step - 1) // a.targets_per_step):.3f}", flush=True)

    model.eval()
    scores = {}
    targets = [byid[i] for i in pool["targets"]]
    need_c = sorted({c["id"] for t in targets for c in cands_of(t)})
    with torch.no_grad():
        ec = {}
        for i in range(0, len(need_c), 64):
            e = emb([full[c] for c in need_c[i:i + 64]])
            ec.update(zip(need_c[i:i + 64], e))
        for i in range(0, len(targets), 64):
            b = targets[i:i + 64]
            e = emb([target_text(full[t["id"]]) for t in b])
            for t, v in zip(b, e):
                scores[t["id"]] = {c["id"]: float(ec[c["id"]] @ v) for c in cands_of(t)}
    json.dump(scores, open(os.path.join(DATA, f"kg_ft_scores_{a.init}.json"), "w"))
    print(f"{a.init}: scored {len(scores)} unseen targets ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
