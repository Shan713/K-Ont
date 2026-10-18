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


MINIMAL = ("Material:", "Type:", "Formula:", "Chemical system:", "Sites:")


def target_text(full, minimal=False):
    """structure-free target sentence; minimal (Step 44, novel compositions): only what a new composition
    has -- name, type, formula, chemical system and the prompt's cell (Sites line), no computed properties"""
    if minimal:
        return "\n".join(l for l in full.split("\n") if l.startswith(MINIMAL))
    return "\n".join(l for l in full.split("\n") if not l.startswith(STRUCT_LINES) or l.startswith("Sites:"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=list(INITS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--targets-per-step", type=int, default=4)
    ap.add_argument("--cands", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0, help="learning curve: share of training targets used")
    ap.add_argument("--tag", default="", help="output suffix (learning curve runs)")
    ap.add_argument("--retain", type=float, default=0.0,
                    help="Step 43: weight of a retention loss keeping the start model's embeddings of KG/ontology "
                         "sentences (MSE to the frozen start model), so K-Ont learns the analogy without forgetting")
    ap.add_argument("--minimal", action="store_true", help="Step 44: composition-only target sentences")
    ap.add_argument("--extra", default=None, help="Step 44: also score these targets (novel_pool.json)")
    ap.add_argument("--retain-galen", type=int, default=0,
                    help="Step 52: also keep this many GALEN concept names at the original OnT-galen embeddings")
    ap.add_argument("--retain-anchors", action="store_true",
                    help="Step 52c: also keep K-Ont's relation names and role/family label phrases, every step")
    ap.add_argument("--retain-balanced", action="store_true",
                    help="Step 52b: GALEN retention as its own 16-sentence term per step, alongside the KG's")
    ap.add_argument("--save", default=None, help="save the fine-tuned encoder here (Step 43 ability checks)")
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
    if a.frac < 1:
        tr = random.Random(a.seed).sample(tr, max(1, round(a.frac * len(tr))))
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

    def raw(texts):
        f = model.tokenize(texts)
        f = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in f.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(f)["sentence_embedding"].float()

    retain, retain_g, retain_a = None, None, None
    if a.retain > 0:  # frozen start-model embeddings of the sentences K-Ont was trained on
        ab = os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2")
        sents = list(full.values())
        sents += list(json.load(open(os.path.join(ab, "verbalizations_crystals_full.json"), encoding="utf-8")).values())
        sents += list(json.load(open(os.path.join(ab, "verbalizations_elements.json"), encoding="utf-8")).values())
        cn = os.path.join(INITS[a.init], "..", "data", "concept_names.json")
        if os.path.exists(cn):
            sents += list(json.load(open(cn)).values())
        model.eval()
        with torch.no_grad():
            teach = torch.cat([raw(sents[i:i + 64]) for i in range(0, len(sents), 64)])
        if a.retain_galen:  # Step 52: GALEN concept names, target = the ORIGINAL OnT-galen model's embeddings
            gn = json.load(open(os.path.join(HERE, "data", "ont_bench", "extracted", "GALEN_OnT", "concept_names.json"), encoding="utf-8"))
            gs = list(gn.values()) if isinstance(gn, dict) else list(gn)
            gs = random.Random(a.seed).sample(gs, min(a.retain_galen, len(gs)))
            ref = HierarchyTransformer.from_pretrained(INITS["base"]).to(dev).eval()
            with torch.no_grad():
                gt = []
                for i in range(0, len(gs), 64):
                    f = ref.tokenize(gs[i:i + 64])
                    f = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in f.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        gt.append(ref(f)["sentence_embedding"].float())
            del ref
            if a.retain_balanced:  # Step 52b: separate GALEN term, so 20k GALEN sentences cannot crowd out the KG's
                retain_g = (gs, torch.cat(gt))
            if a.retain_anchors:  # Step 52c: K-Ont's relation names and role/family label sentences, EVERY step --
                # GALEN retention pulls short phrases toward OnT-galen, and these drive K-Ont's role rotations
                from phase5_checks import ROLE_SENT, FAMILY_READABLE, FAMILY_CLASS
                rn = json.load(open(os.path.join(INITS[a.init], "..", "data", "role_names.json")))
                anchors = sorted(set(rn.values()) | set(ROLE_SENT.values()) | set(FAMILY_CLASS.values())
                                 | {f"Structure family: {v}" for v in FAMILY_READABLE.values()})
                model.eval()
                with torch.no_grad():
                    retain_a = (anchors, raw(anchors))
                print(f"  + {len(anchors)} relation/label anchor phrases kept every step", flush=True)
            else:
                sents, teach = sents + gs, torch.cat([teach] + gt)
            print(f"  + {len(gs)} GALEN concept names kept at OnT-galen's embeddings", flush=True)
        model.train()
        retain = (sents, teach)
        print(f"retention set: {len(sents)} sentences, weight {a.retain}", flush=True)

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
                texts_t.append(target_text(full[t["id"]], a.minimal))
                groups.append((len(texts_c), [c["space_group"] == t["space_group"] for c in pick]))
                texts_c += [full[c["id"]] for c in pick]
            et, ec = emb(texts_t), emb(texts_c)
            loss = 0
            for j, (off, lab) in enumerate(groups):
                s = logit_scale.exp() * (ec[off:off + len(lab)] @ et[j])
                lab = torch.tensor(lab, device=dev)
                loss = loss + torch.logsumexp(s, 0) - torch.logsumexp(s[lab], 0)
            loss = loss / len(groups)
            for rs in (retain, retain_g, retain_a):
                if rs is not None:
                    idx = random.sample(range(len(rs[0])), min(16, len(rs[0])))
                    loss = loss + a.retain * F.mse_loss(raw([rs[0][j] for j in idx]), rs[1][idx]) / rs[1][idx].pow(2).mean()
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
    if a.save:
        model.save(a.save)
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
            e = emb([target_text(full[t["id"]], a.minimal) for t in b])
            for t, v in zip(b, e):
                scores[t["id"]] = {c["id"]: float(ec[c["id"]] @ v) for c in cands_of(t)}
        if a.extra:  # Step 44: novel compositions, {id: {"text": minimal sentence, "candidates": [pool ids]}}
            ex = json.load(open(a.extra))
            need = sorted({c for v in ex.values() for c in v["candidates"]} - set(ec))
            for i in range(0, len(need), 64):
                ec.update(zip(need[i:i + 64], emb([full[c] for c in need[i:i + 64]])))
            ids = sorted(ex)
            for i in range(0, len(ids), 64):
                e = emb([ex[k]["text"] for k in ids[i:i + 64]])
                for k, v in zip(ids[i:i + 64], e):
                    scores[k] = {c: float(ec[c] @ v) for c in ex[k]["candidates"]}
    json.dump(scores, open(os.path.join(DATA, f"kg_ft_scores_{a.init}{a.tag}.json"), "w"))
    print(f"{a.init}: scored {len(scores)} unseen targets ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
