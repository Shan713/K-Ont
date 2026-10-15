"""K-Ont-conditioned generation, strong version: a KG analogue's CRYSTAL vector, deep injection, guidance.

Step 37 (abox/kg_condition.py) injected the material's own property vector as a 4-token prefix: it added
almost nothing, because that vector repeats what the formula already says. This version changes what is
injected, where, and how strongly:

  WHAT   the K-Ont embedding of a KG analogue's crystal (`crystal_tan` from export_embeddings.py: space
         group, crystal system, lattice, volume, site counts) -- structure knowledge the prompt lacks.
         Training: each material is conditioned on a same-prototype neighbour (same anonymous formula and
         space group, most similar chemistry), so the model learns to follow the vector; materials without
         one get a composition-retrieved analogue. Test: the analogue is retrieved from the composition only
         (same anonymous formula, most similar elements, KG train split) -- never from the true structure.
  WHERE  an 8-token prefix AND a per-layer adapter (vector -> added to every hidden state before every
         block; zero-initialised, so training starts from CrystaLLM).
  HOW    classifier-free guidance: the condition is dropped (zero vector) for 15% of training examples;
         sampling uses logits_uncond + w (logits_cond - logits_uncond).

Arms (85 test materials held out from K-Ont and the generator, composition prompt, 16 samples, same seeds):
  deep_w1, deep_w2     analogue vector, guidance w = 1, 2
  deep_shuf_w2         another target's analogue vector, w = 2          (content control)
  text_sg              no vector; the analogue's space group in the prompt (Task A text seeding), on the
                       Step 37 no-K-Ont fine-tune
  + Step 37's base / const / kg results, same targets and seeds.
Per sample: valid, right composition, strict match to the true structure, framework compliance (the
generated structure fits the analogue's with elements anonymised, StructureMatcher.fit_anonymous).
Acceptance (fixed before results): deep (best w) vs const on compliance AND on true-structure share,
CI low > 0; deep vs deep_shuf on compliance, CI low > 0.

    .venv-crystallm/Scripts/python.exe -u abox/kg_condition2.py --stage analogues
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition2.py --stage train
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition2.py --stage eval
    .venv-crystallm/Scripts/python.exe -u abox/kg_condition2.py --stage report
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict
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
import kg_condition as k1  # noqa: E402

DATA = gen.DATA
ANALOGUES = os.path.join(DATA, "kgcond2_analogues.json")
TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split())


# ---------------------------------------------------------------- analogues
def _ordered(formula):
    from pymatgen.core import Composition
    c = Composition(formula).reduced_composition
    return sorted(c.elements, key=lambda e: (c[e], e.X if e.X == e.X else 0))  # amount, then electronegativity


def similarity(f1, f2):
    """Element-by-element chemical similarity of two formulas with the same anonymous formula."""
    s = 0
    for a, b in zip(_ordered(f1), _ordered(f2)):
        if a == b:
            s += 3
        elif a.group == b.group or (a.symbol in TM and b.symbol in TM):
            s += 2
        elif a.block == b.block:
            s += 1
    return s


def stage_analogues(a):
    from pymatgen.core import Composition
    anon = lambda f: Composition(f).anonymized_formula
    train = k1.corpus("train")
    by_anon, by_proto = defaultdict(list), defaultdict(list)
    for r in train:
        by_anon[anon(r["formula"])].append(r)
        by_proto[(anon(r["formula"]), r["space_group"])].append(r)

    def best(cands, formula, exclude):
        cands = [c for c in cands if c["id"] not in exclude and c["formula"] != formula]
        if not cands:
            return None
        return max(cands, key=lambda c: (similarity(formula, c["formula"]), -(c["e_above_hull"] or 0)))["id"]

    out = {"train": {}, "val": {}, "test": {}, "how": {}}
    for split in ("train", "val"):
        for r in k1.corpus(split):
            p = best(by_proto.get((anon(r["formula"]), r["space_group"]), []), r["formula"], {r["id"]})
            c = p or best(by_anon.get(anon(r["formula"]), []), r["formula"], {r["id"]})
            if c:
                out[split][r["id"]] = c
                out["how"][r["id"]] = "prototype" if p else "composition"
    for t in k1.eval_targets(a):
        c = best(by_anon.get(anon(t["formula"]), []), t["formula"], set())  # composition only
        if c:
            out["test"][t["id"]] = c
    json.dump(out, open(ANALOGUES, "w"), indent=1)
    hows = list(out["how"].values())
    print(f"train {len(out['train'])}, val {len(out['val'])} conditioned "
          f"({hows.count('prototype')} prototype, {hows.count('composition')} composition); "
          f"test {len(out['test'])}/{len(k1.eval_targets(a))} with an analogue")


# ---------------------------------------------------------------- model
class Deep(nn.Module):
    """CrystaLLM + prefix + per-layer additive adapters, all driven by one condition vector."""

    def __init__(self, gpt, in_dim, k, gated=False):
        super().__init__()
        self.gpt, self.k = gpt, k
        d, L = gpt.config.n_embd, gpt.config.n_layer
        self.norm = nn.LayerNorm(in_dim)
        self.prefix = nn.Linear(in_dim, k * d)
        self.adapters = nn.ModuleList(nn.Linear(in_dim, d) for _ in range(L))
        self.gated = gated
        if gated:  # Step 38: tanh gates start closed (output = CrystaLLM), open only as far as they help
            self.gates = nn.Parameter(torch.zeros(L))
        else:  # Step 38 v1: zero-initialised adapters, no gate
            for ad in self.adapters:
                nn.init.zeros_(ad.weight)
                nn.init.zeros_(ad.bias)

    def cond(self, vec):
        z = self.norm(vec)
        adds = [ad(z)[:, None] for ad in self.adapters]
        if self.gated:
            adds = [torch.tanh(g) * x for g, x in zip(self.gates, adds)]
        return self.prefix(z).view(len(vec), self.k, -1), adds

    def forward(self, vec, idx):
        g = self.gpt.transformer
        pre, adds = self.cond(vec)
        x = torch.cat([pre, g.wte(idx)], 1)
        x = g.drop(x + g.wpe(torch.arange(x.size(1), device=idx.device))[None])
        for blk, add in zip(g.h, adds):
            x = blk(x + add)
        return self.gpt.lm_head(g.ln_f(x))[:, self.k:]


def crystal_vectors(export, key="crystal"):
    z = np.load(os.path.join(k1.EXPORT, export, "embeddings.npz"))[key + "_tan"]
    ids = json.load(open(os.path.join(k1.EXPORT, export, "id_maps.json")))["crystal"]
    z = z / np.median(np.linalg.norm(z, axis=1))
    return {i: torch.tensor(v, dtype=torch.float32) for i, v in zip(ids, z)}


def stage_train(a):
    torch.manual_seed(a.seed)
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = crystal_vectors(a.export, a.vectors)
    an = json.load(open(ANALOGUES))
    dim = next(iter(vecs.values())).shape[0]
    model = Deep(gen.load_model(dev, "crystallm_v1_small"), dim, a.prefix_len, a.gated).to(dev).train()
    lim = model.gpt.config.block_size - a.prefix_len + 1
    data = {s: [(an[s].get(r["id"]), e) for r in k1.corpus(s) for e in [k1.encode(tok, r["cif"])] if len(e) <= lim]
            for s in ("train", "val")}
    zero = torch.zeros(dim)
    rng = random.Random(a.seed)

    def loss_of(rows, drop):
        seqs = [s for _, s in rows]
        T = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        y = torch.full((len(seqs), T), -1, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])
            y[i, :len(s) - 1] = torch.tensor(s[1:])
        v = torch.stack([vecs[c] if c and not (drop and rng.random() < a.p_drop) else zero for c, _ in rows])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(v.to(dev), x.to(dev))
        return F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.to(dev).reshape(-1), ignore_index=-1)

    opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": a.lr},
                             {"params": [p for n, p in model.named_parameters() if not n.startswith("gpt.")], "lr": a.lr_cond}],
                            weight_decay=0.1)

    def val_loss():
        model.eval()
        with torch.no_grad():
            ls = [loss_of(data["val"][i:i + a.batch], False).item() for i in range(0, len(data["val"]), a.batch)]
        model.train()
        return sum(ls) / len(ls)

    out = os.path.join(DATA, f"kgcond2_deep{a.tag}.pt")
    best, t0, log = val_loss(), time.time(), []
    log.append({"epoch": 0, "val_loss": best})
    torch.save(model.state_dict(), out)
    print(f"deep: {len(data['train'])} train / {len(data['val'])} val; start val loss {best:.4f}", flush=True)
    for ep in range(1, a.epochs + 1):
        rows = data["train"][:]
        rng.shuffle(rows)
        for i in range(0, len(rows), a.batch):
            loss = loss_of(rows[i:i + a.batch], True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_loss()
        log.append({"epoch": ep, "val_loss": v})
        print(f"deep epoch {ep}: val loss {v:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if v < best:
            best = v
            torch.save(model.state_dict(), out)
    json.dump(log, open(os.path.join(DATA, f"kgcond2_deep{a.tag}_log.json"), "w"), indent=1)


# ---------------------------------------------------------------- sampling
def _run(gpt, h, cache, start, adds):
    """kg_condition._cached_run with the per-layer additive adapters."""
    g = gpt.transformer
    h = g.drop(h + g.wpe(torch.arange(start, start + h.size(1), device=h.device))[None])
    for li, blk in enumerate(g.h):
        h = h + adds[li]
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
def sample_cfg(model, tok, prompt, vec, n, dev, w, max_new=1000, temperature=0.8, top_k=10):
    """n samples; conditioned and unconditioned streams run as one batch of 2n with one shared token
    stream; logits = u + w (c - u). w = 1 is plain conditional sampling."""
    gpt = model.gpt
    ids = torch.tensor(tok.encode(tok.tokenize_cif(prompt)), dtype=torch.long, device=dev)
    vecs = torch.stack([vec] * n + [torch.zeros_like(vec)] * n).to(dev)
    pre, adds = model.cond(vecs)
    wte = gpt.transformer.wte
    h = torch.cat([pre, wte(ids)[None].repeat(2 * n, 1, 1)], 1)
    out = ids[None].repeat(n, 1)
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, device=dev)
    cache, pos = [None] * len(gpt.transformer.h), 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            if pos + h.size(1) > gpt.config.block_size:
                break
            lg = _run(gpt, h, cache, pos, adds).float()
            pos += h.size(1)
            c, u = lg[:n], lg[n:]
            logits = (u + w * (c - u)) / temperature
            kth, _ = torch.topk(logits, top_k)
            logits[logits < kth[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, nl), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= (prev == nl) & (nxt == nl)
            prev = nxt
            if done.all():
                break
            h = wte(torch.cat([nxt, nxt]))[:, None]
    return [tok.decode(r.tolist()).split("\n\n")[0] + "\n" for r in out]


# ---------------------------------------------------------------- evaluation
def _score(args):
    path, target, ref, analogue = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    path_row = b2._kg((path, target, ref))[1]
    if path_row.get("composition_ok") and analogue is not None:
        try:
            s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
            path_row["complies"] = bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit_anonymous(
                s, Structure.from_dict(analogue)))
        except Exception:
            path_row["complies"] = False
    return path, path_row


def analogue_structures():
    rows = {r["id"]: r for s in ("train",) for r in k1.corpus(s)}
    an = json.load(open(ANALOGUES))["test"]
    return {t: gen.reference_from_text(rows[c]["cif"]).as_dict() for t, c in an.items()}, \
        {t: rows[c]["space_group"] for t, c in an.items()}, an


def stage_eval(a):
    tok, dev = gen.CIFTokenizer(), "cuda"
    vecs = crystal_vectors(a.export, a.vectors)
    targets = k1.eval_targets(a)
    an_struct, an_sg, an = analogue_structures()
    ids = [t["id"] for t in targets]
    have = [i for i in ids if i in an]
    shuf = dict(zip(have, have[1:] + have[:1]))
    dim = next(iter(vecs.values())).shape[0]
    print(f"{len(targets)} targets, {len(an)} with an analogue", flush=True)
    for arm in a.arms.split(","):
        out = os.path.join(DATA, f"kgcond_eval_{arm}.json")
        if not os.path.exists(out):
            gdir = os.path.join(DATA, "kgcond_gen", arm)
            if arm == "text_sg":
                m = k1.Conditioned(gen.load_model(dev, "crystallm_v1_small"), dim, 4).to(dev)
                m.load_state_dict(torch.load(os.path.join(DATA, "kgcond_const.pt"), map_location=dev))
                with torch.no_grad():
                    pre = m.proj(torch.zeros(1, dim, device=dev)).view(m.k, -1)
            else:
                m = Deep(gen.load_model(dev, "crystallm_v1_small"), dim, a.prefix_len, a.gated).to(dev)
                m.load_state_dict(torch.load(os.path.join(DATA, f"kgcond2_deep{a.tag}.pt"), map_location=dev))
                w = float(arm.split("_w")[1])
            m.eval()
            t0 = time.time()
            for i, t in enumerate(targets):
                os.makedirs(os.path.join(gdir, t["id"]), exist_ok=True)
                torch.manual_seed(a.seed + sum(map(ord, t["id"])))
                if arm == "text_sg":
                    sg = gen.sg_symbol(an_sg[t["id"]]) if t["id"] in an else None
                    cifs = k1.sample_cached(m.gpt, tok, gen.prompt_for(t["cell"], sg), a.samples, dev, prefix=pre)
                else:
                    src = shuf.get(t["id"]) if "shuf" in arm else t["id"]
                    v = vecs[an[src]] if src in an else torch.zeros(dim)
                    cifs = sample_cfg(m, tok, gen.prompt_for(t["cell"]), v, a.samples, dev, w)
                for k, s in enumerate(cifs):
                    open(os.path.join(gdir, t["id"], f"s_{k}.cif"), "w").write(s)
                if (i + 1) % 20 == 0:
                    print(f"{arm}: {i + 1}/{len(targets)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            del m
            torch.cuda.empty_cache()
        score_arm(a, arm, targets, an_struct)


def score_arm(a, arm, targets, an_struct):
    """(Re)score an arm, adding framework compliance; also used for the Step 37 arms."""
    from pymatgen.core import Composition
    out = os.path.join(DATA, f"kgcond_eval_{arm}.json")
    if os.path.exists(out) and "complies" in json.dumps(json.load(open(out))[:3]):
        return
    gdir = os.path.join(DATA, "kgcond_gen", arm)
    jobs = [(os.path.join(gdir, t["id"], f"s_{k}.cif"), Composition(t["cell"]).reduced_formula, t["ref"],
             an_struct.get(t["id"])) for t in targets for k in range(a.samples)]
    with ProcessPoolExecutor(a.workers) as ex:
        sc = dict(ex.map(_score, jobs, chunksize=4))
    res = [{"id": t["id"], "formula": t["formula"],
            "samples": [sc[os.path.join(gdir, t["id"], f"s_{k}.cif")] for k in range(a.samples)]} for t in targets]
    json.dump(res, open(out, "w"), indent=1)
    print(f"{arm}: scored", flush=True)


def per_comp(m):
    d = k1.per_comp(m)
    s = m["samples"]
    d["complies"] = sum(bool(x.get("complies")) for x in s) / len(s)
    return d


def stage_report(a):
    targets = k1.eval_targets(a)
    an_struct, _, an = analogue_structures()
    for arm in ("base", "const", "kg"):  # Step 37 arms: add compliance
        score_arm(a, arm, targets, an_struct)
    arms = [x for x in a.report_arms.split(",") if os.path.exists(os.path.join(DATA, f"kgcond_eval_{x}.json"))]
    data = {x: {m["id"]: per_comp(m) for m in json.load(open(os.path.join(DATA, f"kgcond_eval_{x}.json")))} for x in arms}
    sets = {"all": sorted(set.intersection(*(set(d) for d in data.values()))),
            "with_analogue": sorted(i for i in set.intersection(*(set(d) for d in data.values())) if i in an)}
    keys = ["valid", "composition", "strict", "found", "complies", "kg_plausible"]
    rng = random.Random(0)
    rep = {"arms": arms}
    for name, ids in sets.items():
        boots = [[rng.choice(ids) for _ in ids] for _ in range(10000)]
        r = {"n": len(ids), "means": {x: {k: round(100 * sum(data[x][i][k] for i in ids) / len(ids), 1) for k in keys}
                                       for x in arms}, "paired": {}}
        deep = [x for x in arms if "deep_w" in x]
        for d_arm in deep:
            for other in [x for x in arms if x != d_arm]:
                row = {}
                for k in ("strict", "complies", "composition", "valid"):
                    diff = {i: data[d_arm][i][k] - data[other][i][k] for i in ids}
                    bs = sorted(100 * sum(diff[i] for i in b) / len(b) for b in boots)
                    row[k] = {"diff": round(100 * sum(diff.values()) / len(ids), 1),
                              "ci95": [round(bs[250], 1), round(bs[9749], 1)],
                              "better": sum(v > 0 for v in diff.values()), "worse": sum(v < 0 for v in diff.values())}
                r["paired"][f"{d_arm} - {other}"] = row
        rep[name] = r
    json.dump(rep, open(os.path.join(DATA, f"kgcond2_report{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: (v["means"] if isinstance(v, dict) and "means" in v else v) for k, v in rep.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["analogues", "train", "eval", "report"])
    ap.add_argument("--export", default="grow5000_v2")
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-cond", type=float, default=1e-3)
    ap.add_argument("--tag", default="", help="Step 38 v2: 'v2' (checkpoint, arm and report names)")
    ap.add_argument("--vectors", default="crystal", choices=["crystal", "crystal_anon"])
    ap.add_argument("--gated", action="store_true")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--arms", default="deep_w1,deep_w2,deep_shuf_w2,text_sg")
    ap.add_argument("--report-arms", default="base,const,kg,text_sg,deep_shuf_w2,deep_w1,deep_w2")
    a = ap.parse_args()
    {"analogues": stage_analogues, "train": stage_train, "eval": stage_eval, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
