"""Template editor: a CrystaLLM that reads a K-Ont-retrieved template and writes the corrected structure
(BUILD_LOG Step 58; design in docs/TEMPLATE_EDITOR.md).

Sequence = compact draft of the substituted template (space group, cell, atom-site rows) + the target CIF in
CrystaLLM layout; loss on the target part only.

    .venv-crystallm/Scripts/python.exe -u abox/template_editor.py --stage pairs [--n-targets 60000 --workers 1]
"""
import argparse
import json
import os
import random
import re
import sys
import time
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
SRC = os.path.join(DATA, "cifs_v1_download", "cifs_v1_train.pkl.gz")
OUT = os.path.join(DATA, "editor")
DRAFT_KEYS = ("_symmetry_space_group_name_H-M", "_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")


def substitute(template_structure, template_formula, target_formula):
    """Target elements on the template's sites, volume scaled by atomic radii (as kg_template_gen)."""
    import kg_condition2 as k2
    from pymatgen.core import Element
    s = template_structure.copy()
    old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.replace_species({a.symbol: b.symbol for a, b in zip(k2._ordered(template_formula), k2._ordered(target_formula))})
    new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.scale_lattice(s.volume * new / old)
    return s


def draft(structure):
    """Compact draft: space group + cell + symmetry-distinct atom-site rows, in CrystaLLM's CIF vocabulary."""
    import kg_corpus
    from pymatgen.io.cif import CifWriter
    cif = kg_corpus.layout(str(CifWriter(structure, symprec=0.1, significant_figures=4)))
    keep = []
    for line in cif.splitlines():
        f = line.split(" ")
        if f[0] in DRAFT_KEYS or (len(f) == 7 and f[2].isdigit() and f[-1] == "1"):
            keep.append(line)
    return "\n".join(keep) + "\n"


def _pair(job):
    """job = (target cif, template cif, kind) -> (kind, draft + target) or None."""
    tcif, ccif, kind = job
    try:
        import phaseC_generate as gen
        from pymatgen.core import Composition
        ft = Composition(re.search(r"^data_(\S+)", tcif, re.M).group(1)).reduced_formula
        fc = Composition(re.search(r"^data_(\S+)", ccif, re.M).group(1)).reduced_formula
        s = substitute(gen.reference_from_text(ccif), fc, ft)
        seq = draft(s) + tcif.strip() + "\n"
        n = len(_TOK.tokenize_cif(seq))
        return (kind, seq, n) if n <= 1024 else None
    except Exception:
        return None


def _init():
    global _TOK
    import phaseC_generate as gen
    _TOK = gen.CIFTokenizer()


def stage_pairs(a):
    import safe_pickle
    import kg_condition2 as k2
    from scratch_gpt import banned_formulas
    from pymatgen.core import Composition
    os.makedirs(OUT, exist_ok=True)
    ban = banned_formulas()
    d = safe_pickle.load(SRC)
    t0 = time.time()
    form, groups = [], {}
    for i, (cid, cif) in enumerate(d):
        try:
            c = Composition(re.search(r"^data_(\S+)", cif, re.M).group(1))
            f = c.reduced_formula
        except Exception:
            form.append(None)
            continue
        form.append(f)
        if f not in ban:
            groups.setdefault(c.anonymized_formula, []).append(i)
    print(f"{len(d):,} CIFs, {sum(len(v) for v in groups.values()):,} allowed in {len(groups):,} formula types "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    rng = random.Random(0)
    allowed = [i for g in groups.values() if len({form[j] for j in g}) > 1 for i in g]
    rng.shuffle(allowed)
    jobs = []
    for i in allowed[:a.n_targets]:
        an = Composition(form[i]).anonymized_formula
        others = [j for j in groups[an] if form[j] != form[i]]
        sample = rng.sample(others, min(200, len(others)))
        near = max(sample, key=lambda j: k2.similarity(form[i], form[j]))
        jobs.append((d[i][1], d[near][1], "near"))
        jobs.append((d[i][1], d[rng.choice(others)][1], "random"))
    del d
    print(f"{len(jobs):,} pairs to build", flush=True)
    path = os.path.join(OUT, "pairs.jsonl")
    kept = {"near": 0, "random": 0}
    _init()
    with open(path, "w", encoding="utf-8") as fh:
        if a.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            ex = ProcessPoolExecutor(a.workers, initializer=_init)
            it = ex.map(_pair, jobs, chunksize=64)
        else:
            it = map(_pair, jobs)
        for n, r in enumerate(it):
            if r:
                kept[r[0]] += 1
                fh.write(json.dumps({"kind": r[0], "seq": r[1], "tokens": r[2]}) + "\n")
            if (n + 1) % 5000 == 0:
                fh.flush()
                print(f"  {n + 1:,}/{len(jobs):,}: kept {kept} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump({"source": "cifs_v1_train.pkl.gz minus banned formulas (scratch_gpt.banned_formulas)",
               "n_targets": a.n_targets, "pairs_kept": kept, "max_tokens": 1024},
              open(os.path.join(OUT, "pairs_meta.json"), "w"), indent=1)
    print("done", kept, flush=True)


BASES = {"small": os.path.join(DATA, "crystallm_v1_small", "ckpt.pt"),     # pre-trained on Materials Project
         "scratch": os.path.join(DATA, "scratch_small", "ckpt.pt")}       # Step 45: never saw an eval composition


def load_gpt(path, dev):
    import torch
    from phaseC_generate import GPT, GPTConfig
    ck = torch.load(path, map_location=dev, weights_only=False)
    args = dict(ck["model_args"])
    args["dropout"] = 0.0
    m = GPT(GPTConfig(**args))
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    return m.to(dev), args


def stage_tokens(a):
    """pairs.jsonl -> token arrays + the draft length of each (the part with no loss)."""
    import numpy as np
    import phaseC_generate as gen
    tok = gen.CIFTokenizer()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "pairs.jsonl"), encoding="utf-8")]
    random.Random(1).shuffle(rows)
    toks, dlen, kinds = [], [], []
    for r in rows:
        i = r["seq"].index("data_")
        d = tok.encode(tok.tokenize_cif(r["seq"][:i]))
        t = tok.encode(tok.tokenize_cif(r["seq"][i:]))
        if len(d) + len(t) <= 1024:
            toks.append(np.array(d + t, dtype=np.uint16))
            dlen.append(len(d))
            kinds.append(r["kind"])
    off = np.cumsum([0] + [len(t) for t in toks])
    n_val = a.n_val
    np.savez(os.path.join(OUT, "tokens.npz"), flat=np.concatenate(toks), off=off, dlen=np.array(dlen),
             kind=np.array([k == "near" for k in kinds]), n_val=n_val)
    print(f"{len(toks):,} sequences ({n_val:,} validation), {off[-1]:,} tokens", flush=True)


def stage_train(a):
    import math
    import numpy as np
    import torch
    import phaseC_generate as gen
    torch.manual_seed(a.seed)
    dev = "cuda"
    z = np.load(os.path.join(OUT, "tokens.npz"))
    flat, off, dlen, n_val = z["flat"], z["off"], z["dlen"], int(z["n_val"])
    idx_val, idx_tr = np.arange(n_val), np.arange(n_val, len(dlen))
    nl = gen.CIFTokenizer().token_to_id["\n"]
    model, args = load_gpt(BASES[a.base], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = int(a.epochs * len(idx_tr) / a.batch)
    out_dir = os.path.join(DATA, f"editor_{a.base}")
    os.makedirs(out_dir, exist_ok=True)
    ck_path = os.path.join(out_dir, "ckpt_train.pt")
    step, log = 0, []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
        print(f"resumed at step {step:,}", flush=True)
    rng = np.random.default_rng(a.seed + step)

    def batch(ids):
        L = max(int(off[i + 1] - off[i]) for i in ids)
        x = np.full((len(ids), L), nl, dtype=np.int64)
        y = np.full((len(ids), L), -1, dtype=np.int64)
        for r, i in enumerate(ids):
            s = flat[off[i]:off[i + 1]].astype(np.int64)
            x[r, :len(s)] = s
            y[r, :len(s) - 1] = s[1:]
            y[r, :dlen[i] - 1] = -1  # no loss on the draft; the first predicted token is "data_"
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr / 10 + 0.5 * (a.lr - a.lr / 10) * (1 + math.cos(math.pi * min(1.0, p)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = []
        for i in range(0, min(n_val, 1600), a.batch):
            x, y = batch(idx_val[i:i + a.batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls.append(model(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    if step == 0:
        log.append({"step": 0, "val_loss": round(val_loss(), 4)})
        print(log[-1], flush=True)
    model.train()
    t0, tl = time.time(), 0.0
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(rng.choice(idx_tr, a.batch, replace=False))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
        step += 1
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "train_loss": round(tl / 500, 4), "val_loss": round(val_loss(), 4),
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            tl = 0.0
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "log": log},
                       ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
    torch.save({"model_args": args, "model": model.state_dict(), "log": log, "base": a.base,
                "note": "template editor (abox/template_editor.py)"}, os.path.join(out_dir, "ckpt.pt"))
    print("saved", out_dir, flush=True)


def sample(model, tok, prompt, n, dev, max_new=900, temperature=0.8, top_k=10):
    """n samples of one prompt, with a KV cache (kg_condition2._run, no adapters)."""
    import torch
    import torch.nn.functional as F
    import kg_condition2 as k2
    ids = torch.tensor([tok.encode(tok.tokenize_cif(prompt))] * n, dtype=torch.long, device=dev)
    wte = model.transformer.wte
    h, out = wte(ids), ids
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, device=dev)
    cache, pos, adds = [None] * len(model.transformer.h), 0, [0] * len(model.transformer.h)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            if pos + h.size(1) > model.config.block_size:
                break
            logits = k2._run(model, h, cache, pos, adds).float() / temperature
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
    texts = [tok.decode(r.tolist()) for r in out]
    return [t[t.index("data_"):].split("\n\n")[0] + "\n" for t in texts]


ORDER = [0, 1, 2, 0, 1, 2, 0, 1]  # draft used by sample k: found@k cycles over the top-3 templates


def stage_generate(a):
    """902 K-Ont-unseen targets: drafts from the same K-Ont ranker top-3 templates as Steps 41/48;
    8 samples per target (3 + 3 + 2 over the drafts), written as CIF text."""
    import kg_pool
    import phaseC_generate as gen
    pool, rows = kg_pool.load()
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    import dpo_pairs as b2
    dev = "cuda"
    model, _ = load_gpt(os.path.join(DATA, f"editor_{a.base}", "ckpt.pt"), dev)
    model.eval()
    tok = gen.CIFTokenizer()
    out_p = os.path.join(OUT, f"gen_{a.base}.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    t0 = time.time()
    for n, t in enumerate(pool["targets"]):
        if t in res:
            continue
        from pymatgen.core import Composition
        ft = Composition(rows[t]["formula"]).reduced_formula
        prompt_tail = gen.prompt_for(b2.cell_of(rows[t]["cif"]))  # the cell composition, as Step 40's prompts
        samples = [None] * 8
        for j, (c, _) in enumerate(top8[t][:3]):
            ks = [k for k, d in enumerate(ORDER) if d == j]
            try:
                fc = Composition(rows[c]["formula"]).reduced_formula
                dr = draft(substitute(gen.reference_from_text(rows[c]["cif"]), fc, ft))
                for k, s in zip(ks, sample(model, tok, dr + prompt_tail, len(ks), dev)):
                    samples[k] = s
            except Exception as e:
                print(t, c, "failed:", repr(e)[:120], flush=True)
        res[t] = samples
        if n % 25 == 0:
            json.dump(res, open(out_p, "w"))
            print(f"{n + 1}/{len(pool['targets'])} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump(res, open(out_p, "w"))


def stop_at_composition(cif):
    """The editor (trained without an end-of-structure marker) keeps writing atom sites after the cell is full:
    keep the atom-site rows until their multiplicities add up exactly to the data_ line's cell composition."""
    from pymatgen.core import Composition
    lines = cif.splitlines()
    want = Composition(re.search(r"^data_(\S+)", cif, re.M).group(1)).as_dict()
    have, out = {}, []
    for line in lines:
        f = line.split()
        if len(f) == 7 and f[2].isdigit() and f[0] in want:
            have[f[0]] = have.get(f[0], 0) + int(f[2])
            if have[f[0]] > want[f[0]]:
                return None
            out.append(line)
            if all(abs(have.get(e, 0) - n) < 1e-6 for e, n in want.items()):
                return "\n".join(out) + "\n"
        elif len(f) == 7 and f[2].isdigit():
            return None
        else:
            out.append(line)
    return None


def _match(job):
    t, ref_cif, cifs = job
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    ref = gen.reference_from_text(ref_cif)
    out = []
    for c in cifs:
        try:
            c = stop_at_composition(c)
            if c is None:
                out.append([False, False])
                continue
            # the editor writes the indented layout of CrystaLLM's training CIFs: collapse it, as reference_from_text
            s = gen.reference_from_text(c)
            ok = s.composition.reduced_formula == ref.composition.reduced_formula
            out.append([True, bool(ok and sm.fit(s, ref))])
        except Exception:
            out.append([False, False])
    return t, out


def stage_eval(a):
    import kg_pool
    from concurrent.futures import ProcessPoolExecutor
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    res = json.load(open(os.path.join(OUT, f"gen_{a.base}.json")))
    jobs = [(t, rows[t]["cif"], [c or "" for c in res[t]]) for t in targets]
    with ProcessPoolExecutor(a.workers) as ex:
        m = dict(ex.map(_match, jobs, chunksize=8))
    tg = json.load(open(os.path.join(DATA, "kg_template_gen.json")))["report"]
    found = {k: round(100 * sum(any(r for _, r in m[t][:k]) for t in targets) / len(targets), 1) for k in (1, 2, 3, 5, 8)}
    rep = {"base": a.base, "n_targets": len(targets),
           "valid_%": round(100 * sum(v for t in targets for v, _ in m[t]) / (8 * len(targets)), 1),
           "editor found@k": found,
           "template found@k (Step 41)": {k: tg["prompt+kg+ft_kont"].get(f"found@{k}") for k in (1, 2, 3, 5, 8)},
           "CrystaLLM found@k (Step 41)": {k: tg["CrystaLLM fine-tuned"].get(f"found@{k}") for k in (1, 2, 3, 5, 8)}}
    json.dump({"report": rep, "per_target": m}, open(os.path.join(OUT, f"eval_{a.base}.json"), "w"))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pairs", "tokens", "train", "generate", "eval"])
    ap.add_argument("--n-targets", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base", choices=list(BASES), default="scratch")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"pairs": stage_pairs, "tokens": stage_tokens, "train": stage_train, "generate": stage_generate,
     "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
