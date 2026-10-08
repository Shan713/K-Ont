"""Task B2: preference pairs for KG-reward fine-tuning (DPO) of CrystaLLM.

For KG compositions from the B1 corpus (train and val splits, <= --max-atoms): generate --samples CIFs
each with CrystaLLM (small by default; composition prompt; seeded), then score every sample:
  valid, right composition, KG plausibility (kg_steer.kg_score), strict match to the KG's own structure,
  CHGNet relaxed energy minus the KG structure's relaxed energy (dE, eV/atom)
Reward, fixed before looking at any result:
  invalid or wrong composition -> -100 (always the worst)
  otherwise r = -kg_total - 10 x max(0, dE)
Pairs per composition: highest-r vs lowest-r sample, if they differ by more than 0.5.

Stages (resumable; each writes its own file):
  generate  (GPU)  data/crystallm/dpo_gen/<id>/s_<k>.cif + dpo_generated_<split>.json
  score     (CPU)  dpo_scored_<split>.json   (KG scores in parallel; CHGNet in --chgnet-workers processes)
  pairs            dpo_pairs_<split>.jsonl + dpo_stats_<split>.json

    python -u abox/dpo_pairs.py --stage generate --split train --n 600
    python -u abox/dpo_pairs.py --stage score --split train
    python -u abox/dpo_pairs.py --stage pairs --split train
"""
import argparse
import gzip
import json
import os
import random
import re
import statistics
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import phaseC_generate as gen  # noqa: E402

DATA = gen.DATA
GEN_DIR = os.path.join(DATA, "dpo_gen")
_CHG = None


def pick(split, n, max_atoms, seed=11):
    """Up to n compositions: every family-labelled one first, then role-labelled, then the rest."""
    rows = json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))
    rows = [r for r in rows if r["n_atoms"] <= max_atoms]
    rng = random.Random(seed)
    fam = [r for r in rows if r["structure_family"]]
    role = [r for r in rows if not r["structure_family"] and r["battery_role"]]
    rest = [r for r in rows if not r["structure_family"] and not r["battery_role"]]
    for g in (fam, role, rest):
        rng.shuffle(g)
    return (fam + role + rest)[:n]


def cell_of(cif):
    return re.search(r"^data_(\S+)", cif, re.M).group(1)


def stage_generate(a):
    comps = pick(a.split, a.n, a.max_atoms)
    import torch
    model, tok = gen.load_model("cuda", a.model), gen.CIFTokenizer()
    out_path = os.path.join(DATA, f"dpo_generated_{a.split}.json")
    done = {m["id"]: m for m in json.load(open(out_path))} if os.path.exists(out_path) else {}
    t0 = time.time()
    for i, c in enumerate(comps):
        if c["id"] in done:
            continue
        torch.manual_seed(a.seed + sum(map(ord, c["id"])))
        prompt = gen.prompt_for(cell_of(c["cif"]))
        cifs = []
        while len(cifs) < a.samples:
            cifs += gen.sample_batch(model, tok, prompt, min(a.batch, a.samples - len(cifs)), "cuda")
        os.makedirs(os.path.join(GEN_DIR, c["id"]), exist_ok=True)
        for k, s in enumerate(cifs):
            open(os.path.join(GEN_DIR, c["id"], f"s_{k}.cif"), "w").write(s)
        done[c["id"]] = {"id": c["id"], "formula": c["formula"], "prompt": prompt, "n": len(cifs),
                         "battery_role": c["battery_role"], "structure_family": c["structure_family"]}
        if (i + 1) % 10 == 0 or i + 1 == len(comps):
            json.dump(list(done.values()), open(out_path, "w"), indent=1)
            print(f"{i + 1}/{len(comps)} generated  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump(list(done.values()), open(out_path, "w"), indent=1)


def _kg(args):
    path, target, truth = args
    import kg_steer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
    except Exception:
        return path, {"valid": False}
    row = {"valid": True, "composition_ok": s.composition.reduced_formula == target}
    if row["composition_ok"]:
        row.update(kg_steer.kg_score(s))
        row["strict"] = bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(s, Structure.from_dict(truth)))
    return path, row


def _init_chgnet(threads):
    global _CHG
    import torch
    torch.set_num_threads(threads)
    from chgnet.model import CHGNet, StructOptimizer
    m = CHGNet.load()
    _CHG = (m, StructOptimizer(model=m, use_device="cpu"))


def _relax(args):
    key, src, is_generated = args  # generated: raw CIF text; reference: Structure dict
    from pymatgen.core import Structure
    m, opt = _CHG
    try:
        s = Structure.from_str(gen.postprocess(src), fmt="cif") if is_generated else Structure.from_dict(src)
        r = opt.relax(s, fmax=0.1, steps=300, verbose=False)
        return key, float(m.predict_structure(r["final_structure"])["e"])
    except Exception:
        return key, None


def stage_score(a):
    gen_meta = json.load(open(os.path.join(DATA, f"dpo_generated_{a.split}.json")))
    corpus = {r["id"]: r for r in json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{a.split}.json.gz"), "rt", encoding="utf-8"))}
    from pymatgen.core import Composition
    # The reference is the KG structure rebuilt from its corpus CIF. A few corpus CIFs do not rebuild to
    # their own formula (symmetry expansion duplicates sites: mp-1296443 Li4Fe3CoO8 -> Li12Fe9Co3O32);
    # those compositions have no trustworthy reference and are dropped. References travel as dicts.
    jobs, refs, dropped = [], {}, []
    for m in gen_meta:
        target = Composition(cell_of(corpus[m["id"]]["cif"])).reduced_formula
        try:
            ref = gen.reference_from_text(corpus[m["id"]]["cif"])
        except Exception:
            ref = None
        if ref is None or ref.composition.reduced_formula != target:
            dropped.append(m["id"])
            continue
        refs[m["id"]] = ref.as_dict()
        jobs += [(os.path.join(GEN_DIR, m["id"], f"s_{k}.cif"), target, refs[m["id"]]) for k in range(m["n"])]
    gen_meta = [m for m in gen_meta if m["id"] in refs]
    print(f"{len(dropped)} compositions dropped, reference does not rebuild to its formula: {dropped}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        kg = dict(ex.map(_kg, jobs, chunksize=4))
    print(f"KG-scored {len(kg)} samples", flush=True)
    relax_jobs = []
    for m in gen_meta:
        relax_jobs.append((f"{m['id']}|truth", refs[m["id"]], False))
        for k in range(m["n"]):
            path = os.path.join(GEN_DIR, m["id"], f"s_{k}.cif")
            if kg[path].get("composition_ok"):
                relax_jobs.append((f"{m['id']}|{k}", open(path).read(), True))
    t0 = time.time()
    energies = {}
    with ProcessPoolExecutor(a.chgnet_workers, initializer=_init_chgnet, initargs=(a.chgnet_threads,)) as ex:
        for i, (key, e) in enumerate(ex.map(_relax, relax_jobs, chunksize=2)):
            energies[key] = e
            if (i + 1) % 200 == 0:
                print(f"relaxed {i + 1}/{len(relax_jobs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    scored = []
    for m in gen_meta:
        e_true = energies.get(f"{m['id']}|truth")
        samples = []
        for k in range(m["n"]):
            r = dict(kg[os.path.join(GEN_DIR, m["id"], f"s_{k}.cif")])
            e = energies.get(f"{m['id']}|{k}")
            if e is not None and e_true is not None:
                r["dE"] = round(e - e_true, 5)
            samples.append({"k": k, **r})
        scored.append({**m, "e_true": e_true, "samples": samples})
    json.dump(scored, open(os.path.join(DATA, f"dpo_scored_{a.split}.json"), "w"), indent=1)
    print(f"scored {len(scored)} compositions", flush=True)


def reward(r):
    if not r.get("valid") or not r.get("composition_ok") or "total" not in r:
        return -100.0
    return -r["total"] - 10 * max(0.0, r.get("dE", 0.5))  # no CHGNet energy -> treated as dE 0.5


def stage_pairs(a):
    scored = json.load(open(os.path.join(DATA, f"dpo_scored_{a.split}.json")))
    pairs, rewards, skipped = [], [], Counter()
    for m in scored:
        rs = [(reward(s), s) for s in m["samples"]]
        rewards += [r for r, _ in rs]
        hi, lo = max(rs, key=lambda x: x[0]), min(rs, key=lambda x: x[0])
        if hi[0] - lo[0] <= 0.5:
            skipped["rewards too close"] += 1
            continue
        read = lambda s: open(os.path.join(GEN_DIR, m["id"], f"s_{s['k']}.cif")).read()
        pairs.append({"id": m["id"], "formula": m["formula"], "prompt": m["prompt"],
                      "chosen": read(hi[1]), "rejected": read(lo[1]),
                      "r_chosen": round(hi[0], 3), "r_rejected": round(lo[0], 3),
                      "chosen_parts": {k: hi[1].get(k) for k in ("total", "plausible", "dE", "strict")},
                      "rejected_parts": {k: lo[1].get(k) for k in ("valid", "composition_ok", "total", "plausible", "dE", "strict")}})
    with open(os.path.join(DATA, f"dpo_pairs_{a.split}.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    samples = [s for m in scored for s in m["samples"]]
    ok = [s for s in samples if s.get("composition_ok") and "total" in s]
    stats = {"compositions": len(scored), "samples": len(samples), "pairs": len(pairs), "skipped": dict(skipped),
             "valid": round(sum(s["valid"] for s in samples) / max(len(samples), 1), 3),
             "right_composition": round(len(ok) / max(len(samples), 1), 3),
             "kg_plausible": round(sum(s["plausible"] for s in ok) / max(len(ok), 1), 3),
             "strict_match": round(sum(s["strict"] for s in ok) / max(len(ok), 1), 3),
             "median_dE": round(statistics.median([s["dE"] for s in ok if "dE" in s]), 4) if any("dE" in s for s in ok) else None,
             "reward_median_valid": round(statistics.median([r for r in rewards if r > -100]), 3) if any(r > -100 for r in rewards) else None,
             "reward_invalid_share": round(sum(r == -100 for r in rewards) / max(len(rewards), 1), 3),
             "reward": "invalid/wrong composition -100; else -kg_total - 10 max(0, dE)"}
    json.dump(stats, open(os.path.join(DATA, f"dpo_stats_{a.split}.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["generate", "score", "pairs"])
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-atoms", type=int, default=40)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="crystallm_v1_small")
    ap.add_argument("--workers", type=int, default=8, help="KG-scoring processes")
    ap.add_argument("--chgnet-workers", type=int, default=4)
    ap.add_argument("--chgnet-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    {"generate": stage_generate, "score": stage_score, "pairs": stage_pairs}[a.stage](a)


if __name__ == "__main__":
    main()
