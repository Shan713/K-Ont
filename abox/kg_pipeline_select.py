"""A complete K-Ont pipeline that commits to ONE structure per composition (BUILD_LOG Step 48).

Steps 41/44 report "found within k attempts". A user needs one answer. For each of the 902 K-Ont-unseen
targets: candidates = the K-Ont ranker's top 3 templates (Step 41 substitution) + CrystaLLM's first 5 samples
(Step 40 plain fine-tune `const`, valid and right composition); each is relaxed briefly with CHGNet (fmax 0.1,
--steps, cell free) and the lowest energy per atom is the pick. Right = the pick matches the reference
(StructureMatcher 0.2 / 0.3 / 5). Compared, at the same 8-candidate budget where relevant:
  CrystaLLM x8, lowest energy        selection by the same physics, no K-Ont
  K-Ont template top-1               the ranker's first choice, no physics
  3 templates + 5 CrystaLLM, lowest energy   (the pipeline)
  oracle: any of the 8 right         (ceiling of selection)
--subset 300 --steps 40: a random 300 of the 902 (seed 0), 40 relaxation steps (laptop time). Crash-tolerant and resumable (robust_pool); energies cached in data/crystallm/kg_pipeline_energies.json.

    .venv-crystallm/Scripts/python.exe abox/kg_pipeline_select.py [--workers 2 --steps 60]
"""
import argparse
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
CACHE = os.path.join(DATA, "kg_pipeline_energies.json")
OUT = os.path.join(DATA, "kg_pipeline_select.json")
GEN = os.path.join(DATA, "kgcond3_gen", "const")


def _init(threads, steps):
    global _M, _OPT, _STEPS
    import torch
    torch.set_num_threads(threads)
    from chgnet.model import CHGNet, StructOptimizer
    _M = CHGNet.load(verbose=False)
    _OPT = StructOptimizer(model=_M, use_device="cpu")
    _STEPS = steps


def _relax(job):
    key, kind, target, payload = job
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Structure
    try:
        ref = Structure.from_dict(target["ref"])
        if kind == "template":
            c = payload
            s = gen.reference_from_text(c["cif"])
            old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
            s.replace_species({a.symbol: b.symbol for a, b in zip(k2._ordered(c["formula"]), k2._ordered(target["formula"]))})
            new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
            s.scale_lattice(s.volume * new / old)
        else:
            s = Structure.from_str(gen.postprocess(open(payload).read()), fmt="cif")
        if s.composition.reduced_formula != ref.composition.reduced_formula:
            return key, None
        fs = _OPT.relax(s, fmax=0.1, steps=_STEPS, verbose=False)["final_structure"]
        e = float(_M.predict_structure(fs)["e"])
        right = bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(fs, ref) or
                     StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(s, ref))
        return key, [round(e, 5), right]
    except Exception:
        return key, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--subset", type=int, default=0, help="evaluate a random subset of targets (seed 0); 0 = all 902")
    a = ap.parse_args()
    import kg_pool
    import phaseC_generate as gen
    from robust_pool import robust_map
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    if a.subset:  # ~10 relaxations/min on the laptop CPU: all 902 x 11 candidates would take ~15 h
        targets = sorted(random.Random(0).sample(targets, a.subset))
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    tinfo = {t: {"formula": rows[t]["formula"], "ref": gen.reference_from_text(rows[t]["cif"]).as_dict()} for t in targets}
    jobs = []
    for t in targets:
        for c, _ in top8[t][:3]:
            jobs.append((f"{t}|T|{c}", "template", tinfo[t], rows[c]))
        for k in range(5):
            jobs.append((f"{t}|C|{k}", "crystallm", tinfo[t], os.path.join(GEN, t, f"s_{k}.cif")))
    for k in range(5, 8):  # CrystaLLM x8 needs samples 5-7 too
        for t in targets:
            jobs.append((f"{t}|C|{k}", "crystallm", tinfo[t], os.path.join(GEN, t, f"s_{k}.cif")))
    cache_p = CACHE.replace(".json", f"_s{a.steps}.json")  # one cache per relaxation length: never mix
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = [j for j in jobs if j[0] not in cache]
    print(f"{len(jobs)} candidates, {len(todo)} to relax ({a.steps} steps)", flush=True)

    def save(r):
        cache.update(r)
        json.dump(cache, open(cache_p, "w"))
    if todo:
        robust_map(_relax, todo, a.workers, save, initializer=_init, initargs=(1, a.steps), chunk=96,
                   log=lambda m: print(m, flush=True))

    def pick(keys):
        got = [cache[k] for k in keys if cache.get(k)]
        return float(min(got)[1]) if got else 0.0

    def anyr(keys):
        return float(any(cache.get(k) and cache[k][1] for k in keys))
    T = lambda t, n: [f"{t}|T|{c}" for c, _ in top8[t][:n]]
    C = lambda t, n: [f"{t}|C|{k}" for k in range(n)]
    arms = {"CrystaLLM x8, lowest energy": {t: pick(C(t, 8)) for t in targets},
            "CrystaLLM x5, lowest energy": {t: pick(C(t, 5)) for t in targets},
            "K-Ont template top-1 (no physics)": {t: anyr(T(t, 1)) for t in targets},
            "K-Ont templates x3, lowest energy": {t: pick(T(t, 3)) for t in targets},
            "3 templates + 5 CrystaLLM, lowest energy (pipeline)": {t: pick(T(t, 3) + C(t, 5)) for t in targets},
            "oracle: any of 3 templates + 5 CrystaLLM": {t: anyr(T(t, 3) + C(t, 5)) for t in targets},
            "oracle: any of CrystaLLM x8": {t: anyr(C(t, 8)) for t in targets}}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y):
        d = {t: arms[x][t] - arms[y][t] for t in targets}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n": len(targets), "subset_of_902": bool(a.subset), "relax_steps": a.steps,
           "right_single_pick": {k: round(100 * sum(v.values()) / len(targets), 1) for k, v in arms.items()},
           "pipeline - CrystaLLM x8 lowest energy": diff("3 templates + 5 CrystaLLM, lowest energy (pipeline)", "CrystaLLM x8, lowest energy"),
           "pipeline - K-Ont template top-1": diff("3 templates + 5 CrystaLLM, lowest energy (pipeline)", "K-Ont template top-1 (no physics)"),
           "K-Ont template top-1 - CrystaLLM x8 lowest energy": diff("K-Ont template top-1 (no physics)", "CrystaLLM x8, lowest energy")}
    json.dump({"report": rep}, open(OUT.replace(".json", f"_n{len(targets)}_s{a.steps}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
