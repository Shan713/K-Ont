"""A K-Ont-guided template generator, no neural decoder (BUILD_LOG Step 41).

For a target composition: the ranker (kg_rerank.py) orders the KG pool materials of the same formula type;
the generator takes the top-k frameworks and places the target's elements on their sites (element order
from kg_condition2._ordered, lattice scaled by atomic radii -- the same substitution as the Step 40
counterfactuals), no relaxation. A candidate is right when it matches the target's reference structure
(StructureMatcher ltol 0.2, stol 0.3, angle 5; scale and primitive cell as default).

Compared, on the same 902 K-Ont-unseen targets (kg_pool.py) with the same tolerances:
  CrystaLLM (Step 40 plain fine-tune, const): found within the first k of its samples
  template generator with each ranker: composition rule (no K-Ont), prompt+kg (KG facts, no K-Ont),
  prompt+kg+ft_kont (K-Ont retriever), prompt+kg+ft_base (plain-encoder retriever, the K-Ont control)
Metric: found@k (any of k candidates matches), k = 1, 2, 3, 5, 8 (rankers' top 8, kg_rerank_top8.json).
Paired bootstrap 95% CI. --relax also CHGNet-relaxes the K-Ont ranker's top 3 (fmax 0.1, 300 steps, cell
free) and re-scores them (relaxed template vs unrelaxed CrystaLLM: an extra, not the matched comparison).

    .venv-crystallm/Scripts/python.exe abox/kg_template_gen.py [--relax]
"""
import json
import os
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

DATA = os.path.join(HERE, "data", "crystallm")
RANKERS = ["composition rule", "prompt+kg", "prompt+kg+ft_kont", "prompt+kg+ft_base"]
KS = (1, 2, 3, 5, 8)
RELAX_K = 3  # --relax: CHGNet-relax the K-Ont ranker's top 3 (CPU)


def _init(threads):
    global _OPT
    import torch
    torch.set_num_threads(threads)
    from chgnet.model import CHGNet, StructOptimizer
    _OPT = StructOptimizer(model=CHGNet.load(verbose=False), use_device="cpu")


def _job(args):
    t, c, relax = args
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element
    try:
        ref = gen.reference_from_text(t["cif"])
        s = gen.reference_from_text(c["cif"])
        old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.replace_species({a.symbol: b.symbol for a, b in zip(k2._ordered(c["formula"]), k2._ordered(t["formula"]))})
        new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.scale_lattice(s.volume * new / old)
        if relax:
            s = _OPT.relax(s, fmax=0.1, steps=300, verbose=False)["final_structure"]
        ok = s.composition.reduced_formula == ref.composition.reduced_formula and \
            StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(s, ref)
        return (t["id"], c["id"]), bool(ok)
    except Exception:
        return (t["id"], c["id"]), False


def main():
    import kg_pool
    pool, rows = kg_pool.load()
    relax = "--relax" in sys.argv
    top = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))
    targets = pool["targets"]
    pairs = sorted({(t, c) for r in RANKERS for t in targets for c, _ in top[r][t][:max(KS)]})
    print(f"{len(targets)} targets, {len(pairs)} substitutions", flush=True)
    with ProcessPoolExecutor(5) as ex:
        ok = dict(ex.map(_job, [(rows[t], rows[c], False) for t, c in pairs], chunksize=16))
    if relax:
        rp = sorted({(t, c) for t in targets for c, _ in top["prompt+kg+ft_kont"][t][:RELAX_K]})
        print(f"relaxing {len(rp)} candidates", flush=True)
        with ProcessPoolExecutor(4, initializer=_init, initargs=(1,)) as ex:
            okr = dict(ex.map(_job, [(rows[t], rows[c], True) for t, c in rp], chunksize=4))
    found = {r: {k: {t: float(any(ok[(t, c)] for c, _ in top[r][t][:k])) for t in targets} for k in KS} for r in RANKERS}
    cl = {m["id"]: m["samples"] for m in json.load(open(os.path.join(DATA, "kgcond3_eval_const.json")))}
    found["CrystaLLM fine-tuned"] = {k: {t: float(any(s.get("strict") for s in cl[t][:k])) for t in targets} for k in KS}
    if relax:
        found["K-Ont template, CHGNet-relaxed"] = {k: {t: float(any(okr[(t, c)] for c, _ in top["prompt+kg+ft_kont"][t][:k]))
                                                       for t in targets} for k in KS if k <= RELAX_K}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]
    rep = {"n": len(targets)}
    for name, d in found.items():
        rep[name] = {f"found@{k}": round(100 * sum(v.values()) / len(targets), 1) for k, v in d.items()}

    def diff(a, b, k):
        dd = {t: found[a][k][t] - found[b][k][t] for t in targets}
        bs = sorted(100 * sum(dd[t] for t in b_) / len(b_) for b_ in B)
        return [round(100 * sum(dd.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    for k in KS:
        rep[f"K-Ont template - CrystaLLM @{k}"] = diff("prompt+kg+ft_kont", "CrystaLLM fine-tuned", k)
        rep[f"K-Ont template - composition template @{k}"] = diff("prompt+kg+ft_kont", "composition rule", k)
        rep[f"K-Ont template - no-K-Ont template @{k}"] = diff("prompt+kg+ft_kont", "prompt+kg", k)
        rep[f"K-Ont template - plain-encoder template @{k}"] = diff("prompt+kg+ft_kont", "prompt+kg+ft_base", k)
    for kt in (1, 2, 3, 4):  # matched budget of 8: kt template candidates + (8 - kt) CrystaLLM samples
        for r in ("prompt+kg+ft_kont", "prompt+kg+ft_base", "composition rule"):
            name = f"hybrid {kt} template ({r}) + {8 - kt} CrystaLLM"
            found[name] = {8: {t: float(any(ok[(t, c)] for c, _ in top[r][t][:kt])
                                        or any(s.get("strict") for s in cl[t][:8 - kt])) for t in targets}}
            rep[name] = {"found@8": round(100 * sum(found[name][8].values()) / len(targets), 1),
                         "vs CrystaLLM x8": diff(name, "CrystaLLM fine-tuned", 8)}
    json.dump({"report": rep, "found": found}, open(os.path.join(DATA, "kg_template_gen_relaxed.json" if relax else "kg_template_gen.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
