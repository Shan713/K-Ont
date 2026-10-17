"""Novel cathode compositions: K-Ont-guided templates vs CrystaLLM, by CHGNet energy (BUILD_LOG Step 44).

The 200 novel compositions of Step 28 (data/crystallm/novel_candidates.json: substitution children of KG
cathodes, NOT in Materials Project). No reference structure exists, so the comparison is energetic, per
composition: the lowest CHGNet energy (eV/atom, after relaxation) a method reaches -- the same relaxation
(fmax 0.1, 300 steps, cell free) and energy (CHGNet.predict_structure) as abox/novel_chgnet.py, which
gave CrystaLLM-large's best of up to 10 relaxed samples (5 composition-prompt + 5 parent-space-group
prompt) per composition (data/crystallm/novel_chgnet_*.json).
Template methods (kg_template_gen substitution into the framework, then relaxation):
  kont3    top 3 frameworks by the composition-only K-Ont retriever (kg_retriever_ft.py --minimal)
  base3    top 3 by the composition-only plain-encoder retriever (the K-Ont control)
  parent   the parent's own framework (how the composition was proposed: a strong, hand-made baseline)
Per composition: delta = method best - CrystaLLM best; win / tie / loss with a 5 meV/atom tie band (as
Step 28); mean delta with a bootstrap CI; and union (CrystaLLM + kont3) vs CrystaLLM alone.

    .venv-crystallm/Scripts/python.exe abox/novel_template.py --stage pool
    .venv/Scripts/python.exe abox/kg_retriever_ft.py --init kont --minimal --tag _min --extra data/crystallm/novel_pool.json
    .venv/Scripts/python.exe abox/kg_retriever_ft.py --init base --minimal --tag _min --extra data/crystallm/novel_pool.json
    .venv-crystallm/Scripts/python.exe abox/novel_template.py --stage relax
"""
import argparse
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
POOL = os.path.join(DATA, "novel_pool.json")
OUT = os.path.join(DATA, "novel_template.json")
TIE = 0.005


def stage_pool(a):
    from pymatgen.core import Composition
    import kg_pool
    pool, rows = kg_pool.load()
    seen = set(pool["seen_train"]) | set(pool["seen_val"])
    by_anon = {}
    for i in seen:
        by_anon.setdefault(Composition(rows[i]["formula"]).anonymized_formula, []).append(i)
    out = {}
    for n in json.load(open(os.path.join(DATA, "novel_candidates.json"))):
        comp, cell = Composition(n["formula"]), Composition(n["cell"])
        cands = [c for c in by_anon.get(comp.anonymized_formula, []) if rows[c]["formula"] != comp.reduced_formula]
        sites = ", ".join(f"{e.symbol}:{int(v)}" for e, v in cell.items())
        text = "\n".join([f"Material: {n['formula']} ({n['id']})", "Type: substance", f"Formula: {n['formula']}",
                          "Chemical system: " + "-".join(sorted(e.symbol for e in comp.elements)),
                          f"Sites: {int(cell.num_atoms)} ({sites})"])
        out[n["id"]] = {"text": text, "candidates": cands, "formula": n["formula"], "cell": n["cell"],
                        "parent": n["parent_id"].split("/")[-1]}
    json.dump(out, open(POOL, "w"), indent=1)
    print(f"{len(out)} novel compositions, {sum(bool(v['candidates']) for v in out.values())} with pool candidates; "
          f"parent in KG corpus: {sum(v['parent'] in rows for v in out.values())}")


def _init(threads):
    global _M, _OPT
    import torch
    torch.set_num_threads(threads)
    from chgnet.model import CHGNet, StructOptimizer
    _M = CHGNet.load(verbose=False)
    _OPT = StructOptimizer(model=_M, use_device=os.environ.get("CHGNET_DEVICE", "cpu"))  # GPU when it is free


def _relax(args):
    key, target_formula, c_formula, c_cif = args
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.core import Composition, Element
    try:
        s = gen.reference_from_text(c_cif)
        old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.replace_species({x.symbol: y.symbol for x, y in zip(k2._ordered(c_formula), k2._ordered(target_formula))})
        if s.composition.reduced_formula != Composition(target_formula).reduced_formula:
            return key, None
        new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.scale_lattice(s.volume * new / old)
        r = _OPT.relax(s, fmax=0.1, steps=300, verbose=False)
        return key, round(float(_M.predict_structure(r["final_structure"])["e"]), 5)
    except Exception:
        return key, None


def stage_relax(a):
    import kg_pool
    _, rows = kg_pool.load()
    novel = json.load(open(POOL))
    sc = {k: json.load(open(os.path.join(DATA, f"kg_ft_scores_{k}_min.json"))) for k in ("kont", "base")}
    plan = {}
    for nid, v in novel.items():
        m = {}
        for k in ("kont", "base"):
            s = sc[k].get(nid, {})
            m[k + "3"] = sorted(s, key=s.get, reverse=True)[:3]
        m["parent"] = [v["parent"]] if v["parent"] in rows else []
        plan[nid] = m
    cache = json.load(open(OUT))["energies"] if os.path.exists(OUT) else {}
    jobs = sorted({(nid, c) for nid, m in plan.items() for cs in m.values() for c in cs} - {tuple(k.split("|")) for k in cache})
    print(f"{len(jobs)} relaxations", flush=True)
    from robust_pool import robust_map  # resumable: a dead worker no longer loses the run

    def save(r):
        cache.update({f"{nid}|{c}": e for (nid, c), e in r.items()})
        json.dump({"energies": cache}, open(OUT, "w"))
    save(robust_map(_relax, [((nid, c), novel[nid]["formula"], rows[c]["formula"], rows[c]["cif"]) for nid, c in jobs],
                    a.workers, save, initializer=_init, initargs=(1,), log=lambda m: print(m, flush=True)))
    crystallm = {}
    for f in ("000", "050", "100", "150"):
        for mrec in json.load(open(os.path.join(DATA, f"novel_chgnet_{f}.json")))["materials"]:
            if "best" in mrec:
                crystallm[mrec["id"]] = mrec["best"]["e"]
    best = {meth: {nid: min((cache[f"{nid}|{c}"] for c in m[meth] if cache.get(f"{nid}|{c}") is not None), default=None)
                   for nid, m in plan.items()} for meth in ("kont3", "base3", "parent")}
    best["kont3 + CrystaLLM"] = {nid: min([x for x in (best["kont3"][nid], crystallm.get(nid)) if x is not None], default=None)
                                 for nid in plan}
    rng = random.Random(0)

    def compare(a_, b_):
        ids = [i for i in plan if a_.get(i) is not None and b_.get(i) is not None]
        d = [a_[i] - b_[i] for i in ids]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
        return {"n": len(ids), "win": sum(x < -TIE for x in d), "tie": sum(abs(x) <= TIE for x in d), "loss": sum(x > TIE for x in d),
                "mean_delta_meV": round(1000 * sum(d) / len(d), 1), "ci95_meV": [round(1000 * boots[125], 1), round(1000 * boots[4874], 1)],
                "median_delta_meV": round(1000 * sorted(d)[len(d) // 2], 1)}
    rep = {"n_novel": len(plan), "crystallm_with_structure": len(crystallm),
           "with_template": {m: sum(v is not None for v in best[m].values()) for m in best},
           "vs CrystaLLM (method - CrystaLLM, negative = lower energy)": {m: compare(best[m], crystallm) for m in best},
           "kont3 - base3": compare(best["kont3"], best["base3"]), "kont3 - parent": compare(best["kont3"], best["parent"])}
    json.dump({"report": rep, "plan": plan, "energies": cache, "best": best, "crystallm": crystallm}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pool", "relax"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"pool": stage_pool, "relax": stage_relax}[a.stage](a)


if __name__ == "__main__":
    main()
