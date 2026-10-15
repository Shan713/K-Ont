"""Can K-Ont pick a better analogue? Retrieval check before any generation (BUILD_LOG Step 39).

For each target material, the analogue is chosen among KG training-corpus materials with the same anonymous
formula (and a different formula), by:
  composition    today's rule (kg_condition2.similarity: element-by-element chemistry, ties -> most stable)
  kont           highest cosine between K-Ont vectors of the structure-free material sentences
                 (export_embeddings.py `material_cond_tan`: formula, chemistry, flags, properties)
  base           the same with the untrained encoder (control: is it K-Ont's training or just text?)
  comp+kont      composition score first, K-Ont cosine to break ties
  random         a random candidate (expected value over candidates)
  oracle         is ANY candidate right? (ceiling)
Scored: the analogue's space group equals the target's; the analogue's structure fits the target's with
elements anonymised (StructureMatcher.fit_anonymous: same framework). Targets: (a) the 85 KG-test materials
also held out from K-Ont (primary), (b) all other KG-test materials (K-Ont saw them in training).
Paired bootstrap over targets vs composition.

    .venv/Scripts/python.exe abox/kg_retrieval.py --stage base        # plain-encoder vectors
    .venv-crystallm/Scripts/python.exe abox/kg_retrieval.py --stage eval
"""
import argparse
import gzip
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
EXPORT = os.path.join(HERE, "data", "kont_export", "grow5000_v2")
DATA = os.path.join(HERE, "data", "crystallm")


def corpus(split):
    return json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))


def stage_base(a):
    """Plain (untrained) encoder on the same structure-free material sentences K-Ont's vectors came from."""
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    import numpy as np
    import torch
    from ont.hit import HierarchyTransformer
    from phase5_checks import STRUCT_LINES
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    ids = json.load(open(os.path.join(EXPORT, "id_maps.json")))["material"]
    sents = ["\n".join(l for l in vm["material/" + i].split("\n") if not l.startswith(STRUCT_LINES)) for i in ids]
    m = HierarchyTransformer.from_pretrained("Hui97/OnT-MiniLM-L12-galen")
    m.eval()
    with torch.no_grad():
        v = m.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()
        np.save(os.path.join(EXPORT, "material_cond_base_tan.npy"), m.manifold.logmap0(v).cpu().numpy())
    print("base vectors:", len(ids))


def stage_eval(a):
    import numpy as np
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Composition
    import phaseC_generate as gen
    from kg_condition2 import similarity
    ids = json.load(open(os.path.join(EXPORT, "id_maps.json")))["material"]
    row = {i: k for k, i in enumerate(ids)}
    vec = {"kont": np.load(os.path.join(EXPORT, "embeddings.npz"))["material_cond_tan"],
           "base": np.load(os.path.join(EXPORT, "material_cond_base_tan.npy"))}
    for k in vec:
        vec[k] = vec[k] / np.linalg.norm(vec[k], axis=1, keepdims=True)
    train, test = corpus("train"), corpus("test")
    anon = lambda f: Composition(f).anonymized_formula
    by_anon = {}
    for r in train:
        by_anon.setdefault(anon(r["formula"]), []).append(r)
    kont_test = {m.split("/")[-1] for m in json.load(open(os.path.join(HERE, "data", "split_grow_5000.json")))["test_materials"]}
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    struct = {}

    def S(r):
        if r["id"] not in struct:
            try:
                struct[r["id"]] = gen.reference_from_text(r["cif"])
            except Exception:
                struct[r["id"]] = None
        return struct[r["id"]]

    def frame(t, c):
        a_, b_ = S(t), S(c)
        return bool(a_ is not None and b_ is not None and sm.fit_anonymous(a_, b_))

    rng = random.Random(0)
    out = []
    for t in test:
        cands = [c for c in by_anon.get(anon(t["formula"]), []) if c["formula"] != t["formula"]]
        if not cands:
            continue
        sg_ok = [c["space_group"] == t["space_group"] for c in cands]
        fr_ok = [frame(t, c) for c in cands]
        comp = [similarity(t["formula"], c["formula"]) for c in cands]
        pick = {"composition": max(range(len(cands)), key=lambda i: (comp[i], -(cands[i]["e_above_hull"] or 0)))}
        for k in ("kont", "base"):
            cos = [float(vec[k][row[t["id"]]] @ vec[k][row[c["id"]]]) for c in cands]
            pick[k] = int(np.argmax(cos))
            if k == "kont":
                pick["comp+kont"] = max(range(len(cands)), key=lambda i: (comp[i], cos[i]))
        rec = {"id": t["id"], "formula": t["formula"], "kont_unseen": t["id"] in kont_test, "n_candidates": len(cands),
               "random_sg": float(np.mean(sg_ok)), "random_frame": float(np.mean(fr_ok)),
               "oracle_sg": float(any(sg_ok)), "oracle_frame": float(any(fr_ok))}
        for k, i in pick.items():
            rec[f"{k}_sg"], rec[f"{k}_frame"], rec[f"{k}_pick"] = float(sg_ok[i]), float(fr_ok[i]), cands[i]["formula"]
        out.append(rec)
    methods = ["composition", "kont", "base", "comp+kont", "random", "oracle"]
    rep = {}
    for name, sel in (("kont_unseen", [r for r in out if r["kont_unseen"]]), ("kont_seen", [r for r in out if not r["kont_unseen"]]),
                      ("all", out)):
        res = {"n": len(sel), "median_candidates": float(np.median([r["n_candidates"] for r in sel]))}
        for m in methods:
            res[m] = {k: round(100 * float(np.mean([r[f"{m}_{k}"] for r in sel])), 1) for k in ("sg", "frame")}
        boots = [[rng.choice(sel) for _ in sel] for _ in range(10000)]
        for m in ("kont", "base", "comp+kont"):
            for k in ("sg", "frame"):
                d = sorted(100 * np.mean([r[f"{m}_{k}"] - r[f"composition_{k}"] for r in b]) for b in boots)
                res[m][f"{k}_vs_composition"] = [round(float(np.mean([r[f"{m}_{k}"] - r[f"composition_{k}"] for r in sel]) * 100), 1),
                                                 round(d[250], 1), round(d[9749], 1)]
        rep[name] = res
    json.dump({"report": rep, "targets": out}, open(os.path.join(DATA, "kg_retrieval.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "eval"])
    a = ap.parse_args()
    {"base": stage_base, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
