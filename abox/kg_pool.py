"""Step 40 split and analogue pool, one source of truth for retrieval, generator training and evaluation.

  seen     KG-corpus materials K-Ont trained on (all three KG-corpus splits): the generator's training data
           (KG val split -> its checkpoint selection) and the ANALOGUE POOL
  targets  KG-corpus materials K-Ont never saw (split_grow_5000 test materials): never in any training --
           not K-Ont, not the retriever, not the generator. Evaluation only.
Candidates of a material = pool materials with the same anonymous formula and a different formula.

    .venv-crystallm/Scripts/python.exe abox/kg_pool.py   ->  data/crystallm/kg_pool.json
"""
import gzip
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
POOL = os.path.join(DATA, "kg_pool.json")


def corpus(split):
    return json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))


def load():
    """(pool dict, {id: corpus row})"""
    rows = {r["id"]: r for s in ("train", "val", "test") for r in corpus(s)}
    return json.load(open(POOL)), rows


def main():
    from pymatgen.core import Composition
    unseen = {m.split("/")[-1] for m in json.load(open(os.path.join(HERE, "data", "split_grow_5000.json")))["test_materials"]}
    rows = [r for s in ("train", "val", "test") for r in corpus(s)]
    anon = {r["id"]: Composition(r["formula"]).anonymized_formula for r in rows}
    seen = [r for r in rows if r["id"] not in unseen]
    by_anon = {}
    for r in seen:
        by_anon.setdefault(anon[r["id"]], []).append(r)
    cands = {r["id"]: [c["id"] for c in by_anon.get(anon[r["id"]], []) if c["formula"] != r["formula"]] for r in rows}
    out = {"seen_train": [r["id"] for r in seen if r["split"] != "val"],
           "seen_val": [r["id"] for r in seen if r["split"] == "val"],
           "targets": [r["id"] for r in rows if r["id"] in unseen and cands[r["id"]]],
           "targets_no_candidate": [r["id"] for r in rows if r["id"] in unseen and not cands[r["id"]]],
           "candidates": cands}
    json.dump(out, open(POOL, "w"))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
