"""Does K-Ont still do OnT's job? GALEN benchmark from the OnT paper (BUILD_LOG Step 49).

OnT's authors' data (Zenodo 17314434, CC-BY-4.0, md5-checked): GALEN_OnT (prediction) and GALEN_OnT_inference
(inference). Every K-Ont model starts from Hui97/OnT-MiniLM-L12-galen, i.e. OnT trained on GALEN, then learns
the battery KG. Same protocol for every model, OnT's own evaluator (ont.evaluation.OnTEvaluator, sentence
mode): the centripetal weight lambda is chosen on the task's val.json (nf1, 0.0-1.9 as in fit()), then test.json
is ranked against all 35,501 GALEN concepts; H@1/H@10/H@100/MRR/AUC per axiom kind and combined.
Models: OnT-galen (published reference), K-Ont v2 (Step 36), K-Ont + framework fine-tune + retention (Step 43),
K-Ont v3 native framework relation (Step 46).

    .venv/Scripts/python.exe abox/ont_galen_eval.py
"""
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
import torch  # noqa: E402
from ont.evaluation.evaluator import OnTEvaluator  # noqa: E402
from ont.model import OntologyTransformer  # noqa: E402
import ont.evaluation.evaluator as _ev  # noqa: E402


class _Rows:
    """stand-in for datasets.Dataset.from_list(...)[column]: the installed `datasets` returns a Column object
    there, which OnTEvaluator then hands to the tokenizer (needs list[str]); this returns a plain list"""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, col):
        return [r[col] for r in self.rows]

    @staticmethod
    def from_list(rows):
        return _Rows(rows)


_ev.Dataset = _Rows  # only the evaluator module's reference; OnT's code is unchanged

BENCH = os.path.join(HERE, "data", "ont_bench", "extracted")
OUT = os.path.join(HERE, "data", "ont_bench", "galen_results.json")
MODELS = {"OnT-galen (published)": "Hui97/OnT-MiniLM-L12-galen",
          "K-Ont v2 (Step 36)": os.path.join(HERE, "data", "runs", "full_grow5000_v2", "final"),
          "K-Ont + retention (Step 43)": os.path.join(HERE, "data", "runs", "kont_fw_retain", "final"),
          "K-Ont v3 native (Step 46)": os.path.join(HERE, "data", "runs", "full_grow5000_v3fw", "final"),
          "K-Ont v4 native + hard negatives (Step 51)": os.path.join(HERE, "data", "runs", "full_grow5000_v4hn", "final"),
          "K-Ont + retention incl. GALEN (Step 52)": os.path.join(HERE, "data", "runs", "kont_fw_retain_galen", "final"),
          "K-Ont + balanced retention incl. GALEN (Step 52b)": os.path.join(HERE, "data", "runs", "kont_fw_retain_galen_bal", "final")}
TASKS = {"prediction": "GALEN_OnT", "inference": "GALEN_OnT_inference"}


def fill(q):
    """OnTEvaluator tokenizes nf3/nf4 'role'/'con' even in sentence mode, where they are not used; entries
    without them would crash the tokenizer, so missing fields become empty strings (scores unaffected)"""
    return {k: [{**x, "role": x.get("role") or "", "con": x.get("con") or ""} if k in ("nf3", "nf4") else x for x in v]
            for k, v in q.items()}


def load(task):
    d = os.path.join(BENCH, TASKS[task])
    names = json.load(open(os.path.join(d, "concept_names.json"), encoding="utf-8"))
    names = [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))] if isinstance(names, dict) else names
    return names, json.load(open(os.path.join(d, "val.json"), encoding="utf-8")), json.load(open(os.path.join(d, "test.json"), encoding="utf-8"))


def main():
    logging.basicConfig(level=logging.WARNING)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for task in TASKS:
        names, val, test = load(task)
        print(task, {k: len(v) for k, v in test["query_sentences"].items()}, f"{len(names)} concepts", flush=True)
        for mname, path in MODELS.items():
            key = f"{task} | {mname}"
            if key in res:
                continue
            m = OntologyTransformer.from_pretrained(path)
            m.hit_model.eval()
            ve = OnTEvaluator(m, fill(val["query_sentences"]), val["answer_ids"], names, batch_size=64)
            with torch.no_grad():
                ve(model=m.hit_model, epoch="validation")
                lam = float(ve.best_centri_weight)
                te = OnTEvaluator(m, fill(test["query_sentences"]), test["answer_ids"], names, batch_size=64)
                te(model=m.hit_model, best_centri_weight=lam, inference_mode="sentence")
            rows = {k: {c: (float(v) if not isinstance(v, str) else v) for c, v in r.items()} for k, r in te.results.iterrows()}
            res[key] = {"lambda": lam, "results": rows}
            json.dump(res, open(OUT, "w"), indent=1)
            print(key, "lambda", lam, {k: {c: round(r[c], 4) for c in ("H@1", "H@10", "MRR", "AUC")} for k, r in rows.items()}, flush=True)
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
