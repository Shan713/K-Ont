"""Does the ONTOLOGY in K-Ont matter? K-Ont retrained without its TBox axioms (BUILD_LOG Step 54).

K-Ont v2 (Step 36) = OnT's TBox axioms (class hierarchy + existential restrictions, oversampled) + the KG's ABox
rows (individuals typed under ontology classes, relation rows). Ablation "no-TBox": the same data minus every
TBox row -- type rows whose child is a class name (10,962) and exist rows whose concept is a class expression
(303, no line break) -- so the KG facts and the typing of individuals stay, the ontology's axioms go.
Trained exactly as Step 36 (train_ont.py full, galen, in-batch negatives, batch 32, 3 epochs), then put through
the same evaluations as K-Ont: phase5_checks, GALEN (ont_galen_eval.py), and the Step 52c one-model recipe
(kont_framework.py variant fw_anchor_notbox, from this model) for crystal retrieval.

    .venv/Scripts/python.exe abox/kont_ablation.py
    .venv/Scripts/python.exe train_ont.py --variant full --epochs 3 --batch-size 32 --in-batch-negs \
        --data-prefix data/battgpt_merged_grow_5000_notbox --output data/runs/full_grow5000_notbox
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "battgpt_merged_grow_5000_v2_full")
DST = os.path.join(HERE, "data", "battgpt_merged_grow_5000_notbox_full")


def main():
    shutil.copytree(SRC, DST, dirs_exist_ok=True)
    classes = set(json.load(open(os.path.join(SRC, "concept_names.json"))).values())
    tr = [json.loads(l) for l in open(os.path.join(SRC, "train.jsonl"), encoding="utf-8") if l.strip()]
    ex = [json.loads(l) for l in open(os.path.join(SRC, "train_exist.jsonl"), encoding="utf-8") if l.strip()]
    keep_tr = [r for r in tr if r["child"] not in classes]
    keep_ex = [r for r in ex if "\n" in r["Concept"]]  # ABox rows: multi-line individual sentences
    for name, rows in (("train.jsonl", keep_tr), ("train_exist.jsonl", keep_ex)):
        with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    s = json.load(open(os.path.join(SRC, "merge_summary.json")))
    s.update({"step54_ablation": "no TBox axioms", "removed_tbox_type_rows": len(tr) - len(keep_tr),
              "removed_tbox_exist_rows": len(ex) - len(keep_ex), "n_merged_type": len(keep_tr), "n_merged_exist": len(keep_ex)})
    json.dump(s, open(os.path.join(DST, "merge_summary.json"), "w"), indent=1)
    print({k: s[k] for k in ("removed_tbox_type_rows", "removed_tbox_exist_rows", "n_merged_type", "n_merged_exist")})


if __name__ == "__main__":
    main()
