"""Phase C, step 1: space-group predictions for CrystaLLM's held-out test compositions.

The random forest of Phase A (composition features only) is trained on every ground-state formula of
the 5,000 KG EXCEPT those also in the CrystaLLM test subset, then ranks space groups for each test
composition. Output feeds the CrystaLLM prompts (top-1 / top-5 conditions).

    .venv/Scripts/python.exe abox/phaseC_predict_sg.py
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
from phaseA_eval import features, system_of  # noqa: E402


def main():
    known = json.load(open(os.path.join(HERE, "data", "phaseA", "examples.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(HERE, "data", "crystallm", "phaseC_test_features.json"), encoding="utf-8"))
    test_keys = {tuple(sorted(t["element_fractions"].items())) for t in test}
    train = [x for x in known if tuple(sorted(x["element_fractions"].items())) not in test_keys]
    print(f"RF trained on {len(train)} KG formulas ({len(known) - len(train)} removed: also in the test set)")
    vocab = sorted({el for x in known + test for el in x["element_fractions"]})
    sc = StandardScaler().fit(features(train, vocab))
    y = np.array([x["label"]["space_group"] for x in train])
    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(sc.transform(features(train, vocab)), y)
    P = rf.predict_proba(sc.transform(features(test, vocab)))
    order = np.argsort(-P, axis=1)
    for t, row, p in zip(test, order, P):
        t["sg_pred_top5"] = [int(rf.classes_[j]) for j in row[:5]]
        t["sg_pred_prob_top5"] = [round(float(p[j]), 3) for j in row[:5]]
    for name, sub in (("all in-scope", test), ("cathode-like", [t for t in test if t["cathode_like"]])):
        n = len(sub)
        t1 = sum(t["sg"] == t["sg_pred_top5"][0] for t in sub) / n
        t5 = sum(t["sg"] in t["sg_pred_top5"] for t in sub) / n
        cs = sum(system_of(t["sg"]) == system_of(t["sg_pred_top5"][0]) for t in sub) / n
        print(f"{name} (n={n}): crystal system top1 {cs:.3f} | space group top1 {t1:.3f} top5 {t5:.3f}")
    json.dump(test, open(os.path.join(HERE, "data", "crystallm", "phaseC_test_with_pred.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
