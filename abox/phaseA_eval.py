"""Phase A, step 2: predict a novel composition's ground-state crystal system / space group.

Inputs: data/phaseA/examples.json (abox/phaseA_features.py; one ground state per formula). Test
formulas are 20% of those whose formula is in NO K-Ont training row (data/split_grow_1000.json train
side), so an embedding can't score by remembering a material it was trained on. Everything else is
the "known" set the methods may learn from or look up.

Methods (each ranks space groups; crystal system = system of the top space group, except where a
method predicts it directly):
  majority            most common label in the known set
  stoichiometry       most common label among known formulas with the same anonymous formula
                      (ABC2 ...) -- the prototype prior; falls back to majority
  knn_composition     5 nearest known formulas (element fractions + element-property statistics)
  random_forest       on the same composition features
  <model> knn / lr    OnT embedding of a composition-only sentence (formula, chemical system --
                      what a novel material has), probed by nearest neighbours (hyperbolic
                      distance) and by logistic regression

    .venv/Scripts/python.exe abox/phaseA_eval.py --out data/phaseA/results.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261006
RUNS = {  # label -> model path (None = untrained base)
    "untrained OnT (galen)": None,
    "OnT full (1k)": "data/runs/full_grow1000_inbatch/final",
    "OnT no_geometry (1k)": "data/runs/no_geometry_grow1000_inbatch/final",
    "OnT stripped sentences (1k)": "data/runs/full_grow1000_nostruct/final",
}
SG_SYSTEM = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
             (167, "trigonal"), (194, "hexagonal"), (230, "cubic")]


def system_of(sg: int) -> str:
    return next(s for hi, s in SG_SYSTEM if sg <= hi)


def sentence(ex):
    elements = sorted(ex["element_fractions"])
    return (f"Material: {ex['formula']}\nType: substance\nFormula: {ex['formula']}\n"
            f"Chemical system: {'-'.join(elements)}")


def features(examples, vocab):
    rows = []
    for ex in examples:
        fr = [ex["element_fractions"].get(el, 0.0) for el in vocab]
        rows.append(fr + [ex["stats"][k] for k in sorted(ex["stats"])])
    return np.nan_to_num(np.array(rows, dtype=float))


def score(ranked, truth):
    """ranked: list of ranked space-group lists; truth: list of ints."""
    n = len(truth)
    top = lambda k: sum(t in r[:k] for r, t in zip(ranked, truth)) / n
    cs = sum(system_of(r[0]) == system_of(t) for r, t in zip(ranked, truth)) / n
    return {"n": n, "crystal_system_top1": round(cs, 3), "space_group_top1": round(top(1), 3),
            "space_group_top3": round(top(3), 3), "space_group_top5": round(top(5), 3)}


def rank_from_proba(proba, classes):
    order = np.argsort(-proba, axis=1)
    return [[int(classes[j]) for j in row] for row in order]


def counter_rank(counter, fallback):
    ranked = [sg for sg, _ in counter.most_common()]
    return ranked + [sg for sg in fallback if sg not in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.join(HERE, "data", "phaseA", "examples.json"))
    ap.add_argument("--split", default=os.path.join(HERE, "data", "split_grow_1000.json"))
    ap.add_argument("--entities1k", default=os.path.join(HERE, "data", "battgpt_abox_grow_1000", "entities.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "phaseA", "results.json"))
    ap.add_argument("--no-embeddings", action="store_true")
    a = ap.parse_args()

    ex = json.load(open(a.examples, encoding="utf-8"))
    e1k = json.load(open(a.entities1k, encoding="utf-8"))
    trained_formulas = {e1k["materials"][m]["formula"] for m in json.load(open(a.split))["train_materials"]}
    eligible = sorted(i for i, x in enumerate(ex) if x["formula"] not in trained_formulas)
    rng = random.Random(SEED)
    test_idx = sorted(rng.sample(eligible, round(0.2 * len(eligible))))
    test_set = set(test_idx)
    train_idx = [i for i in range(len(ex)) if i not in test_set]
    tr, te = [ex[i] for i in train_idx], [ex[i] for i in test_idx]
    y_tr = np.array([x["label"]["space_group"] for x in tr])
    truth = [x["label"]["space_group"] for x in te]
    cathode = [j for j, x in enumerate(te) if x["battery_role"] == "PositiveElectrode"]
    print(f"known {len(tr)} / test {len(te)} formulas ({len(cathode)} cathodes); "
          f"{len(set(y_tr))} space groups in the known set", flush=True)

    ranks = {}
    majority = [sg for sg, _ in Counter(y_tr.tolist()).most_common()]
    ranks["majority"] = [majority] * len(te)

    by_anon = defaultdict(Counter)
    for x in tr:
        by_anon[x["anonymous_formula"]][x["label"]["space_group"]] += 1
    ranks["stoichiometry prototype"] = [counter_rank(by_anon.get(x["anonymous_formula"], Counter()), majority) for x in te]

    vocab = sorted({el for x in ex for el in x["element_fractions"]})
    sc = StandardScaler().fit(features(tr, vocab))
    Xtr, Xte = sc.transform(features(tr, vocab)), sc.transform(features(te, vocab))

    def knn_rank(Dte_tr, k=5):
        out = []
        for row in Dte_tr:
            nn = np.argsort(row)[:k]
            c = Counter()
            for r, j in enumerate(nn):
                c[int(y_tr[j])] += 1.0 / (r + 1)
            out.append(counter_rank(c, majority))
        return out

    D = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    ranks["nearest compositions (k=5)"] = knn_rank(D)
    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1, class_weight=None).fit(Xtr, y_tr)
    ranks["random forest"] = rank_from_proba(rf.predict_proba(Xte), rf.classes_)

    if not a.no_embeddings:
        sys.path.insert(0, os.path.join(HERE, "OnT"))
        import torch
        from ont.hit import HierarchyTransformer
        from ont.model import OntologyTransformer
        sent_tr, sent_te = [sentence(x) for x in tr], [sentence(x) for x in te]
        for label, path in RUNS.items():
            if path is None:
                model = HierarchyTransformer.from_pretrained("Hui97/OnT-MiniLM-L12-galen"); man = model.manifold
            else:
                if not os.path.isdir(os.path.join(HERE, path)):
                    print(f"skip {label}: {path} missing"); continue
                model = OntologyTransformer.from_pretrained(os.path.join(HERE, path)); man = model.manifold
            enc = lambda s: model.encode(s, batch_size=128, convert_to_tensor=True, show_progress_bar=False).float()
            with torch.no_grad():
                Etr, Ete = enc(sent_tr), enc(sent_te)
                Dh = torch.cat([man.dist(Ete[i:i + 64].unsqueeze(1), Etr.unsqueeze(0)) for i in range(0, len(Ete), 64)]).cpu().numpy()
            ranks[f"{label}: nearest (k=5)"] = knn_rank(Dh)
            Ztr, Zte = Etr.cpu().numpy(), Ete.cpu().numpy()
            zs = StandardScaler().fit(Ztr)
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(zs.transform(Ztr), y_tr)
            ranks[f"{label}: linear probe"] = rank_from_proba(lr.predict_proba(zs.transform(Zte)), lr.classes_)
            # does the embedding add anything to the composition features?
            rf2 = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(
                np.hstack([Xtr, zs.transform(Ztr)]), y_tr)
            ranks[f"random forest + {label}"] = rank_from_proba(
                rf2.predict_proba(np.hstack([Xte, zs.transform(Zte)])), rf2.classes_)
            print(f"done {label}", flush=True)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {"n_known": len(tr), "n_test": len(te), "n_test_cathodes": len(cathode),
           "chance_space_group_top1_if_uniform": round(1 / len(set(y_tr)), 4), "results": {}}
    for name, r in ranks.items():
        out["results"][name] = {"all": score(r, truth),
                                "cathodes": score([r[j] for j in cathode], [truth[j] for j in cathode])}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    w = max(map(len, ranks))
    print(f"\n{'method':{w}}  | all: CS@1  SG@1  SG@3  SG@5 | cathodes: CS@1  SG@1  SG@3  SG@5")
    for name, v in out["results"].items():
        A, Cc = v["all"], v["cathodes"]
        print(f"{name:{w}}  |      {A['crystal_system_top1']:.3f} {A['space_group_top1']:.3f} {A['space_group_top3']:.3f} {A['space_group_top5']:.3f}"
              f" |           {Cc['crystal_system_top1']:.3f} {Cc['space_group_top1']:.3f} {Cc['space_group_top3']:.3f} {Cc['space_group_top5']:.3f}")


if __name__ == "__main__":
    main()
