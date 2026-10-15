"""Heart-disease K-Ont check, part 2: the two follow-up tests and the data for the report's visualisations.

  fewlabels   classifiers see 10 / 20 / 50 / 100% of the training fold's LABELS (3 random subsets each); the
              embeddings (OnT, K-Ont, untrained encoder) and the scalers / PCA use the whole training fold, which
              is legitimate because none of them ever sees a diagnosis. Same 10 folds as kont_heart.py.
  missing     columns hidden from the TABLE but still present in the KG: (a) chest pain + thallium scan,
              (b) + number of major vessels. Does an embedding carry the hidden information back?
  visuals     fold-0 K-Ont model (retrained, kept): patient 2D maps, k-NN label agreement, depth in the ball,
              typing of patients, relation check (does rotating a patient by "has chest pain" land on its chest
              pain type?), and a worked example of real patients.

    python heart/report_data.py --stage fold0
    python heart/report_data.py --stage tests
    python heart/report_data.py --stage visuals
Output: data/heart/report/*.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "heart"))
sys.path.insert(0, os.path.join(HERE, "OnT"))
import kont_heart as kh  # noqa: E402

OUT = os.path.join(kh.OUT, "report")
KEEP = os.path.join(kh.OUT, "runs", "kont_fold0_keep")


def stage_fold0(a):
    import numpy as np
    P = json.load(open(os.path.join(kh.OUT, "patients.json")))
    y = np.array(kh.labels())
    tr, te = kh.folds(y)[0]
    if not os.path.exists(os.path.join(KEEP, "final")):
        kh.kont_data([P["ids"][i] for i in tr], P, os.path.join(kh.OUT, "folds", "fold0"))
        kh.fit(os.path.join(kh.OUT, "folds", "fold0"), KEEP, 3, 32, True)
    print("fold-0 K-Ont kept at", KEEP)


def feature_sets(k, tr, num, cat, hide_cols, emb):
    """raw (minus hidden one-hot groups), and raw + each embedding (PCA 16, fitted on the training fold)."""
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(kh.OUT, "heart_dataset.csv"), sep=";", encoding="utf-8-sig")
    cats = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"] if c not in hide_cols]
    cat = pd.get_dummies(df[cats].astype(str)).to_numpy(float)
    sc = StandardScaler().fit(num[tr])
    raw = np.hstack([sc.transform(num), cat])
    out = {"raw": raw}
    for name, E in emb.items():
        p = PCA(n_components=16, random_state=0).fit(E[tr])
        z = p.transform(E)
        z = StandardScaler().fit(z[tr]).transform(z)
        out[f"raw+{name}"] = np.hstack([raw, z])
    return out


def run_models(X, y, tr_lab, te, mlp_seeds=(0,)):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.svm import SVC
    models = {"LogReg": lambda: LogisticRegression(max_iter=5000),
              "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
              "SVM": lambda: SVC(probability=True, random_state=0),
              "GradBoost": lambda: HistGradientBoostingClassifier(random_state=0),
              "MLP": None}
    res = {}
    for m, mk in models.items():
        for s, M in X.items():
            if mk is None:
                prob = kh.mlp_probs(M[tr_lab], y[tr_lab], M[te], seeds=mlp_seeds)
            else:
                prob = mk().fit(M[tr_lab], y[tr_lab]).predict_proba(M[te])[:, 1]
            res[(m, s)] = {"acc": accuracy_score(y[te], (prob >= 0.5).astype(int)), "auc": roc_auc_score(y[te], prob)}
    return res


def stage_tests(a):
    import numpy as np
    from scipy.stats import wilcoxon
    y = np.array(kh.labels())
    num, _ = kh.raw_matrix()
    emb_dir = os.path.join(kh.OUT, "emb")
    base, ont = np.load(os.path.join(emb_dir, "base.npy")), np.load(os.path.join(emb_dir, "ont_concepts.npy"))
    rows = []  # (test, setting, fold, rep, model, set, acc, auc)
    for k, (tr, te) in enumerate(kh.folds(y)):
        emb = {"ont": ont, "kont": np.load(os.path.join(emb_dir, f"kont_fold{k}.npy")), "base": base}
        X = feature_sets(k, tr, num, None, [], emb)
        for frac in (0.1, 0.2, 0.5, 1.0):
            for rep in range(3 if frac < 1 else 1):
                rng = np.random.RandomState(1000 * k + rep)
                if frac < 1:  # stratified label subset
                    lab = np.concatenate([rng.choice(tr[y[tr] == c], max(2, int(round(frac * (y[tr] == c).sum()))),
                                                     replace=False) for c in (0, 1)])
                else:
                    lab = tr
                for (m, s), r in run_models(X, y, lab, te).items():
                    rows.append(["fewlabels", frac, k, rep, m, s, r["acc"], r["auc"]])
        for name, hide in (("hide cp+thal", ["cp", "thal"]), ("hide cp+thal+ca", ["cp", "thal", "ca"])):
            Xh = feature_sets(k, tr, num, None, hide, emb)
            for (m, s), r in run_models(Xh, y, tr, te).items():
                rows.append(["missing", name, k, 0, m, s, r["acc"], r["auc"]])
        print(f"fold {k} done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "tests_rows.json"), "w"))
    # summary: mean AUC/acc per test x setting x model x set, and paired raw+kont vs others
    summ = {}
    import itertools
    keyf = lambda r: (r[0], str(r[1]), r[4], r[5])
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    for (t, st, m, s), g in groups.items():
        summ.setdefault(t, {}).setdefault(st, {}).setdefault(m, {})[s] = {
            "auc": round(float(np.mean([x[7] for x in g])), 4), "acc": round(float(np.mean([x[6] for x in g])), 4)}
    paired = {}
    for t, st, m in {(k[0], k[1], k[2]) for k in groups}:
        base_rows = {(x[2], x[3]): x for x in groups[(t, st, m, "raw+kont")]}
        for ref in ("raw", "raw+ont", "raw+base"):
            d = np.array([base_rows[(x[2], x[3])][7] - x[7] for x in groups[(t, st, m, ref)]])
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            paired.setdefault(t, {}).setdefault(st, {}).setdefault(m, {})[f"raw+kont - {ref}"] = {
                "auc_diff": round(float(d.mean()), 4), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": round(p, 4)}
        d = np.array([x[7] - y_[7] for x, y_ in zip(groups[(t, st, m, "raw+ont")], groups[(t, st, m, "raw")])])
        paired[t][st][m]["raw+ont - raw"] = {"auc_diff": round(float(d.mean()), 4), "wins": int((d > 0).sum()),
                                              "losses": int((d < 0).sum()),
                                              "p": round(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, 4)}
    json.dump({"mean": summ, "paired": paired}, open(os.path.join(OUT, "tests_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)[:3000])


def stage_visuals(a):
    import numpy as np
    import torch
    from ont.hit import HierarchyTransformer
    from ont.model import OntologyTransformer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    P = json.load(open(os.path.join(kh.OUT, "patients.json")))
    y = np.array(kh.labels())
    tr, te = kh.folds(y)[0]
    ids = P["ids"]
    sents = [P["sentences"][p] for p in ids]
    concepts = [v for _, v in sorted(json.load(open(os.path.join(kh.TBOX, "concept_names.json"))).items(), key=lambda kv: int(kv[0]))]
    models = {"OnT": OntologyTransformer.from_pretrained(os.path.join(kh.OUT, "runs", "ont", "final")),
              "K-Ont": OntologyTransformer.from_pretrained(os.path.join(KEEP, "final"))}
    out = {"fold0_test_ids": [ids[i] for i in te], "labels": y.tolist(), "ids": ids}

    def enc(m, s):
        with torch.no_grad():
            return m.encode(s, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()

    reps, per_model = {}, {}
    for name, m in models.items():
        m.hit_model.eval()
        man, lam = m.manifold, (m.best_lambda or 0.0)
        C = enc(m, concepts)
        Pt = enc(m, sents)  # the patient sentence through this model
        score = lambda child, par: -(man.dist(child.unsqueeze(1), par.unsqueeze(0))
                                     + lam * (man.dist0(par).unsqueeze(0) - man.dist0(child).unsqueeze(1)))
        typ = score(Pt, C).argmax(1).cpu().numpy()
        top = {}
        for t in typ:
            top[concepts[t]] = top.get(concepts[t], 0) + 1
        # relation check on the fold-0 TEST patients: rotate the value concepts by the role; nearest to the patient?
        rel = {}
        for prop, role in kh.ROLE.items():
            vals = sorted({kh.concept_name(P["patients"][p]["obj"][prop]) for p in ids if prop in P["patients"][p]["obj"]})
            R = torch.tensor(m.encode_existence([role] * len(vals), vals), device=Pt.device)
            sc = score(Pt[te], R)
            truth = [vals.index(kh.concept_name(P["patients"][ids[i]]["obj"][prop])) for i in te]
            rel[role] = round(float((sc.argmax(1).cpu().numpy() == np.array(truth)).mean()), 4)
            rel[role + " (chance)"] = round(1 / len(vals), 4)
        depth = {c: round(float(man.dist0(C[i]).item()), 3) for i, c in enumerate(concepts)}
        per_model[name] = {"patient_typed_as": dict(sorted(top.items(), key=lambda kv: -kv[1])),
                           "patient_typed_patient_test": round(float((typ[te] == concepts.index("patient")).mean()), 4),
                           "relation_accuracy_test": rel, "concept_depth": depth,
                           "patient_depth_median": round(float(man.dist0(Pt).median().item()), 3)}
        reps[name + " (patient sentence)"] = man.logmap0(Pt).cpu().numpy()
    # the representations the classifiers used
    emb = os.path.join(kh.OUT, "emb")
    reps["OnT (5 concepts)"] = np.load(os.path.join(emb, "ont_concepts.npy"))
    reps["K-Ont"] = np.load(os.path.join(emb, "kont_fold0.npy"))
    reps["Untrained encoder"] = np.load(os.path.join(emb, "base.npy"))
    out["models"] = per_model

    def knn_agree(E, k=10):
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        S = En @ En.T
        np.fill_diagonal(S, -9)
        nn = np.argsort(-S, 1)[:, :k]
        return float((y[nn] == y[:, None]).mean()), nn
    maps, knn = {}, {}
    for name in ("OnT (5 concepts)", "K-Ont", "Untrained encoder"):
        E = reps[name]
        agree, nn = knn_agree(E)
        knn[name] = round(agree, 4)
        xy = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(PCA(30, random_state=0).fit_transform(E))
        pc = PCA(2, random_state=0).fit_transform(E)
        maps[name] = {"tsne": np.round(xy, 3).tolist(), "pca": np.round(pc, 3).tolist(), "nn": nn[:, :5].tolist(),
                      "distinct_points": int(len({tuple(np.round(r, 5)) for r in E}))}
    out["maps"], out["knn_label_agreement_k10"], out["knn_chance"] = maps, knn, round(float(max(y.mean(), 1 - y.mean())), 4)
    # worked example: patient_001 (no disease) and patient_002 (disease)
    ex = {}
    for p in ("patient_001", "patient_002"):
        i = ids.index(p)
        ex[p] = {"sentence": P["sentences"][p], "label": int(y[i]), "in_fold0_test": bool(i in set(te.tolist())),
                 "kont_vector_first24": np.round(reps["K-Ont"][i][:24], 3).tolist(),
                 "kont_vector_norm": round(float(np.linalg.norm(reps["K-Ont"][i])), 3),
                 "ont_concepts": [kh.concept_name(P["patients"][p]["obj"][prop]) for prop in kh.ROLE if prop in P["patients"][p]["obj"]],
                 "ont_vector_first24": np.round(reps["OnT (5 concepts)"][i][:24], 3).tolist(),
                 "neighbours": {name: [{"id": ids[j], "label": int(y[j]),
                                        "summary": P["sentences"][ids[j]].split("\n")[1:6]} for j in maps[name]["nn"][i]]
                                for name in ("OnT (5 concepts)", "K-Ont")}}
    out["examples"] = ex
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "visuals.json"), "w"), indent=1)
    print(json.dumps({"models": per_model, "knn": knn, "distinct": {k: v["distinct_points"] for k, v in maps.items()}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fold0", "tests", "visuals"])
    a = ap.parse_args()
    {"fold0": stage_fold0, "tests": stage_tests, "visuals": stage_visuals}[a.stage](a)


if __name__ == "__main__":
    main()
