"""Does K-Ont help a downstream predictor? Heart-disease check (UCI Cleveland, 303 patients), OnT vs K-Ont.

Task: predict heart disease (label 0/1) from the 13 clinical features, with and without ontology
embeddings, using classic ML and a small neural network. Fixed before any result:

Feature sets (per patient)
  raw            the 13 CSV features: categoricals one-hot, numerics standardised
  raw+ont        + standard OnT (TBox only, OnT's own DeepOnto data, as in the paper). OnT cannot embed a
                 patient, so a patient = its 5 categorical values as ontology concepts (gender, chest pain,
                 resting ECG, ST slope, thallium scan), each encoded by OnT, concatenated
  raw+kont       + K-Ont (TBox + the fold's TRAINING patients as KG individuals: one sentence per patient, typed
                 under "patient", linked by the 5 relations to their value concepts). A patient = K-Ont's
                 encoding of its sentence (test patients encoded, never trained on)
  raw+base       + the untrained encoder (OnT-galen start point) on the same patient sentence: does K-Ont's
                 TRAINING matter, or just encoding the text?
  ont, kont, base  the embeddings alone (how much each representation carries by itself)
Embeddings -> tangent space (logmap0) -> PCA to 16 dims fitted on the training fold.

Leakage: hasDiagnosis is removed from the TBox and never written into any sentence or row; K-Ont is retrained
inside every fold on that fold's training patients only.

Models: logistic regression, random forest, RBF-SVM, histogram gradient boosting (ML); an MLP, 2 x 64, dropout
0.2, early stopping on an inner validation split, 3 seeds averaged (DL).
Protocol: stratified 5-fold CV repeated twice (10 folds), identical folds for every feature set and model.
Metrics: accuracy, ROC-AUC, F1. Paired across folds: raw+kont vs raw+ont, vs raw, vs raw+base (mean difference,
Wilcoxon signed-rank p).

    python heart/kont_heart.py --stage prep
    python heart/kont_heart.py --stage ont         # standard OnT, once
    python heart/kont_heart.py --stage folds       # K-Ont per fold + all embeddings
    python heart/kont_heart.py --stage eval
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data", "heart")
TBOX = os.path.join(OUT, "ont_tbox")
ROLE = {"hasGender": "has gender", "hasChestPain": "has chest pain",
        "hasRestingElectrocardiographicResult": "has resting electrocardiographic result",
        "hasSlopePeakExerciseSTSegment": "has slope peak exercise ST segment",
        "hasThalliumScanResult": "has thallium scan result"}
LABEL = {"hasGender": "Gender", "hasChestPain": "Chest pain", "hasRestingElectrocardiographicResult": "Resting ECG",
         "hasSlopePeakExerciseSTSegment": "ST slope", "hasThalliumScanResult": "Thallium scan"}
DATA_LINES = [("hasAge", "Age", ""), ("hasBloodPressure", "Resting blood pressure", " mm Hg"),
              ("hasSerumCholestoral", "Serum cholesterol", " mg/dl"),
              ("hasFastingBloodSugarGreaterThan120", "Fasting blood sugar over 120 mg/dl", ""),
              ("hasMaximumHeartRateAchieved", "Maximum heart rate", ""),
              ("hasExerciseInducedAngina", "Exercise-induced angina", ""),
              ("hasSTDepressionInducedByExercise", "ST depression by exercise", ""),
              ("hasMajorVesselsColoredByFlourosopy", "Major vessels coloured by fluoroscopy", "")]
TYPE_NEGATIVES = ["chest pain", "gender", "heart rate", "resting electrocardiographic results",
                  "slope peak exercise ST segment"]  # true non-types of a patient (Patient ⊑ Person ⊑ Agent ⊑ Object)


def concept_name(iri_local):
    """DeepOnto's verbalisation of a class name: CamelCase -> lower-case words, keeping 'ST'."""
    words = re.findall(r"ST|[A-Z][a-z]+|[a-z]+", iri_local)
    return " ".join(w if w == "ST" else w.lower() for w in words)


# ---------------------------------------------------------------- prep
def stage_prep(a):
    import xml.etree.ElementTree as ET
    ns = {"o": "http://www.w3.org/2002/07/owl#"}
    root = ET.parse(a.kg).getroot()
    pats = {}
    for el in root.findall("o:ObjectPropertyAssertion", ns):
        prop = el.find("o:ObjectProperty", ns).get("IRI")[1:]
        s, o = (x.get("IRI")[1:] for x in el.findall("o:NamedIndividual", ns))
        if s.startswith("patient_") and prop != "hasDiagnosis":
            # the KG's ECG value individual "STWaveAnormality" (4 patients) is never typed; the ontology's
            # class for it is WaveAbnormality
            pats.setdefault(s, {"obj": {}, "data": {}})["obj"][prop] = {"STWaveAnormality": "WaveAbnormality"}.get(o, o)
    for el in root.findall("o:DataPropertyAssertion", ns):
        prop = el.find("o:DataProperty", ns).get("IRI")[1:]
        s = el.find("o:NamedIndividual", ns).get("IRI")[1:]
        if s.startswith("patient_"):
            pats.setdefault(s, {"obj": {}, "data": {}})["data"][prop] = el.find("o:Literal", ns).text
    ids = sorted(pats)
    sentences = {}
    for p in ids:
        lines = ["Type: patient"]
        for prop in ROLE:
            if prop in pats[p]["obj"]:
                lines.append(f"{LABEL[prop]}: {concept_name(pats[p]['obj'][prop])}")
        for prop, lab, unit in DATA_LINES:
            v = pats[p]["data"].get(prop)
            if v is not None:
                v = v if v in ("true", "false") else (str(int(float(v))) if float(v).is_integer() else v)
                lines.append(f"{lab}: {v}{unit}")
        sentences[p] = "\n".join(lines)
    import csv
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig"), delimiter=";"))
    assert len(rows) == len(ids), (len(rows), len(ids))
    for p, r in zip(ids, rows):  # the KG and the CSV list patients in the same order: check it
        assert abs(float(pats[p]["data"]["hasAge"]) - float(r["age"])) < 1e-6, p
    json.dump({"ids": ids, "patients": pats, "sentences": sentences}, open(os.path.join(OUT, "patients.json"), "w"), indent=1)
    shutil.copy(a.csv, os.path.join(OUT, "heart_dataset.csv"))
    print(len(ids), "patients; example sentence:\n" + sentences[ids[0]])


# ---------------------------------------------------------------- training (OnT's own fit, as train_ont.py)
def fit(data_dir, out_dir, epochs, batch, in_batch):
    from ont import pipeline
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    for f in os.listdir(data_dir):
        shutil.copy(os.path.join(data_dir, f), os.path.join(out_dir, "data", f))
    pipeline.fit(owl_path=os.path.join(OUT, "heart_tbox.owl"), output_dir=out_dir, num_epochs=epochs,
                 batch_size=batch, device=None, gradient_checkpointing=True, base_model="Hui97/OnT-MiniLM-L12-galen",
                 select_best_epoch=False, exist_in_batch_negatives=in_batch)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_ont(a):
    """Standard OnT: the TBox data exactly as OnT's prepare.py wrote it."""
    out = os.path.join(OUT, "runs", "ont")
    if not os.path.exists(os.path.join(out, "final")):
        fit(TBOX, out, a.ont_epochs, a.batch, False)
    print("standard OnT trained:", out)


def kont_data(train_ids, P, fold_dir):
    """TBox rows (fanned out, self-negatives dropped, as merge_datasets.py) at a fixed share + patient rows."""
    import merge_datasets as md
    tbox_type, _, _ = md.fan_out_and_filter_type_rows(read_jsonl(os.path.join(TBOX, "train.jsonl")))
    type_rows, exist_rows = [], []
    for p in train_ids:
        s = P["sentences"][p]
        type_rows += [{"child": s, "parent": "patient", "negative": [n]} for n in TYPE_NEGATIVES]
        for prop, o in P["patients"][p]["obj"].items():
            exist_rows.append({"Concept": s, "role": ROLE[prop], "con": concept_name(o)})
    concepts = set(json.load(open(os.path.join(TBOX, "concept_names.json"))).values())
    assert {r["con"] for r in exist_rows} <= concepts, {r["con"] for r in exist_rows} - concepts
    factor = max(1, round(0.15 / 0.85 * len(type_rows) / len(tbox_type)))
    import random
    rng = random.Random(0)
    merged = type_rows + tbox_type * factor
    rng.shuffle(merged)
    rng.shuffle(exist_rows)
    os.makedirs(fold_dir, exist_ok=True)
    write_jsonl(os.path.join(fold_dir, "train.jsonl"), merged)
    write_jsonl(os.path.join(fold_dir, "train_exist.jsonl"), exist_rows)
    for f in ("train_conj.jsonl", "val.json", "concept_names.json", "role_names.json", "role_inverse.json"):
        shutil.copy(os.path.join(TBOX, f), os.path.join(fold_dir, f))
    return len(merged), len(exist_rows)


def encode(run, sents):
    import torch
    from ont.model import OntologyTransformer
    m = OntologyTransformer.from_pretrained(os.path.join(run, "final"))
    m.hit_model.eval()
    with torch.no_grad():
        v = m.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()
        return m.manifold.logmap0(v).cpu().numpy()


def encode_base(sents):
    import torch
    from ont.hit import HierarchyTransformer
    m = HierarchyTransformer.from_pretrained("Hui97/OnT-MiniLM-L12-galen")
    m.eval()
    with torch.no_grad():
        v = m.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()
        return m.manifold.logmap0(v).cpu().numpy()


def concept_matrix(run_or_base, P):
    """patient x (5 x d): each patient's 5 value concepts, encoded by the given model, concatenated."""
    import numpy as np
    vocab = sorted({concept_name(o) for p in P["ids"] for o in P["patients"][p]["obj"].values()})
    E = encode_base(vocab) if run_or_base == "base" else encode(run_or_base, vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    d = E.shape[1]
    X = np.zeros((len(P["ids"]), len(ROLE) * d), dtype="float32")
    for i, p in enumerate(P["ids"]):
        for j, prop in enumerate(ROLE):
            o = P["patients"][p]["obj"].get(prop)
            if o:
                X[i, j * d:(j + 1) * d] = E[idx[concept_name(o)]]
    return X


def folds(y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(y, y))


def labels():
    import csv
    return [int(r["label"]) for r in csv.DictReader(open(os.path.join(OUT, "heart_dataset.csv"), encoding="utf-8-sig"), delimiter=";")]


def stage_folds(a):
    import numpy as np
    P = json.load(open(os.path.join(OUT, "patients.json")))
    y = np.array(labels())
    emb = os.path.join(OUT, "emb")
    os.makedirs(emb, exist_ok=True)
    sents = [P["sentences"][p] for p in P["ids"]]
    if not os.path.exists(os.path.join(emb, "base.npy")):
        np.save(os.path.join(emb, "base.npy"), encode_base(sents))
        np.save(os.path.join(emb, "ont_concepts.npy"), concept_matrix(os.path.join(OUT, "runs", "ont"), P))
    for k, (tr, te) in enumerate(folds(y)):
        out = os.path.join(emb, f"kont_fold{k}.npy")
        if os.path.exists(out):
            continue
        t0 = time.time()
        run = os.path.join(OUT, "runs", f"kont_fold{k}")
        shutil.rmtree(run, ignore_errors=True)
        n_t, n_e = kont_data([P["ids"][i] for i in tr], P, os.path.join(OUT, "folds", f"fold{k}"))
        fit(os.path.join(OUT, "folds", f"fold{k}"), run, a.kont_epochs, a.batch, True)
        np.save(out, encode(run, sents))
        np.save(os.path.join(emb, f"kont_concepts_fold{k}.npy"), concept_matrix(run, P))
        print(f"fold {k}: K-Ont on {len(tr)} patients ({n_t} type / {n_e} exist rows), "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        shutil.rmtree(os.path.join(run, "final"), ignore_errors=True)  # embeddings saved; keep disk small


# ---------------------------------------------------------------- evaluation
def raw_matrix():
    import pandas as pd
    df = pd.read_csv(os.path.join(OUT, "heart_dataset.csv"), sep=";", encoding="utf-8-sig")
    num = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    return df[num].to_numpy(float), pd.get_dummies(df[cat].astype(str)).to_numpy(float)


def mlp_probs(Xtr, ytr, Xte, seeds=(0, 1, 2)):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    out = []
    for sd in seeds:
        torch.manual_seed(sd)
        a, v, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=sd)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU(),
                            nn.Dropout(0.2), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        A, Ya, V, Yv = (torch.tensor(z, dtype=torch.float32) for z in (a, ya, v, yv))
        best, best_state, wait = 1e9, None, 0
        for ep in range(300):
            net.train()
            perm = torch.randperm(len(A))
            for i in range(0, len(A), 32):
                b = perm[i:i + 32]
                loss = nn.functional.binary_cross_entropy_with_logits(net(A[b]).squeeze(1), Ya[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = nn.functional.binary_cross_entropy_with_logits(net(V).squeeze(1), Yv).item()
            if vl < best - 1e-4:
                best, best_state, wait = vl, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 25:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            out.append(torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32)).squeeze(1)).numpy())
    return np.mean(out, 0)


def stage_eval(a):
    import numpy as np
    from scipy.stats import wilcoxon
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    y = np.array(labels())
    num, cat = raw_matrix()
    emb = os.path.join(OUT, "emb")
    base, ont = np.load(os.path.join(emb, "base.npy")), np.load(os.path.join(emb, "ont_concepts.npy"))
    models = {"LogReg": lambda: LogisticRegression(max_iter=5000),
              "RandomForest": lambda: RandomForestClassifier(n_estimators=500, random_state=0),
              "SVM": lambda: SVC(probability=True, random_state=0),
              "GradBoost": lambda: HistGradientBoostingClassifier(random_state=0),
              "MLP": None}
    sets = ["raw", "raw+ont", "raw+kont", "raw+base", "ont", "kont", "base"]
    res = {m: {s: [] for s in sets} for m in models}
    for k, (tr, te) in enumerate(folds(y)):
        kont = np.load(os.path.join(emb, f"kont_fold{k}.npy"))
        sc = StandardScaler().fit(num[tr])
        raw_tr, raw_te = np.hstack([sc.transform(num[tr]), cat[tr]]), np.hstack([sc.transform(num[te]), cat[te]])

        def pca(E):
            p = PCA(n_components=a.pca, random_state=0).fit(E[tr])
            s2 = StandardScaler().fit(p.transform(E[tr]))
            return s2.transform(p.transform(E[tr])), s2.transform(p.transform(E[te]))
        e = {"ont": pca(ont), "kont": pca(kont), "base": pca(base)}
        X = {"raw": (raw_tr, raw_te)}
        for n in ("ont", "kont", "base"):
            X[f"raw+{n}"] = (np.hstack([raw_tr, e[n][0]]), np.hstack([raw_te, e[n][1]]))
            X[n] = e[n]
        for mname, mk in models.items():
            for s in sets:
                Xtr, Xte = X[s]
                if mk is None:
                    prob = mlp_probs(Xtr, y[tr], Xte)
                else:
                    prob = mk().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
                pred = (prob >= 0.5).astype(int)
                res[mname][s].append({"acc": accuracy_score(y[te], pred), "auc": roc_auc_score(y[te], prob),
                                      "f1": f1_score(y[te], pred)})
        print(f"fold {k} done", flush=True)
    summary = {"protocol": "stratified 5-fold x 2 repeats; PCA %d on embeddings, fitted per training fold" % a.pca,
               "mean": {}, "paired": {}}
    for m in models:
        summary["mean"][m] = {s: {k: [round(float(np.mean([r[k] for r in res[m][s]])), 4),
                                      round(float(np.std([r[k] for r in res[m][s]])), 4)] for k in ("acc", "auc", "f1")}
                              for s in sets}
        summary["paired"][m] = {}
        for x, ref in (("raw+kont", "raw+ont"), ("raw+kont", "raw"), ("raw+kont", "raw+base"), ("raw+ont", "raw"),
                       ("kont", "ont"), ("kont", "base")):
            row = {}
            for k in ("acc", "auc"):
                d = np.array([r1[k] - r2[k] for r1, r2 in zip(res[m][x], res[m][ref])])
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                row[k] = {"diff": round(float(d.mean()), 4), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                          "p": round(p, 4)}
            summary["paired"][m][f"{x} - {ref}"] = row
    json.dump({"summary": summary, "folds": res}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "ont", "folds", "eval"])
    ap.add_argument("--kg", default=r"C:\Users\Shantharam P\Downloads\heart_disease_knowledge_graph.owl")
    ap.add_argument("--csv", default=r"C:\Users\Shantharam P\Downloads\heart_dataset.csv")
    ap.add_argument("--ont-epochs", type=int, default=20)
    ap.add_argument("--kont-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pca", type=int, default=16)
    a = ap.parse_args()
    {"prep": stage_prep, "ont": stage_ont, "folds": stage_folds, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
