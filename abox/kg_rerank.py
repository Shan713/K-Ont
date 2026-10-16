"""Learned analogue retrieval with K-Ont knowledge (BUILD_LOG Step 40).

Step 39: a single cosine between K-Ont material vectors picks worse analogues than the composition rule.
Step 38's split shows where the gain from injection has to come from: the analogue was right for 43/79
targets (generation already 68-70% right there with or without K-Ont) and wrong for 36 (16%); a right
candidate exists for 73-80% of targets. So: LEARN which candidate shares the target's framework, from
features that are available before generation, and ablate what each knowledge source adds.

Pair features (target t = composition prompt + its KG facts, candidate c = KG train material, same
anonymous formula, different formula):
  prompt   element-by-element similarity (today's rule), identical-element count, Z match (the prompt's cell
           gives the atom count), candidate's e_above_hull, pool size, share of the pool with c's space group
           / crystal system (how common c's prototype is for this formula type)
  kg       the target's and candidate's KG property facts (band gap, formation energy, hull distance, Fermi
           energy, magnetisation, bulk/shear modulus, Poisson ratio, metallic, stable) and their differences
  kont     K-Ont: cosine of structure-free material vectors (t, c), cosine of t's material vector with c's
           framework (crystal_anon) vector, t's K-Ont score for c's structure-family concept, and the
           elementwise product of the two material vectors in a 16-d PCA (lets the model weigh directions)
  base     the same cosines / product from the untrained encoder (control for "is it just text?")
Model: gradient-boosted trees, pointwise "same space group" label, pick = top score per target.
Candidates come from the K-Ont-seen pool (kg_pool.py); training and evaluation only on the 902 materials
K-Ont never saw (grouped 5-fold cross-fitting, so every prediction is out-of-fold); the 79 Step 37-38
generation targets are reported separately. Also writes each method's top 3 with scores (generation),
and top 8 to kg_rerank_top8.json (template generator).
Metric: pick has the true space group / same framework (fit_anonymous), bootstrap CI vs composition.

    .venv-crystallm/Scripts/python.exe abox/kg_rerank.py
"""
import gzip
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

EXPORT = os.path.join(HERE, "data", "kont_export", "grow5000_v2")
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(DATA, "kg_rerank.json")
PROPS = {"Band gap": "gap", "Formation energy": "form", "Energy above hull": "hull", "Fermi energy": "fermi",
         "Total magnetization": "mag", "Bulk modulus": "bulk", "Shear modulus": "shear", "Poisson ratio": "poisson",
         "Metallic": "metal", "Stable": "stable"}
GROUPS = {"composition rule": None, "prompt": ["prompt"], "prompt+kg": ["prompt", "kg"],
          "prompt+kont": ["prompt", "kont"], "prompt+kg+kont": ["prompt", "kg", "kont"],
          "prompt+kg+base": ["prompt", "kg", "base"],
          # fine-tuned retrievers (kg_retriever_ft.py): K-Ont init vs plain-encoder init
          "ft_kont alone": ["ft_kont"], "ft_base alone": ["ft_base"],
          "prompt+ft_kont": ["prompt", "ft_kont"], "prompt+ft_base": ["prompt", "ft_base"],
          "prompt+kg+ft_kont": ["prompt", "kg", "ft_kont"], "prompt+kg+ft_base": ["prompt", "kg", "ft_base"]}
PAIRS = (("prompt+kg+kont", "prompt+kg"), ("prompt+kont", "prompt"), ("prompt+kg+kont", "prompt+kg+base"),
         ("ft_kont alone", "ft_base alone"), ("prompt+ft_kont", "prompt"), ("prompt+ft_kont", "prompt+ft_base"),
         ("prompt+kg+ft_kont", "prompt+kg"), ("prompt+kg+ft_kont", "prompt+kg+ft_base"))


def corpus(split):
    return json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "rt", encoding="utf-8"))


def facts(sentence):
    d = {}
    for line in sentence.split("\n"):
        k, _, v = line.partition(": ")
        if k in PROPS:
            if v in ("true", "false"):
                d[PROPS[k]] = float(v == "true")
            else:
                m = re.match(r"-?[\d.]+", v)
                d[PROPS[k]] = float(m.group()) if m else np.nan
    return [d.get(p, np.nan) for p in PROPS.values()]


def _fit(args):
    a, b = args
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    try:
        return bool(StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit_anonymous(Structure.from_dict(a), Structure.from_dict(b)))
    except Exception:
        return False


def main():
    from pymatgen.core import Composition, Element
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from kg_condition2 import similarity, _ordered
    import phaseC_generate as gen

    ids = json.load(open(os.path.join(EXPORT, "id_maps.json")))
    row = {i: k for k, i in enumerate(ids["material"])}
    crow = {i: k for k, i in enumerate(ids["crystal"])}
    concept = {c.replace(" ", "").lower(): k for k, c in enumerate(ids["concept"])}
    z = np.load(os.path.join(EXPORT, "embeddings.npz"))
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    V = {"kont": unit(z["material_cond_tan"]), "base": unit(np.load(os.path.join(EXPORT, "material_cond_base_tan.npy")))}
    anon_c = unit(z["crystal_anon_tan"])
    conc = unit(z["concept_tan"])
    P = {k: PCA(16, random_state=0).fit_transform(v) for k, v in V.items()}
    for k in P:
        P[k] /= P[k].std(0)
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    F = {i: facts(vm["material/" + i]) for i in ids["material"]}

    import kg_pool
    pool, allrows = kg_pool.load()  # Step 40: K-Ont-seen analogue pool, K-Ont-unseen targets
    rows = {s: corpus(s) for s in ("test",)}
    unseen = set(pool["targets"])
    targets = [allrows[i] for i in pool["targets"]]
    eval85 = {r["id"] for r in rows["test"] if r["id"] in unseen}

    def fam_score(t, fam):
        name = (fam or "").lower()
        return float(V["kont"][row[t]] @ conc[concept[name]]) if name in concept else np.nan

    FT = {k: json.load(open(os.path.join(DATA, f"kg_ft_scores_{k}.json"))) for k in ("kont", "base")}
    X, meta = {g: [] for g in ("prompt", "kg", "kont", "base", "ft_kont", "ft_base")}, []
    for t in targets:
        cands = [allrows[c] for c in pool["candidates"][t["id"]]]
        if not cands:
            continue
        sg_n, cs_n = Counter(c["space_group"] for c in cands), Counter(c["crystal_system"] for c in cands)
        to = [e.symbol for e in _ordered(t["formula"])]
        ft = F[t["id"]]
        comp = [similarity(t["formula"], c["formula"]) for c in cands]
        ftsc = {}
        for k in ("kont", "base"):
            v = [FT[k].get(t["id"], {}).get(c["id"], np.nan) for c in cands]
            mx = np.nanmax(v) if not all(np.isnan(v)) else np.nan
            order = {j: r for r, j in enumerate(sorted(range(len(v)), key=lambda j: -(v[j] if v[j] == v[j] else -9)))}
            ftsc[k] = [[v[j], v[j] - mx, order[j] / len(v)] for j in range(len(v))]
        best_comp = max(range(len(cands)), key=lambda i: (comp[i], -(cands[i]["e_above_hull"] or 0)))
        for i, c in enumerate(cands):
            same = sum(a == b for a, b in zip(to, [e.symbol for e in _ordered(c["formula"])]))
            X["prompt"].append([comp[i], same, float(c["n_atoms"] == t["n_atoms"]), c["n_atoms"] / t["n_atoms"],
                                c["e_above_hull"] or 0, len(cands), sg_n[c["space_group"]] / len(cands),
                                cs_n[c["crystal_system"]] / len(cands), float(i == best_comp)])
            fc = F[c["id"]]
            X["kg"].append(ft + fc + [abs(a - b) for a, b in zip(ft, fc)])
            for k in ("kont", "base"):
                vt, vc = V[k][row[t["id"]]], V[k][row[c["id"]]]
                f = [float(vt @ vc)] + list(P[k][row[t["id"]]] * P[k][row[c["id"]]])
                if k == "kont":
                    f += [float(vt @ anon_c[crow[c["id"]]]), fam_score(t["id"], c["structure_family"])]
                X[k].append(f)
            for k in ("kont", "base"):
                X["ft_" + k].append(ftsc[k][i])
            meta.append((t["id"], i, c["id"], c["space_group"] == t["space_group"], float(i == best_comp)))
    X = {k: np.array(v, dtype=np.float32) for k, v in X.items()}
    tid = np.array([m[0] for m in meta])
    y = np.array([m[3] for m in meta], dtype=int)
    tlist = sorted(set(tid))
    print(f"{len(tlist)} unseen targets with candidates, {len(meta)} pairs, positive share {y.mean():.3f}", flush=True)

    rng = np.random.RandomState(0)
    fold_of = dict(zip(tlist, rng.permutation(len(tlist)) % 5))
    folds = np.array([fold_of[t] for t in tid])
    picks, top3, top8 = {}, {}, {}
    for name, groups in GROUPS.items():
        if groups is None:
            score = np.array([m[4] for m in meta])
        else:
            Xg = np.hstack([X[g] for g in groups])
            score = np.zeros(len(y))
            if groups in (["ft_kont"], ["ft_base"]):  # the retriever's own score, no learning on top
                score = np.nan_to_num(Xg[:, 0], nan=-9)
                groups = None
            for f in (range(5) if groups else ()):
                tr, te = folds != f, folds == f
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                     l2_regularization=1.0, random_state=0).fit(Xg[tr], y[tr])
                score[te] = clf.predict_proba(Xg[te])[:, 1]
        ranked = {}
        for k, m in enumerate(meta):
            ranked.setdefault(m[0], []).append((float(score[k]), m[2], bool(m[3])))
        for t in ranked:
            ranked[t].sort(key=lambda x: -x[0])
        top3[name] = {t: [(c, s_) for s_, c, _ in v[:3]] for t, v in ranked.items()}
        top8[name] = {t: [(c, s_) for s_, c, _ in v[:8]] for t, v in ranked.items()}
        picks[name] = {t: (v[0][1], v[0][2], v[0][0]) for t, v in ranked.items()}
        print(f"{name}: SG {100 * np.mean([v[1] for v in picks[name].values()]):.1f}", flush=True)

    # framework check for every distinct (target, pick)
    S = {}
    need = sorted({(t, p[0]) for d in picks.values() for t, p in d.items()})
    for i in {x for pair in need for x in pair}:
        try:
            S[i] = gen.reference_from_text(allrows[i]["cif"]).as_dict()
        except Exception:
            S[i] = None
    jobs = [(S[t], S[c]) for t, c in need]
    with ProcessPoolExecutor(6) as ex:
        fr = dict(zip(need, ex.map(_fit, jobs, chunksize=8)))
    oracle = {t: bool(y[tid == t].any()) for t in tlist}

    rep, boot = {}, random.Random(0)
    for subset, tset in (("unseen_all", tlist), ("eval79", [t for t in tlist if t in eval85])):
        B = [[boot.choice(tset) for _ in tset] for _ in range(10000)]
        res = {"n": len(tset), "oracle_sg": round(100 * np.mean([oracle[t] for t in tset]), 1)}
        for name, d in picks.items():
            sg = {t: float(d[t][1]) for t in tset}
            fw = {t: float(fr[(t, d[t][0])]) for t in tset}
            r = {"sg": round(100 * np.mean(list(sg.values())), 1), "frame": round(100 * np.mean(list(fw.values())), 1)}
            if name != "composition rule":
                base = picks["composition rule"]
                for key, m in (("sg", sg), ("frame", fw)):
                    ref = {t: float(base[t][1]) if key == "sg" else float(fr[(t, base[t][0])]) for t in tset}
                    diff = {t: m[t] - ref[t] for t in tset}
                    bs = sorted(100 * sum(diff[t] for t in b) / len(b) for b in B)
                    r[f"{key}_vs_composition"] = [round(100 * np.mean(list(diff.values())), 1), round(bs[250], 1), round(bs[9749], 1)]
            res[name] = r
        rep[subset] = res
    # paired: does K-Ont add to prompt+kg?
    for subset, tset in (("unseen_all", tlist), ("eval79", [t for t in tlist if t in eval85])):
        B = [[boot.choice(tset) for _ in tset] for _ in range(10000)]
        for a_, b_ in PAIRS:
            for key in ("sg", "frame"):
                g = lambda n, t: float(picks[n][t][1]) if key == "sg" else float(fr[(t, picks[n][t][0])])
                diff = {t: g(a_, t) - g(b_, t) for t in tset}
                bs = sorted(100 * sum(diff[t] for t in b) / len(b) for b in B)
                rep[subset][f"{a_} - {b_} ({key})"] = [round(100 * np.mean(list(diff.values())), 1), round(bs[250], 1), round(bs[9749], 1)]
    json.dump(top8, open(os.path.join(DATA, "kg_rerank_top8.json"), "w"))  # Step 41 template generator
    json.dump({"report": rep, "picks": {n: {t: list(p) + [fr[(t, p[0])]] for t, p in d.items()} for n, d in picks.items()},
               "top3": top3},
              open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
