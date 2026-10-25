"""A learned picker for the template + CrystaLLM pipeline (BUILD_LOG Step 57).

Step 48 commits to one structure per composition by CHGNet energy: right 69.3% on 300 targets, while some
candidate is right for 79.3% -- 10 points are lost in the choice. This learns the choice from signals that are
already there, per candidate (3 K-Ont templates + 5 CrystaLLM samples, same candidates and same CHGNet energies
as Step 48, cache kg_pipeline_energies_s40.json):
  energy        CHGNet energy/atom after relaxation, minus the target's lowest; its rank among the 8
  consensus     how many of the other candidates are the same structure (StructureMatcher 0.2/0.3/5, unrelaxed),
                split into "agreeing templates" and "agreeing CrystaLLM samples"; space-group agreement count
  source        template or CrystaLLM, its rank in its own list, the K-Ont ranker score (templates)
  crystal       space group number, crystal system, sites, density
Model: gradient boosting (and a logistic regression) on "candidate is right", the pick = highest score per
target. Evaluation: grouped 5-fold cross-validation over targets, repeated 5 times with different splits (no
target's candidates on both sides); paired bootstrap vs the energy pick. Rule baseline without learning:
"most-agreed structure, ties by energy".

    .venv-crystallm/Scripts/python.exe abox/kg_picker.py --stage feats
    .venv-crystallm/Scripts/python.exe abox/kg_picker.py --stage eval
"""
import argparse
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
ENERGY = os.path.join(DATA, "kg_pipeline_energies_s40.json")
FEATS = os.path.join(DATA, "kg_picker_feats.json")
OUT = os.path.join(DATA, "kg_picker.json")
GEN = os.path.join(DATA, "kgcond3_gen", "const")


def candidates(targets, top8):
    return {t: [(f"{t}|T|{c}", "T", i, s, c) for i, (c, s) in enumerate(top8[t][:3])] +
               [(f"{t}|C|{k}", "C", k, None, None) for k in range(5)] for t in targets}


def build(kind, target_formula, payload):
    """The unrelaxed candidate structure, exactly as kg_pipeline_select._relax builds it."""
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.core import Element, Structure
    if kind == "T":
        s = gen.reference_from_text(payload["cif"])
        old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.replace_species({a.symbol: b.symbol for a, b in zip(k2._ordered(payload["formula"]), k2._ordered(target_formula))})
        new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
        s.scale_lattice(s.volume * new / old)
        return s
    return Structure.from_str(gen.postprocess(open(payload).read()), fmt="cif")


def stage_feats(a):
    import kg_pool
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    pool, rows = kg_pool.load()
    E = json.load(open(ENERGY))
    targets = sorted({k.split("|")[0] for k in E})
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    cands = candidates(targets, top8)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    out = json.load(open(FEATS)) if os.path.exists(FEATS) else {}
    for n, t in enumerate(targets):
        if t in out:
            continue
        S, info = {}, {}
        for key, kind, rank, score, c in cands[t]:
            try:
                s = build(kind, rows[t]["formula"], rows[c] if kind == "T" else os.path.join(GEN, t, f"s_{rank}.cif"))
                sga = SpacegroupAnalyzer(s, symprec=0.1)
                S[key] = s
                info[key] = {"sg": sga.get_space_group_number(), "system": sga.get_crystal_system(),
                             "sites": len(s), "density": round(float(s.density), 4)}
            except Exception:
                continue
        keys = list(S)
        same = {k: set() for k in keys}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                try:
                    if sm.fit(S[keys[i]], S[keys[j]]):
                        same[keys[i]].add(keys[j])
                        same[keys[j]].add(keys[i])
                except Exception:
                    pass
        rec = {}
        for key, kind, rank, score, c in cands[t]:
            e = E.get(key)
            r = {"kind": kind, "rank": rank, "score": score, "energy": e[0] if e else None, "right": bool(e and e[1])}
            if key in info:
                r.update(info[key])
                r["agree_T"] = sum(x.split("|")[1] == "T" for x in same[key])
                r["agree_C"] = sum(x.split("|")[1] == "C" for x in same[key])
                r["agree_sg"] = sum(info[x]["sg"] == info[key]["sg"] for x in keys if x != key)
            rec[key] = r
        out[t] = rec
        if n % 20 == 0:
            json.dump(out, open(FEATS, "w"))
            print(f"{n + 1}/{len(targets)}", flush=True)
    json.dump(out, open(FEATS, "w"))
    print("features for", len(out), "targets")


def _init_refine(steps, fmax):
    global _M, _OPT, _STEPS, _FMAX
    import torch
    torch.set_num_threads(1)
    from chgnet.model import CHGNet, StructOptimizer
    _M = CHGNet.load(verbose=False)
    _OPT = StructOptimizer(model=_M, use_device="cpu")
    _STEPS, _FMAX = steps, fmax


def _refine(job):
    key, kind, formula, payload = job
    try:
        s = build(kind, formula, payload)
        fs = _OPT.relax(s, fmax=_FMAX, steps=_STEPS, verbose=False)["final_structure"]
        return key, round(float(_M.predict_structure(fs)["e"]), 5)
    except Exception:
        return key, None


def stage_refine(a):
    """MatterSim re-pick (the force field of the oxidation-state MatterGen pipeline): every DISTINCT candidate
    structure of a target within --window meV/atom of its lowest CHGNet 40-step energy (default: all of them) is
    relaxed with MatterSim on the GPU (abox/mattersim_relax.py, run in .venv-mattersim; --steps, --fmax);
    copies (same structure, unrelaxed) share the result. "right" is re-matched on the MatterSim-relaxed
    structure (or the unrelaxed one, as Step 48)."""
    import subprocess
    import kg_pool
    import phaseC_generate as gen
    from pymatgen.analysis.structure_matcher import StructureMatcher
    pool, rows = kg_pool.load()
    F = json.load(open(FEATS))
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    cands = candidates(sorted(F), top8)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    groups, jobs = {}, []
    for t in sorted(F):
        ok = {k: r for k, r in F[t].items() if r["energy"] is not None and "sg" in r}
        if not ok:
            continue
        e0 = min(r["energy"] for r in ok.values())
        near = [c for c in cands[t] if c[0] in ok and 1000 * (ok[c[0]]["energy"] - e0) <= a.window]
        ref = gen.reference_from_text(rows[t]["cif"]).as_dict()
        reps = []
        for key, kind, rank, score, c in near:
            payload = rows[c] if kind == "T" else os.path.join(GEN, t, f"s_{rank}.cif")
            try:
                s = build(kind, rows[t]["formula"], payload)
            except Exception:
                continue
            for rk, rs in reps:
                if sm.fit(s, rs):
                    groups[key] = rk
                    break
            else:
                reps.append((key, s))
                groups[key] = key
                jobs.append({"key": key, "structure": s.as_dict(), "ref": ref})
    jobs_p = os.path.join(DATA, "kg_picker_ms_jobs.json")
    json.dump(jobs, open(jobs_p, "w"))
    cache_p = os.path.join(DATA, f"kg_picker_ms_s{a.steps}.json")
    print(f"{len(groups)} candidates, {len(jobs)} distinct structures -> MatterSim", flush=True)
    py = os.path.join(HERE, ".venv-mattersim", "Scripts", "python.exe")
    subprocess.run([py, "-u", os.path.join(HERE, "abox", "mattersim_relax.py"), jobs_p, cache_p,
                    "--steps", str(a.steps), "--fmax", str(a.fmax)], check=True)
    cache = json.load(open(cache_p))
    json.dump({"model": "MatterSim-v1.0.0-1M", "groups": groups, "energy": {k: v and v[0] for k, v in cache.items()},
               "right": {k: v and v[1] for k, v in cache.items()}, "window_meV": a.window, "steps": a.steps,
               "fmax": a.fmax}, open(os.path.join(DATA, "kg_picker_refine.json"), "w"))
    print("MatterSim energies:", sum(v is not None for v in cache.values()))


def stage_refine_chgnet(a):
    """(superseded by the MatterSim re-pick) near-ties re-relaxed with CHGNet, longer and tighter."""
    import kg_pool
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from robust_pool import robust_map
    pool, rows = kg_pool.load()
    F = json.load(open(FEATS))
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    cands = candidates(sorted(F), top8)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    groups, jobs = {}, []
    for t in sorted(F):
        ok = {k: r for k, r in F[t].items() if r["energy"] is not None and "sg" in r}
        if len(ok) < 2:
            continue
        e0 = min(r["energy"] for r in ok.values())
        near = [c for c in cands[t] if c[0] in ok and 1000 * (ok[c[0]]["energy"] - e0) <= a.window]
        reps = []  # (representative candidate, structure)
        for key, kind, rank, score, c in near:
            payload = rows[c] if kind == "T" else os.path.join(GEN, t, f"s_{rank}.cif")
            try:
                s = build(kind, rows[t]["formula"], payload)
            except Exception:
                continue
            for rk, rs in reps:
                if sm.fit(s, rs):
                    groups[key] = rk
                    break
            else:
                reps.append((key, s))
                groups[key] = key
                jobs.append((key, kind, rows[t]["formula"], payload))
    cache_p = os.path.join(DATA, f"kg_picker_refine_s{a.steps}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = [j for j in jobs if j[0] not in cache]
    print(f"{len(groups)} near-tied candidates, {len(jobs)} distinct structures, {len(todo)} to relax", flush=True)

    def save(r):
        cache.update(r)
        json.dump(cache, open(cache_p, "w"))
    if todo:
        robust_map(_refine, todo, a.workers, save, initializer=_init_refine, initargs=(a.steps, a.fmax), chunk=24,
                   log=lambda m: print(m, flush=True))
    json.dump({"groups": groups, "energy": cache, "window_meV": a.window, "steps": a.steps, "fmax": a.fmax},
              open(os.path.join(DATA, "kg_picker_refine.json"), "w"))
    print("refined energies:", sum(v is not None for v in cache.values()))


SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def matrix(F, t):
    """Candidate rows of one target that can be picked (relaxed, energy known), with features."""
    import numpy as np
    keys = [k for k, r in F[t].items() if r["energy"] is not None and "sg" in r]
    if not keys:
        return keys, np.zeros((0, 1)), np.zeros(0)
    es = np.array([F[t][k]["energy"] for k in keys])
    order = es.argsort().argsort()
    X, y = [], []
    for i, k in enumerate(keys):
        r = F[t][k]
        X.append([es[i] - es.min(), order[i], float(r["kind"] == "T"), r["rank"], r["score"] if r["score"] is not None else -1.0,
                  r["agree_T"], r["agree_C"], r["agree_T"] + r["agree_C"], r["agree_sg"], r["sg"],
                  SYSTEMS.index(r["system"]), r["sites"], r["density"], len(keys)])
        y.append(float(r["right"]))
    return keys, np.array(X), np.array(y)


def stage_eval(a):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F = json.load(open(FEATS))
    targets = sorted(F)
    M = {t: matrix(F, t) for t in targets}

    def pick_by(scores_fn):
        return {t: (float(M[t][2][int(np.argmax(scores_fn(t)))]) if len(M[t][0]) else 0.0) for t in targets}
    arms = {"energy (Step 48)": pick_by(lambda t: -M[t][1][:, 0]),
            "consensus, ties by energy (rule)": pick_by(lambda t: M[t][1][:, 7] - 1e-3 * M[t][1][:, 0]),
            "oracle (any right)": {t: float(M[t][2].max()) if len(M[t][0]) else 0.0 for t in targets}}
    rp = os.path.join(DATA, "kg_picker_refine.json")
    if os.path.exists(rp):  # near-ties re-relaxed longer: the lowest refined energy among them wins
        R = json.load(open(rp))
        got = {}
        for t in targets:
            keys = M[t][0]
            if not keys:
                got[t] = 0.0
                continue
            near = [k for k in keys if k in R["groups"] and R["energy"].get(R["groups"][k]) is not None]
            if near:
                k = min(near, key=lambda k: (R["energy"][R["groups"][k]], R["groups"][k] != k))
                rep = R["groups"][k]
                got[t] = float(R.get("right", {}).get(rep, F[t][rep]["right"]) or 0)
            else:
                got[t] = arms["energy (Step 48)"][t]
        arms[f"{R.get('model', 'CHGNet')} pick ({R['steps']} steps, window {R['window_meV']:g} meV)"] = got
        oracle = {}
        for t in targets:  # ceiling with MatterSim-relaxed matching
            reps = {R["groups"][k] for k in M[t][0] if k in R["groups"]}
            oracle[t] = float(any(R.get("right", {}).get(r) for r in reps)) if reps else 0.0
        arms[f"oracle, {R.get('model', 'CHGNet')}-relaxed (any right)"] = oracle
    learners = {"learned picker (gradient boosting)": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                                               max_iter=200, random_state=0),
                "learned picker (logistic)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))}
    for name, mk in learners.items():
        reps = []
        for rep in range(a.repeats):
            got = {}
            for tr, te in KFold(5, shuffle=True, random_state=rep).split(targets):
                trt = [targets[i] for i in tr if len(M[targets[i]][0])]
                model = mk().fit(np.vstack([M[t][1] for t in trt]), np.concatenate([M[t][2] for t in trt]))
                for i in te:
                    t = targets[i]
                    got[t] = float(M[t][2][int(np.argmax(model.predict_proba(M[t][1])[:, 1]))]) if len(M[t][0]) else 0.0
            reps.append(got)
        arms[name] = {t: float(np.mean([r[t] for r in reps])) for t in targets}  # averaged over the CV repeats
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y):
        d = {t: arms[x][t] - arms[y][t] for t in targets}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
        return [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n_targets": len(targets), "cv": f"grouped 5-fold over targets x {a.repeats} repeats",
           "right_single_pick": {k: round(100 * sum(v.values()) / len(targets), 1) for k, v in arms.items()}}
    for k in arms:
        if k != "energy (Step 48)" and not k.startswith("oracle"):
            rep[f"{k} - energy"] = diff(k, "energy (Step 48)")
    json.dump({"report": rep}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["feats", "refine", "eval"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--window", type=float, default=1e9, help="meV/atom above the lowest CHGNet 40-step energy (default: all)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"feats": stage_feats, "refine": stage_refine, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
