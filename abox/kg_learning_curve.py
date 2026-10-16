"""Does K-Ont's own training help when KG supervision is scarce? Retriever learning curve (BUILD_LOG Step 42).

kg_retriever_ft.py from the K-Ont model vs from the plain encoder it started from, on 5 / 10 / 25 / 50 /
100% of the training targets, 3 seeds; at each (fraction, seed) both inits get the SAME subset, candidates
and steps. Scored on the 902 K-Ont-unseen targets, retriever alone (no ranker on top): top-1 candidate has
the true space group, and the template generator's top-1 (kg_template_gen substitution, unrelaxed) matches
the reference structure. Paired difference K-Ont - plain per fraction, averaged over seeds, bootstrap over
targets.

    .venv-crystallm/Scripts/python.exe abox/kg_learning_curve.py --stage run    (calls .venv for training)
    .venv-crystallm/Scripts/python.exe abox/kg_learning_curve.py --stage eval
"""
import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
DATA = os.path.join(HERE, "data", "crystallm")
FRACS = (0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS = (0, 1, 2)
INITS = ("kont", "base")
OUT = os.path.join(DATA, "kg_learning_curve.json")


def tag(frac, seed):
    return f"_lc{int(frac * 100)}s{seed}"


def path(init, frac, seed):
    if frac == 1.0 and seed == 0:  # the Step 40 retrievers are exactly this configuration
        return os.path.join(DATA, f"kg_ft_scores_{init}.json")
    return os.path.join(DATA, f"kg_ft_scores_{init}{tag(frac, seed)}.json")


def stage_run(a):
    py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
    log = os.path.join(DATA, "kgcond3_logs", "learning_curve.log")
    for seed in SEEDS:
        for frac in FRACS:
            for init in INITS:
                if os.path.exists(path(init, frac, seed)):
                    continue
                with open(log, "a") as f:
                    f.write(f"== {init} frac {frac} seed {seed}\n")
                    f.flush()
                    subprocess.run([py, "-u", os.path.join(HERE, "abox", "kg_retriever_ft.py"), "--init", init, "--frac", str(frac),
                                    "--seed", str(seed), "--tag", tag(frac, seed)], stdout=f, stderr=subprocess.STDOUT, check=True)


def stage_eval(a):
    import kg_pool
    from kg_template_gen import _job
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    picks = {}
    for init in INITS:
        for frac in FRACS:
            for seed in SEEDS:
                p = path(init, frac, seed)
                if os.path.exists(p):
                    sc = json.load(open(p))
                    picks[(init, frac, seed)] = {t: max(sc[t], key=sc[t].get) for t in targets if sc.get(t)}
    need = sorted({(t, c) for d in picks.values() for t, c in d.items()})
    cache_p = os.path.join(DATA, "kg_template_ok_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = [(t, c) for t, c in need if f"{t}|{c}" not in cache]
    print(f"{len(picks)} runs, {len(need)} distinct picks, {len(todo)} new substitutions", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for (t, c), ok in ex.map(_job, [(rows[t], rows[c], False) for t, c in todo], chunksize=16):
            cache[f"{t}|{c}"] = ok
    json.dump(cache, open(cache_p, "w"))
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]
    rep = {"n_targets": len(targets), "fractions": {}}
    for frac in FRACS:
        seeds = [s for s in SEEDS if all((i, frac, s) in picks for i in INITS)]
        if not seeds:
            continue
        row = {"seeds": seeds}
        per = {}
        for init in INITS:
            sg = {t: sum(float(rows[picks[(init, frac, s)][t]]["space_group"] == rows[t]["space_group"]) for s in seeds) / len(seeds)
                  for t in targets}
            fw = {t: sum(float(cache[f"{t}|{picks[(init, frac, s)][t]}"]) for s in seeds) / len(seeds) for t in targets}
            per[init] = {"sg": sg, "template": fw}
            row[init] = {k: round(100 * sum(v.values()) / len(targets), 1) for k, v in per[init].items()}
            row[init]["per_seed_template"] = [round(100 * sum(float(cache[f"{t}|{picks[(init, frac, s)][t]}"]) for t in targets)
                                                    / len(targets), 1) for s in seeds]
        for k in ("sg", "template"):
            d = {t: per["kont"][k][t] - per["base"][k][t] for t in targets}
            bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
            row[f"kont - base ({k})"] = [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
        rep["fractions"][str(frac)] = row
    json.dump(rep, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    {"run": stage_run, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
