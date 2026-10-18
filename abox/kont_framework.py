"""One K-Ont model for ontology reasoning AND crystal retrieval? (BUILD_LOG Step 43)

The retriever that drives the template generator (kg_retriever_ft.py) is K-Ont fine-tuned on one new
KG relation -- "this material's framework is that pool material's" (same anonymous formula, same space
group). Does K-Ont keep what it knew? Two variants from the trained K-Ont (data/runs/full_grow5000_v2):
  fw_plain    the retrieval objective alone
  fw_retain   + retention loss (MSE to the start model's embeddings of every KG material, crystal, element
              and ontology-class sentence, weight 1)
Each is packaged as a K-Ont run (fine-tuned encoder + the original role model / wrapper / concept names)
and put through the same phase5_checks as K-Ont itself (typing, hasStructure, roles, families, hierarchy,
with the split), and its retrieval scores feed the template generator (kg_template_gen-style top-1/2/3).

    .venv/Scripts/python.exe abox/kont_framework.py --stage train
    .venv/Scripts/python.exe abox/kont_framework.py --stage checks
    .venv-crystallm/Scripts/python.exe abox/kont_framework.py --stage eval
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
SRC = os.path.join(HERE, "data", "runs", "full_grow5000_v2")
DATA = os.path.join(HERE, "data", "crystallm")
VARIANTS = {"fw_plain": [], "fw_retain": ["--retain", "1.0"],
            # Step 52: + 20,000 GALEN concept names kept at the original OnT-galen embeddings (GALEN forgetting, Step 49)
            "fw_retain_galen": ["--retain", "1.0", "--retain-galen", "20000"],
            # Step 52b: the same, GALEN as a separate balanced term (Step 52 lost battery roles/families)
            "fw_retain_galen_bal": ["--retain", "1.0", "--retain-galen", "20000", "--retain-balanced"],
            # Step 52c: + K-Ont's relation names and role/family label phrases retained every step
            "fw_retain_galen_anchor": ["--retain", "1.0", "--retain-galen", "20000", "--retain-balanced", "--retain-anchors"]}


def run_dir(v):
    return os.path.join(HERE, "data", "runs", f"kont_{v}")


def stage_train(a):
    for v, extra in VARIANTS.items():
        d = run_dir(v)
        if os.path.exists(os.path.join(DATA, f"kg_ft_scores_kont_{v}.json")):
            continue
        cmd = [sys.executable, "-u", os.path.join(HERE, "abox", "kg_retriever_ft.py"), "--init", "kont", "--tag", f"_{v}",
               "--save", os.path.join(d, "final")] + extra
        subprocess.run(cmd, check=True)
        for f in ("role_model.pt", "wrapper_config.json"):  # the K-Ont parts the encoder fine-tune does not touch
            shutil.copy(os.path.join(SRC, "final", f), os.path.join(d, "final", f))
        shutil.copytree(os.path.join(SRC, "data"), os.path.join(d, "data"), dirs_exist_ok=True)
        shutil.copy(os.path.join(SRC, "step_times.json"), d)


def stage_checks(a):
    for v in VARIANTS:
        d = run_dir(v)
        if os.path.exists(os.path.join(d, "phase5_checks.json")):
            continue
        with open(os.path.join(d, "phase5_stdout.txt"), "w") as f:
            subprocess.run([sys.executable, "-u", os.path.join(HERE, "abox", "phase5_checks.py"), "--run", d, "--variant", "full",
                            "--split", os.path.join(HERE, "data", "split_grow_5000.json"),
                            "--abox-dir", os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2")],
                           stdout=f, stderr=subprocess.STDOUT, check=True)


def headline(path):
    """the Step 36 acceptance numbers, held-out side where the check has one"""
    r = json.load(open(path))["after"]

    def g(*ks):
        x = r
        for k in ks:
            x = x[k]
        return x
    out = {"material typing H@1": g("B_typing", "materials", "H@1"), "crystal typing H@1": g("B_typing", "crystals", "H@1"),
           "role, no label line (test)": g("G_belongsToElectrode", "test", "labels_removed", "acc"),
           "family, no label line (test)": g("H_hasStructureFamily", "test", "labels_removed", "acc")}
    hs = r.get("C_hasStructure", {})
    for k in ("MRR", "H@1"):
        if k in hs:
            out[f"hasStructure {k}"] = hs[k]
    ih = r.get("I_hierarchy", {}).get("tbox_axioms", {})
    for k in ("MRR", "H@1", "auc"):
        if k in ih:
            out[f"TBox {k}"] = ih[k]
    return out


def stage_eval(a):
    import random
    import kg_pool
    from concurrent.futures import ProcessPoolExecutor
    from kg_template_gen import _job
    pool, rows = kg_pool.load()
    targets = pool["targets"]
    rep = {"checks": {"K-Ont (Step 36)": headline(os.path.join(SRC, "phase5_checks.json"))}}
    for v in VARIANTS:
        p = os.path.join(run_dir(v), "phase5_checks.json")
        if os.path.exists(p):
            rep["checks"][v] = headline(p)
    scores = {"K-Ont retriever (Step 40)": "kg_ft_scores_kont.json", "plain-encoder retriever": "kg_ft_scores_base.json",
              **{v: f"kg_ft_scores_kont_{v}.json" for v in VARIANTS}}
    top = {n: {t: sorted(s[t], key=s[t].get, reverse=True)[:3] for t in targets if s.get(t)}
           for n, f in scores.items() if os.path.exists(os.path.join(DATA, f)) for s in [json.load(open(os.path.join(DATA, f)))]}
    cache_p = os.path.join(DATA, "kg_template_ok_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = sorted({(t, c) for d in top.values() for t, cs in d.items() for c in cs if f"{t}|{c}" not in cache})
    with ProcessPoolExecutor(5) as ex:
        for (t, c), ok in ex.map(_job, [(rows[t], rows[c], False) for t, c in todo], chunksize=16):
            cache[f"{t}|{c}"] = ok
    json.dump(cache, open(cache_p, "w"))
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]
    found = {n: {k: {t: float(any(cache[f"{t}|{c}"] for c in d[t][:k])) for t in targets} for k in (1, 2, 3)} for n, d in top.items()}
    rep["template"] = {n: {f"found@{k}": round(100 * sum(v.values()) / len(targets), 1) for k, v in f.items()} for n, f in found.items()}
    for v in VARIANTS:
        if v in found:
            for k in (1, 3):
                d = {t: found[v][k][t] - found["K-Ont retriever (Step 40)"][k][t] for t in targets}
                bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B)
                rep["template"][f"{v} - Step 40 retriever @{k}"] = [round(100 * sum(d.values()) / len(targets), 1), round(bs[125], 1), round(bs[4874], 1)]
    json.dump(rep, open(os.path.join(DATA, "kont_framework.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "checks", "eval"])
    a = ap.parse_args()
    {"train": stage_train, "checks": stage_checks, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
