"""Phase C overnight queue: keeps the GPU busy with generation while CPU jobs score structures.

GPU queue (one job at a time):
  g1  calibration, small model: the 95 non-cathode Li/Na test compositions after the 55 cathode-like
      ones (composition + oracle prompts)                                   ~1 h
  g2  sanity set, small model: well-known materials CrystaLLM trained on   ~15 min
  g3  sanity set, large model                                               ~1 h
  g4  calibration, large model: the same 95 non-cathode compositions        ~5.5 h
CPU jobs (alongside):
  re-score the existing large and small_seeded generations with the corrected references
  (phaseC_nearmiss.py), CHGNet stability scoring of those generations (phaseC_chgnet.py), and a
  near-miss scoring of every new GPU job as soon as it finishes.
At the end: data/crystallm/overnight/summary.md, a tarball of all generated CIFs, and (with --push)
a git commit + push of the result files only.

    python -u abox/phaseC_overnight.py --push
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
LOGS = os.path.join(DATA, "overnight")
PY = sys.executable


def script(name, *args):
    return [PY, "-u", os.path.join(HERE, "abox", name), *args]


def gen(model, gen_dir, out, extra):
    return script("phaseC_generate.py", "--model", model, "--samples", "10", "--conditions", "composition,oracle",
                  "--gen-dir", gen_dir, "--out", os.path.join(DATA, out), *extra)


def nearmiss(results, gen_dir, out, candidates="phaseC_test_with_pred.json"):
    return script("phaseC_nearmiss.py", "--results", results, "--gen-dir", gen_dir, "--out", out,
                  "--candidates", candidates)


def chgnet(results, gen_dir, out, threads):
    return script("phaseC_chgnet.py", "--results", results, "--gen-dir", gen_dir, "--out", out,
                  "--max-per-cond", "5", "--device", "cpu", "--threads", str(threads))


GPU = [
    ("g1_calib_small", gen("crystallm_v1_small", "phaseC_gen_calib_small", "phaseC_results_calib_small.json",
                           ["--start", "55", "--n-materials", "150"]),
     nearmiss("phaseC_results_calib_small.json", "phaseC_gen_calib_small", "phaseC_nearmiss_calib_small.json")),
    ("g2_sanity_small", gen("crystallm_v1_small", "phaseC_gen_sanity_small", "phaseC_results_sanity_small.json",
                            ["--candidates", "phaseC_sanity_candidates.json", "--n-materials", "100"]),
     nearmiss("phaseC_results_sanity_small.json", "phaseC_gen_sanity_small", "phaseC_nearmiss_sanity_small.json",
              "phaseC_sanity_candidates.json")),
    ("g3_sanity_large", gen("crystallm_v1_large", "phaseC_gen_sanity_large", "phaseC_results_sanity_large.json",
                            ["--candidates", "phaseC_sanity_candidates.json", "--n-materials", "100"]),
     nearmiss("phaseC_results_sanity_large.json", "phaseC_gen_sanity_large", "phaseC_nearmiss_sanity_large.json",
              "phaseC_sanity_candidates.json")),
    ("g4_calib_large", gen("crystallm_v1_large", "phaseC_gen_calib_large", "phaseC_results_calib_large.json",
                           ["--start", "55", "--n-materials", "150"]),
     nearmiss("phaseC_results_calib_large.json", "phaseC_gen_calib_large", "phaseC_nearmiss_calib_large.json")),
]


def cpu_chain(threads):
    return [
        ("c1_rescore_large", nearmiss("phaseC_results_large.json", "phaseC_gen_large", "phaseC_nearmiss_large.json")),
        ("c2_rescore_small_seeded", nearmiss("phaseC_results_small_seeded.json", "phaseC_gen_small_seeded",
                                             "phaseC_nearmiss_small_seeded.json")),
        ("c3_chgnet_large", chgnet("phaseC_results_large.json", "phaseC_gen_large", "phaseC_chgnet_large.json", threads)),
        ("c4_chgnet_small_seeded", chgnet("phaseC_results_small_seeded.json", "phaseC_gen_small_seeded",
                                          "phaseC_chgnet_small_seeded.json", threads)),
    ]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def start(name, cmd):
    log(f"start {name}: {' '.join(cmd[2:])}")
    return subprocess.Popen(cmd, cwd=HERE, stdout=open(os.path.join(LOGS, name + ".log"), "w"),
                            stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="commit and push the result files at the end")
    ap.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="threads for CHGNet (default: half the cores, leaving the rest for the GPU jobs)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    chain = cpu_chain(a.cpu_threads)
    if a.dry_run:
        for name, cmd, *_ in GPU + chain:
            print(name, " ".join(cmd[2:]))
        return
    log(f"overnight queue started; CHGNet threads {a.cpu_threads}")
    pending = list(chain)
    cpu_proc = start(*pending.pop(0))
    post = []
    for name, cmd, after in GPU:
        p = start(name, cmd)
        while p.poll() is None:
            time.sleep(30)
            if cpu_proc and cpu_proc.poll() is not None:
                log(f"cpu job finished (exit {cpu_proc.returncode})")
                cpu_proc = start(*pending.pop(0)) if pending else None
        log(f"{name} finished (exit {p.returncode})")
        post.append(start(name + "_nearmiss", after))
    while (cpu_proc and cpu_proc.poll() is None) or pending:
        time.sleep(30)
        if cpu_proc and cpu_proc.poll() is not None:
            log(f"cpu job finished (exit {cpu_proc.returncode})")
            cpu_proc = start(*pending.pop(0)) if pending else None
    for p in post:
        p.wait()
    log("all jobs finished; writing summary")

    lines = ["# Phase C overnight summary", ""]
    for f in sorted(os.listdir(DATA)):
        if (f.startswith("phaseC_nearmiss_") or f.startswith("phaseC_chgnet_")) and f.endswith(".json"):
            try:
                summ = json.load(open(os.path.join(DATA, f))).get("summary", {})
            except Exception as e:
                summ = {"error": str(e)}
            lines += [f"## {f}", "```", json.dumps(summ, indent=1), "```", ""]
    open(os.path.join(LOGS, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))

    tar_path = os.path.join(DATA, "phaseC_generated_cifs.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for d in sorted(os.listdir(DATA)):
            if d.startswith("phaseC_gen") and os.path.isdir(os.path.join(DATA, d)):
                tar.add(os.path.join(DATA, d), arcname=d)
    log(f"generated CIFs archived: {os.path.getsize(tar_path) / 1e6:.1f} MB")

    if a.push:
        files = [os.path.join("data", "crystallm", f) for f in sorted(os.listdir(DATA))
                 if f.startswith(("phaseC_results_", "phaseC_nearmiss_", "phaseC_chgnet_")) and f.endswith(".json")]
        files += [os.path.join("data", "crystallm", "overnight"), os.path.join("data", "crystallm", "phaseC_generated_cifs.tar.gz")]
        subprocess.run(["git", "add", *files], cwd=HERE)
        subprocess.run(["git", "commit", "-m", "Phase C overnight: calibration, sanity set, re-scoring, CHGNet"], cwd=HERE)
        r = subprocess.run(["git", "push"], cwd=HERE)
        log(f"git push exit {r.returncode}")
    log("done")


if __name__ == "__main__":
    main()
