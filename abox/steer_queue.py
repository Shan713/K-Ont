"""KG-steering queue (workstation), after the novel-cathode pilot:
  s1  CPU  re-score the Phase C generations with per-CIF scores + the combined KG-filter -> CHGNet metric
  s2  GPU  KG-steered sampling on the 55 test cathodes (large model; until 5 KG-plausible, max 40)
  s3  CPU  KG/ontology steering applied to the novel pilot's structures + shortlist
Waits for the pilot (data/crystallm/novel_logs/driver.log ending in "done") before s2 and s3.
At the end, with --push, commits and pushes the result files only.

    python -u abox/steer_queue.py --push
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
LOGS = os.path.join(DATA, "steer_logs")
PY = sys.executable


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def start(name, *args):
    log(f"start {name}")
    return subprocess.Popen([PY, "-u", os.path.join(HERE, "abox", args[0]), *args[1:]], cwd=HERE,
                            stdout=open(os.path.join(LOGS, name + ".log"), "w"), stderr=subprocess.STDOUT)


def pilot_done():
    p = os.path.join(DATA, "novel_logs", "driver.log")
    return os.path.exists(p) and open(p, encoding="utf-8").read().strip().endswith("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    w = str(a.workers)
    s1 = start("s1_rescore", "kg_steer.py", "--runs", "large,small_seeded,calib_large,calib_small",
               "--workers", w, "--save-scores", "--out", "kg_steer_eval.json")
    while not pilot_done():
        log("waiting for the novel pilot to finish")
        time.sleep(300)
    s2 = start("s2_steered_sampling", "kg_steer_sample.py", "--model", "crystallm_v1_large", "--n-materials", "55",
               "--workers", w, "--out", "kg_steer_sample_large.json")
    s1.wait(); log(f"s1 finished (exit {s1.returncode})")
    s3 = start("s3_novel_analysis", "novel_steer_analysis.py", "--workers", w)
    s3.wait(); log(f"s3 finished (exit {s3.returncode})")
    s2.wait(); log(f"s2 finished (exit {s2.returncode})")
    if os.path.isdir(os.path.join(DATA, "kg_steer_gen")):
        with tarfile.open(os.path.join(DATA, "kg_steer_generated_cifs.tar.gz"), "w:gz") as tar:
            tar.add(os.path.join(DATA, "kg_steer_gen"), arcname="kg_steer_gen")
    if a.push:
        names = ["kg_steer_eval.json", "kg_steer_sample_large.json", "novel_steer_analysis.json", "novel_shortlist.md",
                 "kg_steer_generated_cifs.tar.gz", "steer_logs"]
        names += [f for f in os.listdir(DATA) if f.startswith("kg_scores_") and f.endswith(".json")]
        files = [os.path.join("data", "crystallm", n) for n in names if os.path.exists(os.path.join(DATA, n))]
        subprocess.run(["git", "add", *files], cwd=HERE)
        subprocess.run(["git", "commit", "-m", "KG steering: combined selection, steered sampling, novel shortlist"], cwd=HERE)
        r = subprocess.run(["git", "push"], cwd=HERE)
        log(f"git push exit {r.returncode}")
    log("done")


if __name__ == "__main__":
    main()
