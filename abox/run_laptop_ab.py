"""Laptop queue for Task A-test and Task B data prep (docs/HANDOFF_KG_STEERING.md).

  b1      CPU  abox/kg_corpus.py                          (starts at once, ~30 min)
  a_test  GPU  abox/seeded_generate.py --set test          (large model, batch 5; ~3 h incl. CHGNet)
  b2_gen  GPU  abox/dpo_pairs.py --stage generate          train (600), then val (150); small model
  b2_scr  CPU  abox/dpo_pairs.py --stage score / pairs     per split, once its generation and b1 are done
Logs: data/crystallm/laptop_logs/. Nothing is committed; results are reviewed first.

    .venv-crystallm/Scripts/python.exe -u abox/run_laptop_ab.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(HERE, "data", "crystallm", "laptop_logs")
PY = sys.executable


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def start(name, *args):
    log(f"start {name}: {' '.join(args)}")
    return subprocess.Popen([PY, "-u", os.path.join(HERE, "abox", args[0]), *args[1:]], cwd=HERE,
                            stdout=open(os.path.join(LOGS, name + ".log"), "w"), stderr=subprocess.STDOUT)


def finish(name, p):
    p.wait()
    log(f"{name} finished (exit {p.returncode})")
    return p.returncode


def main():
    os.makedirs(LOGS, exist_ok=True)
    b1 = start("b1_corpus", "kg_corpus.py", "--workers", "6")
    a = start("a_test", "seeded_generate.py", "--set", "test", "--model", "crystallm_v1_large", "--batch", "5",
              "--workers", "6", "--threads", "6")
    if finish("b1_corpus", b1) != 0:
        log("b1 failed: B2 needs the corpus, stopping after A-test")
        finish("a_test", a)
        return
    finish("a_test", a)
    scorers = []
    for split, n in (("train", "600"), ("val", "150")):
        g = start(f"b2_gen_{split}", "dpo_pairs.py", "--stage", "generate", "--split", split, "--n", n)
        if finish(f"b2_gen_{split}", g) != 0:
            continue
        # score + pairs for this split on the CPU while the next split generates on the GPU
        cmd = (f'"{PY}" -u abox/dpo_pairs.py --stage score --split {split} --workers 6 --chgnet-workers 3 '
               f'--chgnet-threads 4 && "{PY}" -u abox/dpo_pairs.py --stage pairs --split {split}')
        log(f"start b2_score_{split}")
        scorers.append((split, subprocess.Popen(cmd, cwd=HERE, shell=True,
                                                stdout=open(os.path.join(LOGS, f"b2_score_{split}.log"), "w"),
                                                stderr=subprocess.STDOUT)))
    for split, p in scorers:
        finish(f"b2_score_{split}", p)
    log("done")


if __name__ == "__main__":
    main()
