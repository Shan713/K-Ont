"""Workstation queue for what is left after dropping Task D (docs/HANDOFF_KG_STEERING.md, BUILD_LOG Step 35).

  corpus  CPU  abox/kg_corpus.py                 rebuild the KG corpus in CrystaLLM's layout (~2 min; must
                                                 precede B3 -- the .json.gz files are not in git)
  s3      CPU  abox/novel_steer_analysis.py      pilot follow-up: KG plausibility, framework, shortlist
                                                 (runs alongside the GPU stages)
  sft     GPU  abox/kg_dpo.py --stage sft        CrystaLLM small on the KG corpus
  dpo     GPU  abox/kg_dpo.py --stage dpo        DPO on the B2 pairs, from the SFT model
  eval_c  GPU+CPU  --stage eval --targets crystallm   base / SFT / DPO on 100 held-out CrystaLLM test compositions
  eval_k  GPU+CPU  --stage eval --targets kg          the same on 100 KG test compositions
  report  paired bootstrap, both target sets
Logs: data/crystallm/ws_logs/. Nothing is committed; results are reviewed first.

    .venv-crystallm/Scripts/python.exe -u abox/run_workstation_b3.py --workers 12 --chgnet-workers 4
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(HERE, "data", "crystallm", "ws_logs")
PY = sys.executable
MODELS = "crystallm_v1_small,crystallm_kg_sft,crystallm_kg_dpo"


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


def run(name, *args):
    return finish(name, start(name, *args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12, help="CPU processes for KG scoring / s3")
    ap.add_argument("--chgnet-workers", type=int, default=4)
    ap.add_argument("--chgnet-threads", type=int, default=4)
    ap.add_argument("--skip-s3", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    w = ["--workers", str(a.workers), "--chgnet-workers", str(a.chgnet_workers), "--chgnet-threads", str(a.chgnet_threads)]
    if run("corpus", "kg_corpus.py", "--workers", str(a.workers)) != 0:
        log("corpus rebuild failed: stopping")
        return
    s3 = None if a.skip_s3 else start("s3", "novel_steer_analysis.py", "--workers", str(max(2, a.workers // 2)))
    if run("sft", "kg_dpo.py", "--stage", "sft") == 0 and run("dpo", "kg_dpo.py", "--stage", "dpo") == 0:
        for t in ("crystallm", "kg"):
            run(f"eval_{t}", "kg_dpo.py", "--stage", "eval", "--targets", t, "--models", MODELS, *w)
            run(f"report_{t}", "kg_dpo.py", "--stage", "report", "--targets", t, "--models", MODELS)
    else:
        log("sft or dpo failed: no evaluation")
    if s3:
        finish("s3", s3)
    log("done")


if __name__ == "__main__":
    main()
