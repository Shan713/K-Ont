"""Novel-cathode pilot queue (workstation): generate in chunks on the GPU, relax each finished chunk
with CHGNet on the CPU while the next chunk generates, then summarise, archive and (--push) commit.

200 novel compositions (data/crystallm/novel_candidates.json, from abox/novel_propose.py) x 2 prompts
(composition only; parent cathode's space group) x 10 samples, large CrystaLLM.

    python -u abox/novel_queue.py --push
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
LOGS = os.path.join(DATA, "novel_logs")
PY = sys.executable


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, wait):
    log(f"start {name}")
    p = subprocess.Popen([PY, "-u", os.path.join(HERE, "abox", args[0]), *args[1:]], cwd=HERE,
                         stdout=open(os.path.join(LOGS, name + ".log"), "w"), stderr=subprocess.STDOUT)
    if wait:
        p.wait()
        log(f"{name} finished (exit {p.returncode})")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--model", default="crystallm_v1_large")
    ap.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    n = len(json.load(open(os.path.join(DATA, "novel_candidates.json"), encoding="utf-8")))
    log(f"novel queue: {n} compositions, chunks of {a.chunk}, model {a.model}")
    cpu = []
    for start in range(0, n, a.chunk):
        tag = f"{start:03d}"
        run(f"gen_{tag}", ["phaseC_generate.py", "--candidates", "novel_candidates.json", "--model", a.model,
                           "--samples", str(a.samples), "--conditions", "composition,parent_sg",
                           "--gen-dir", "novel_gen", "--start", str(start), "--n-materials", str(start + a.chunk),
                           "--out", os.path.join(DATA, f"novel_results_{tag}.json")], wait=True)
        cpu.append(run(f"chgnet_{tag}", ["novel_chgnet.py", "--results", f"novel_results_{tag}.json",
                                         "--gen-dir", "novel_gen", "--out", f"novel_chgnet_{tag}.json",
                                         "--threads", str(max(1, a.cpu_threads // 2))], wait=False))
    for p in cpu:
        p.wait()
    log("all chunks finished")

    rows, wins = [], {"composition": 0, "parent_sg": 0, "tie": 0}
    cands = {c["id"]: c for c in json.load(open(os.path.join(DATA, "novel_candidates.json"), encoding="utf-8"))}
    for f in sorted(os.listdir(DATA)):
        if f.startswith("novel_chgnet_") and f.endswith(".json"):
            d = json.load(open(os.path.join(DATA, f)))
            for k, v in d["summary"]["lower_energy_prompt (5 meV/atom tie band)"].items():
                wins[k] += v
            for m in d["materials"]:
                if "best" in m:
                    c = cands[m["id"]]
                    rows.append((m["id"], m["formula"], c["substitution"], c["parent_formula"], c["parent_sg"],
                                 m["best"]["sg_after_relax"], m["best"]["condition"], m["best"]["e"],
                                 c["theoretical_capacity_mAh_g"]))
    lines = ["# Novel-cathode pilot summary", "",
             f"{len(rows)} of {n} compositions have a relaxed structure.",
             f"Lower-energy prompt per composition (5 meV/atom tie band): {wins}", "",
             "| id | formula | swap | parent | parent sg | best sg | from prompt | E (eV/atom) | capacity (mAh/g, upper bound) |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.4f} | {r[8]} |" for r in rows]
    open(os.path.join(LOGS, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    with tarfile.open(os.path.join(DATA, "novel_generated_cifs.tar.gz"), "w:gz") as tar:
        for d in ("novel_gen", "novel_best"):
            if os.path.isdir(os.path.join(DATA, d)):
                tar.add(os.path.join(DATA, d), arcname=d)
    log("summary + archive written")
    if a.push:
        files = [os.path.join("data", "crystallm", f) for f in sorted(os.listdir(DATA))
                 if f.startswith(("novel_results_", "novel_chgnet_")) and f.endswith(".json")]
        files += [os.path.join("data", "crystallm", "novel_logs"), os.path.join("data", "crystallm", "novel_generated_cifs.tar.gz")]
        subprocess.run(["git", "add", *files], cwd=HERE)
        subprocess.run(["git", "commit", "-m", "Novel-cathode pilot: generation (CrystaLLM large) and CHGNet relaxation"], cwd=HERE)
        r = subprocess.run(["git", "push"], cwd=HERE)
        log(f"git push exit {r.returncode}")
    log("done")


if __name__ == "__main__":
    main()
