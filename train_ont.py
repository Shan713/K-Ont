"""Run OnT's own fit() on a merged K-Ont dataset, timing every real training step.

Why this exists instead of the one-liner in docs/GPU_SETUP.md step 6: that snippet never configures
logging, so fit()'s own "on device: ..." line is silently dropped, and the only speed number you get
is tqdm's smoothed s/it. This wraps the trainer's training_step with a wall-clock timer (after a
cuda.synchronize(), so GPU work is actually finished before the clock stops) and writes every
step's time, plus peak GPU memory, to step_times.json next to the model. See BUILD_LOG.md Step 15.

    .venv\\Scripts\\python train_ont.py --variant full --epochs 1 --output data/battgpt_ont_output

OnT's fit() itself is called unmodified (beyond patches/ont_pipeline_device_fix.patch).
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "OnT"))

import torch  # noqa: E402
from ont import pipeline  # noqa: E402

MERGED_FILES = ("train.jsonl", "train_exist.jsonl", "train_conj.jsonl", "val.json",
                "concept_names.json", "role_names.json", "role_inverse.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "no_geometry"], default="full")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base-model", default="Hui97/OnT-MiniLM-L12-galen",
                    help="design spec Phase 5: the pretrained OnT galen model if available")
    ap.add_argument("--select-best-epoch", action="store_true",
                    help="upstream best-val-MRR epoch pick; off by default (val.json = 2 queries)")
    ap.add_argument("--data-prefix", default="data/battgpt_merged",
                    help="merged data dir is <prefix>_<variant> (Step 17 split: data/battgpt_merged_split)")
    ap.add_argument("--in-batch-negs", action="store_true",
                    help="in-batch contrastive term in the exist loss (Step 17, patches/ont_exist_inbatch_negatives.patch)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (OOMs an 8 GiB GPU on this data, Step 15)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("train_ont")

    # fit() reuses output/data if it's already populated, and otherwise runs DeepOnto prep on the
    # TBox alone -- which would silently train on 19 TBox axioms instead of the merged data. So
    # copy the merged files in first, and refuse to overwrite a different dataset already there.
    src = os.path.join(HERE, f"{a.data_prefix}_{a.variant}")
    dst = os.path.join(a.output, "data")
    os.makedirs(dst, exist_ok=True)
    for f in MERGED_FILES:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(d):
            with open(s, "rb") as x, open(d, "rb") as y:
                if x.read() != y.read():
                    sys.exit(f"{d} exists and differs from {s} -- refusing to mix datasets")
        else:
            shutil.copyfile(s, d)
    log.info(f"Training data: {src} -> {dst}")

    step_times = []
    orig = pipeline.HierarchyTransformerTrainer.training_step

    def timed_step(self, *args, **kwargs):
        if not step_times:  # read the real state off the model, don't trust the flag we passed
            hf = getattr(self.model[0], "auto_model", None)
            log.info(f"step 1 model device {next(self.model.parameters()).device}, HF gradient "
                     f"checkpointing active: {getattr(hf, 'is_gradient_checkpointing', 'unknown')}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        n = len(step_times)
        if n <= 5 or n % 10 == 0:
            log.info(f"step {n}: {dt:.3f}s (loss {float(out):.4f})")
        return out

    pipeline.HierarchyTransformerTrainer.training_step = timed_step

    t0 = time.time()
    pipeline.fit(owl_path=os.path.join(HERE, "data", "battgpt_tbox.owl"), output_dir=a.output,
                 num_epochs=a.epochs, batch_size=a.batch_size, device=a.device,
                 gradient_checkpointing=not a.no_grad_ckpt, base_model=a.base_model,
                 select_best_epoch=a.select_best_epoch, exist_in_batch_negatives=a.in_batch_negs)
    total = time.time() - t0

    s = sorted(step_times)
    summary = {
        "variant": a.variant, "epochs": a.epochs, "batch_size": a.batch_size,
        "gradient_checkpointing": not a.no_grad_ckpt, "base_model": a.base_model,
        "select_best_epoch": a.select_best_epoch, "data_dir": src, "exist_in_batch_negatives": a.in_batch_negs,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device != "cpu" else "cpu",
        "torch": torch.__version__,
        "n_steps": len(step_times), "total_wall_s": round(total, 1),
        "train_steps_total_s": round(sum(step_times), 1),
        "step_s_first": round(step_times[0], 3) if step_times else None,
        "step_s_median": round(s[len(s) // 2], 3) if s else None,
        "step_s_min": round(s[0], 3) if s else None, "step_s_max": round(s[-1], 3) if s else None,
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "step_times_s": [round(x, 4) for x in step_times],
    }
    with open(os.path.join(a.output, "step_times.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log.info("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "step_times_s"}))


if __name__ == "__main__":
    main()
