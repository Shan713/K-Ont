"""Merges ABox rows (abox/rows.py) with real TBox rows (tbox/prepare.py, via DeepOnto) into a
complete, self-contained directory OnT's loader (OnT/ont/data/load.py::load_local_dataset) can
read directly: train.jsonl, train_exist.jsonl, train_conj.jsonl, val.json, concept_names.json,
role_names.json, role_inverse.json all in one place.

This is the design doc's Phase 4 (merge_abox.py), with two fixes applied to the TBox rows on the
way in -- both to problems found by reading Step 12's real DeepOnto output, both implemented as
transformations OUR code applies at merge time, not as edits to OnT's own vendored files:

  1. FAN OUT negatives, one row per negative (same fix already applied to every ABox row in
     abox/rows.py). OnT's own prepare.py writes all 10 sampled negatives into one row; the loader
     keeps only negative[0]. Fanning out here means every sampled negative actually gets used as
     its own training example instead of 9 of them being silently discarded.
  2. DROP a fanned-out row if its single negative equals its own true parent -- a real collision
     found in Step 12 (7/19 raw TBox rows contained their own answer somewhere in the sampled
     negative list; fixed to a single negative per row, that collision would otherwise survive
     into `negative[0]` for 3 of them and train on a self-contradictory example forever).

Both are safe: neither touches what facts exist, only how many (correct) training examples each
fact contributes and whether an obviously-wrong one is filtered out before it can be trained on.

**Vocabulary consistency, checked, not assumed.** The ABox type rows (abox/rows.py) use plain
English sibling-kind labels ("crystal structure", "chemical element", ...) as parent/negative
values. Those strings only mean anything if they are EXACTLY the strings the real TBox
(concept_names.json / role_names.json, from Step 12) verbalizes those same classes/properties as
-- otherwise "substance" in an ABox row and "substance" in a TBox row would silently be two
unrelated strings that happen to look alike instead of the same concept OnT is meant to learn one
embedding for. Checked directly (not assumed) before this script was written: every ABox
class/role label is already an exact match. `verify_vocabulary()` below re-checks this on every
run, so a future change to either side that breaks the match fails loudly instead of silently.

**TBox oversampling** (design doc SS4, target ~1:1 between TBox and ABox type rows, "do not pick
50-100x in isolation -- compute the counts"): computed here from the real post-fan-out counts, not
asserted. The exist-row channel needs a much larger factor than the type-row channel purely
because our schema currently declares only 3 object properties total -- flagged in the log output,
not silently applied without comment.

Usage:
    .venv/bin/python3 merge_datasets.py
"""
from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 20260922  # same seed already used in abox/rows.py -- consistent, reproducible merges
ABOX_DIR = Path("data/battgpt_abox")
TBOX_DIR = Path("data/battgpt_ont")
VARIANTS = ("full", "no_geometry")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fan_out_and_filter_type_rows(rows: list[dict]) -> tuple[list[dict], int, int]:
    """One row per negative (fixes the loader's negative[:1] truncation -- Step 2/12), dropping
    any resulting row whose single negative equals its own true PARENT or its own CHILD.

    The parent case (Step 12) makes the row a no-op: d(child,parent) < d(child,parent) can never
    be satisfied or improved, contributing margin-constant, zero-gradient loss.

    The CHILD case is worse, and was missed on the first pass of this script -- caught only by
    reading actual merged rows by eye, not by re-deriving it from Step 12's numbers (which only
    checked parent-collisions). d(child, child) = 0, so the row asks the model to make
    d(child, parent) SMALLER than zero, which is impossible -- an unsatisfiable target rather
    than a merely wasted one. 9/19 raw TBox rows (more common than the parent case's 7/19) contain
    their own child somewhere in the sampled negatives, because prepare.py's negative sampling
    (OnT/ont/data/prepare.py) draws from the full concept list without excluding EITHER endpoint
    of the axiom it's building a negative for.

    Returns (fanned_rows, n_before_fanout, n_dropped_self_negative)."""
    out, dropped = [], 0
    for r in rows:
        for neg in r["negative"]:
            if neg == r["parent"] or neg == r["child"]:
                dropped += 1
                continue
            out.append({"child": r["child"], "parent": r["parent"], "negative": [neg]})
    return out, len(rows), dropped


def verify_vocabulary(abox_type_rows: list[dict], abox_exist_rows: list[dict],
                       concept_names: dict, role_names: dict) -> None:
    """Every parent/negative string an ABox type row uses, and every role string an ABox exist
    row uses, must be an exact member of the real TBox vocabulary -- checked here on every run,
    not assumed from a one-time manual check. Raises, loudly, rather than silently merging
    mismatched vocabularies."""
    concept_values = set(concept_names.values())
    role_values = set(role_names.values())

    used_concepts = set()
    for r in abox_type_rows:
        used_concepts.add(r["parent"])
        used_concepts.add(r["negative"][0])
    # Hard-negative rows use a full crystal/material SENTENCE as their negative, not a class
    # label -- those are individuals, deliberately not TBox concepts (abox/rows.py), so only
    # check the ones that look like a short class-tier label (no newline = not a full sentence).
    used_concepts = {c for c in used_concepts if "\n" not in c}
    missing_concepts = used_concepts - concept_values
    if missing_concepts:
        raise ValueError(f"ABox type rows use class labels not in the real TBox concept_names.json: {missing_concepts}")

    used_roles = {r["role"] for r in abox_exist_rows}
    missing_roles = used_roles - role_values
    if missing_roles:
        raise ValueError(f"ABox exist rows use role labels not in the real TBox role_names.json: {missing_roles}")

    logger.info(f"Vocabulary check passed: {len(used_concepts)} ABox class labels and "
               f"{len(used_roles)} ABox role labels all match the real TBox vocabulary exactly.")


def main():
    tbox_type_raw = _read_jsonl(TBOX_DIR / "train.jsonl")
    tbox_exist = _read_jsonl(TBOX_DIR / "train_exist.jsonl")
    concept_names = json.loads((TBOX_DIR / "concept_names.json").read_text())
    role_names = json.loads((TBOX_DIR / "role_names.json").read_text())

    tbox_type, n_tbox_raw, n_dropped = fan_out_and_filter_type_rows(tbox_type_raw)
    logger.info(f"TBox type rows: {n_tbox_raw} raw (10 negatives each) -> {len(tbox_type)} after "
               f"fan-out, {n_dropped} dropped as self-negative (see module docstring)")

    for variant in VARIANTS:
        rng = random.Random(SEED)
        abox_type = _read_jsonl(ABOX_DIR / f"train_abox_type_{variant}.jsonl")
        abox_exist = _read_jsonl(ABOX_DIR / f"train_abox_exist_{variant}.jsonl")

        verify_vocabulary(abox_type, abox_exist, concept_names, role_names)

        # Oversample TBox rows toward the design doc's ~1:1 target against the ABox count in the
        # SAME channel. Computed from real counts, applied as stated -- but logged with a clear
        # flag when the resulting factor is unusually high, since that says more about our
        # schema's small size (23 classes, 3 relations) than about the right oversampling policy.
        type_factor = max(1, round(len(abox_type) / len(tbox_type)))
        exist_factor = max(1, round(len(abox_exist) / len(tbox_exist))) if tbox_exist else 1

        tbox_type_oversampled = tbox_type * type_factor
        tbox_exist_oversampled = tbox_exist * exist_factor

        merged_type = abox_type + tbox_type_oversampled
        merged_exist = abox_exist + tbox_exist_oversampled
        rng.shuffle(merged_type)
        rng.shuffle(merged_exist)

        out_dir = Path(f"data/battgpt_merged_{variant}")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "train.jsonl", merged_type)
        _write_jsonl(out_dir / "train_exist.jsonl", merged_exist)
        # Conjunction rows, val.json, and the concept/role registries are TBox-only artifacts
        # (design doc pitfall #2: no ABox instance strings in concept_names.json) -- copied
        # through unchanged, not regenerated, so they stay exactly what Step 12 produced.
        for fname in ("train_conj.jsonl", "val.json", "concept_names.json", "role_names.json", "role_inverse.json"):
            shutil.copy(TBOX_DIR / fname, out_dir / fname)

        meta = {
            "variant": variant,
            "n_abox_type": len(abox_type), "n_tbox_type_raw": n_tbox_raw,
            "n_tbox_type_fanned": len(tbox_type), "n_tbox_type_dropped_self_negative": n_dropped,
            "type_oversample_factor": type_factor, "n_tbox_type_oversampled": len(tbox_type_oversampled),
            "n_merged_type": len(merged_type),
            "n_abox_exist": len(abox_exist), "n_tbox_exist_raw": len(tbox_exist),
            "exist_oversample_factor": exist_factor, "n_tbox_exist_oversampled": len(tbox_exist_oversampled),
            "n_merged_exist": len(merged_exist),
        }
        (out_dir / "merge_summary.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[{variant}] wrote {out_dir}/: {meta}")
        if type_factor > 100 or exist_factor > 100:
            logger.warning(f"[{variant}] oversample factor exceeds the design doc's own "
                           f"'do not pick 50-100x in isolation' guidance (type={type_factor}x, "
                           f"exist={exist_factor}x) -- current TBox has only {len(concept_names)} "
                           f"concepts / {len(role_names)} roles; growing the schema itself would "
                           f"bring this down more honestly than a larger repeat count would.")


if __name__ == "__main__":
    main()
