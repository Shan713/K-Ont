"""Runs OnT's own prepare_ontology_data() (OnT/ont/data/prepare.py) against our tiny schema OWL
(tbox/build_schema.py's output), producing the real TBox training data: train.jsonl,
train_exist.jsonl, train_conj.jsonl, val.json, concept_names.json, role_names.json,
role_inverse.json.

This is OnT's own code, unmodified and imported directly from the vendored OnT/ checkout -- we do
not reimplement DeepOnto verbalization ourselves. See BUILD_LOG.md for whether/how this actually
ran in this environment (needs a JVM + the deeponto package, both heavier dependencies than
anything else in K-Ont so far).

Usage:
    .venv/bin/python3 -m tbox.prepare --owl data/battgpt_tbox.owl --out-dir data/battgpt_ont/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ONT_ROOT = Path(__file__).resolve().parents[1] / "OnT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=Path, default=Path("data/battgpt_tbox.owl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/battgpt_ont/"))
    ap.add_argument("--jvm-memory", default="4g")
    args = ap.parse_args()

    if str(ONT_ROOT) not in sys.path:
        sys.path.insert(0, str(ONT_ROOT))
    from ont.data.prepare import prepare_ontology_data  # OnT's own code, unmodified

    logger.info(f"Running OnT's prepare_ontology_data() on {args.owl} -> {args.out_dir}")
    prepare_ontology_data(str(args.owl.resolve()), str(args.out_dir.resolve()))
    logger.info("Done.")


if __name__ == "__main__":
    main()
