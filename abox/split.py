"""Held-out material split: entities.json -> split.json (train / test material + crystal keys).

Why (BUILD_LOG.md Step 17): every Phase 5 number so far was measured on materials the model also
trained on -- training fit, not generalization. The design spec's Phase 5b needs held-out materials,
and Phase 6 needs the numeric scaler fit on train materials only (Step 9's placeholder).

Rules:
  * 20% of materials held out, with a fixed seed, so the split is reproducible.
  * Stratified by structure family: each family with >= 2 members contributes round(20%) (at least
    one) to test, so held-out typing/hasStructureFamily covers the families. Families with a single
    member (garnet, LGPS-type today) stay in train -- holding out the only example would test a family
    the ABox side has never shown the model at all, a different question.
  * A material and its crystal always go to the SAME side (the 1:1 hasStructure link).
  * The 195 family-less materials are sampled at the same 20%.

What "held out" means downstream (enforced by abox/rows.py --split, not here): a test material and
its crystal appear in NO training row -- not as a child, not as an exist Concept/con, and not as
another material's hard-negative crystal. Elements are shared schema-level entities and stay in.

Usage:
    python -m abox.split --entities data/battgpt_abox/entities.json --out data/split.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260922
TEST_FRACTION = 0.2


def make_split(entities: dict, seed: int = SEED, frac: float = TEST_FRACTION) -> dict:
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for mid, mat in sorted(entities["materials"].items()):
        fam = entities["crystals"][mat["structure"]]["structure_family"] if mat["structure"] else None
        by_family[fam or "None"].append(mid)

    test = []
    for fam, mids in sorted(by_family.items()):
        if len(mids) < 2:
            continue
        k = max(1, round(frac * len(mids)))
        test += rng.sample(mids, k)
    test = sorted(test)
    train = sorted(m for m in entities["materials"] if m not in set(test))
    crystal = lambda mids: sorted(entities["materials"][m]["structure"] for m in mids if entities["materials"][m]["structure"])
    return {
        "seed": seed, "test_fraction": frac,
        "train_materials": train, "test_materials": test,
        "train_crystals": crystal(train), "test_crystals": crystal(test),
        "test_by_family": {f: sum(m in set(test) for m in ms) for f, ms in sorted(by_family.items())},
        "total_by_family": {f: len(ms) for f, ms in sorted(by_family.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/battgpt_abox/entities.json"))
    ap.add_argument("--out", type=Path, default=Path("data/split.json"))
    a = ap.parse_args()
    entities = json.loads(a.entities.read_text(encoding="utf-8"))
    split = make_split(entities)
    a.out.write_text(json.dumps(split, indent=2))
    print(f"train {len(split['train_materials'])} / test {len(split['test_materials'])} materials")
    print("test per family:", split["test_by_family"], "of", split["total_by_family"])


if __name__ == "__main__":
    main()
