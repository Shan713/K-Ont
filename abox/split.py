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
  * --group-polymorphs (Step 22, grown KG): materials sharing a formula (polymorphs) are one unit and
    go to the same side; otherwise a test crystal's sibling polymorph sits in training. Units are
    stratified by their family set ("mixed" when the polymorphs differ in family) and round(20%) of
    the units in each stratum go to test. Off by default, so the 250-material split is unchanged.

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


def make_grouped_split(entities: dict, seed: int = SEED, frac: float = TEST_FRACTION) -> dict:
    rng = random.Random(seed)
    fam_of = lambda m: (entities["crystals"][m["structure"]]["structure_family"] if m["structure"] else None) or "None"
    units = defaultdict(list)
    for mid, mat in sorted(entities["materials"].items()):
        units[mat["formula"]].append(mid)
    by_stratum = defaultdict(list)
    for formula, mids in sorted(units.items()):
        fams = {fam_of(entities["materials"][m]) for m in mids}
        by_stratum[fams.pop() if len(fams) == 1 else "mixed"].append(formula)

    test_units = []
    for stratum, formulas in sorted(by_stratum.items()):
        if len(formulas) < 2:
            continue
        test_units += rng.sample(formulas, max(1, round(frac * len(formulas))))
    test = sorted(m for f in test_units for m in units[f])
    train = sorted(m for m in entities["materials"] if m not in set(test))
    crystal = lambda mids: sorted(entities["materials"][m]["structure"] for m in mids if entities["materials"][m]["structure"])
    count = lambda mids: {f: sum(fam_of(entities["materials"][m]) == f for m in mids)
                          for f in sorted({fam_of(m) for m in entities["materials"].values()})}
    multi = [f for f, ms in units.items() if len(ms) > 1]
    return {
        "seed": seed, "test_fraction": frac, "grouped_by": "formula",
        "train_materials": train, "test_materials": test,
        "train_crystals": crystal(train), "test_crystals": crystal(test),
        "test_by_family": count(test), "total_by_family": count(list(entities["materials"])),
        "polymorph_groups_total": len(multi),
        "polymorph_groups_in_test": sum(f in set(test_units) for f in multi),
    }


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
    ap.add_argument("--group-polymorphs", action="store_true", help="keep materials sharing a formula on one side")
    a = ap.parse_args()
    entities = json.loads(a.entities.read_text(encoding="utf-8"))
    split = make_grouped_split(entities) if a.group_polymorphs else make_split(entities)
    a.out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"train {len(split['train_materials'])} / test {len(split['test_materials'])} materials")
    print("test per family:", split["test_by_family"], "of", split["total_by_family"])
    if a.group_polymorphs:
        print(f"polymorph groups in test: {split['polymorph_groups_in_test']} of {split['polymorph_groups_total']}")


if __name__ == "__main__":
    main()
