"""Row generator: entities.json + verbalizations -> the actual JSONL files OnT reads.

Two real OnT row shapes (verified against the real training/loading code in Step 2 -- NOT the
{child, parent, negative} shape the design doc used for BOTH row kinds):

  train.jsonl        {"child": "...", "parent": "...", "negative": [...]}
                      Hierarchy loss (child nearer parent than negative; child farther from
                      origin than parent). The loader keeps only negative[:1] per row -- so this
                      module writes ONE ROW PER NEGATIVE, never a multi-element negative list.

  train_exist.jsonl   {"Concept": "...", "role": "...", "con": "..."}
                      Existential-role loss. NO negative field -- the exist loss borrows its
                      negatives from whatever train.jsonl batch happens to be running at the same
                      training step (ont/losses/logical_loss.py: LogicalConstraintLoss.forward
                      passes sentence_features[2], the CURRENT type-row batch's negative column,
                      into exist_loss as neg_samples). This is why hard negatives for hasStructure
                      are injected via train.jsonl below, not as a field here -- there is no field
                      to put them in.

**A modeling correction made before writing any of this** (checked directly against
BattGpt-Ontology/battgpt.ttl, not assumed): `hasStructureFamily` and `belongsToElectrode` are
`owl:ObjectProperty` -- relations between two individuals (crystal -> family individual, material
-> role individual) -- NOT rdf:type assertions. A materials-science reading might call a material
"an olivine" the way you'd call it "a substance", but the ontology itself models substance-hood as
a type and family/role as relations. So only ONE type/hierarchy fact exists per entity in this KG
(material rdf:type Substance, crystal rdf:type CrystalStructure, element rdf:type ChemicalElement)
-- structure family and battery role are BOTH exist rows here, alongside hasStructure, not type
rows. (This also means concept_names.json -- the class-only registry Phase 3/DeepOnto will build
from the TBox -- correctly has no place for "olivine" as a class name; it was never one.)

**Hard negatives for hasStructure** (design-doc pitfall #5: "easy exist negatives... include hard
crystals, not only random ones"), implemented the way the actual code allows rather than the way
that would be simplest to imagine: since exist-row negatives are borrowed from a CONCURRENTLY
SAMPLED train.jsonl batch (not a 1:1 pairing we can control), the only real lever is to raise how
often a genuinely confusable crystal (same chemical system, or failing that the same space group)
shows up as a NEGATIVE value somewhere in train.jsonl -- so every material gets one extra type row
whose parent is (truthfully) "substance" and whose negative is a hard-negative crystal sentence
instead of an abstract sibling-kind label. True fact, harder negative, no new loss pathway
invented.

This module produces ABox rows ONLY. Merging with TBox class-hierarchy rows (needs
DeepOnto + the ontology's OWL API + a tiny extracted schema file -- none of which exist in this
repo yet) and TBox oversampling are the design doc's Phase 3/4 -- separate, not-yet-built steps.
Output filenames say so (`train_abox_*`, not plain `train.jsonl`) so nobody mistakes this for the
final, ready-to-train file.

Usage:
    .venv/bin/python3 -m abox.rows --entities data/battgpt_abox/entities.json --out-dir data/battgpt_abox/
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from .verbalize import STRUCTURE_FAMILY_READABLE, BATTERY_ROLE_READABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 20260922  # fixed, so re-running reproduces identical rows -- same principle as "same URI,
                 # same string" in the verbalizer: a fact shouldn't silently change between runs.

# "What kind of thing is this" negatives -- short, definitely-true-negative labels for the one
# real rdf:type fact each entity kind has. Doc's own Phase 4 example uses exactly this style
# (parent="substance", negative=["crystal", "element", "battery role", "space group"]).
SIBLING_KIND_LABELS = {
    "substance": ["crystal structure", "chemical element", "battery role", "space group", "crystal system"],
    "crystal structure": ["substance", "chemical element", "battery role", "space group", "crystal system"],
    "chemical element": ["substance", "crystal structure", "battery role", "space group", "crystal system"],
}


def _family_individual_sentence(family_key: str) -> str:
    """Short V(.) for one of the 9 canonical StructureFamily individuals -- shared across every
    crystal that hasStructureFamily it, computed once, same string every time (there's exactly
    one such individual per family in the ontology). Uses the same "Label: value" shape as every
    other sentence in the verbalizer, deliberately NOT a capitalized standalone phrase -- an
    earlier version did `.capitalize()` on the readable name, which is correct for "spinel" ->
    "Spinel" but mangles the two names that carry their own internal casing ("NASICON" ->
    "Nasicon", "LGPS-type" -> "Lgps-type"). Caught by reading real generated rows, not by
    inspection -- see BUILD_LOG.md Step 10."""
    return f"Structure family: {STRUCTURE_FAMILY_READABLE[family_key]}"


def _role_individual_sentence(role_key: str) -> str:
    """Short V(.) for one of the 4 canonical BatteryRole individuals -- same reasoning/fix."""
    return f"Battery role: {BATTERY_ROLE_READABLE[role_key]}"


def _find_hard_negative_crystal(mid: str, mat: dict, entities: dict, rng: random.Random,
                                v_cryst: dict) -> str | None:
    """A crystal, NOT the material's own, to use as a hard negative: same chemical system first
    (near-identical composition -- the hardest real confusion), else same space group (same
    symmetry, different chemistry), else any other crystal at random. Returns a crystal key, or
    None if this is the only material in the KG (can't happen at 250, but don't assume).

    A candidate whose SENTENCE is identical to the material's own crystal's sentence is skipped at
    every tier: since Step 16, V(crystal) no longer names its material, so in no_geometry two crystals
    with the same symmetry read identically -- and "push away from the exact text you're also being
    pulled toward" is the same impossible target as Step 13's self-negative rows, not a hard negative.
    """
    own_crystal = mat["structure"]
    own_text = v_cryst.get(own_crystal) if own_crystal else None

    def distinct(keys):
        return [k for k in keys if v_cryst[k] != own_text]

    chemsys = mat["chemsys"]
    same_chemsys = distinct([
        m["structure"] for k, m in entities["materials"].items()
        if k != mid and m["chemsys"] == chemsys and m["structure"] and m["structure"] != own_crystal
    ])
    if same_chemsys:
        return rng.choice(same_chemsys)
    own_sg = entities["crystals"][own_crystal]["space_group"] if own_crystal else None
    if own_sg:
        same_sg = distinct([
            k for k, c in entities["crystals"].items()
            if k != own_crystal and c["space_group"] == own_sg
        ])
        if same_sg:
            return rng.choice(same_sg)
    others = distinct([k for k in entities["crystals"] if k != own_crystal])
    return rng.choice(others) if others else None


def build_type_rows(entities: dict, v_mat: dict, v_cryst: dict, v_elem: dict, rng: random.Random) -> list[dict]:
    rows = []
    for mid, mat in entities["materials"].items():
        for neg in SIBLING_KIND_LABELS["substance"]:
            rows.append({"child": v_mat[mid], "parent": "substance", "negative": [neg]})
        hard_neg_crystal = _find_hard_negative_crystal(mid, mat, entities, rng, v_cryst)
        if hard_neg_crystal:
            rows.append({"child": v_mat[mid], "parent": "substance", "negative": [v_cryst[hard_neg_crystal]]})
    for cid, cryst in entities["crystals"].items():
        for neg in SIBLING_KIND_LABELS["crystal structure"]:
            rows.append({"child": v_cryst[cid], "parent": "crystal structure", "negative": [neg]})
    for eid, el in entities["elements"].items():
        for neg in SIBLING_KIND_LABELS["chemical element"]:
            rows.append({"child": v_elem[eid], "parent": "chemical element", "negative": [neg]})
    return rows


def build_exist_rows(entities: dict, v_mat: dict, v_cryst: dict) -> list[dict]:
    rows = []
    for mid, mat in entities["materials"].items():
        if mat["structure"]:
            rows.append({"Concept": v_mat[mid], "role": "has structure", "con": v_cryst[mat["structure"]]})
        if mat["battery_role"]:
            rows.append({"Concept": v_mat[mid], "role": "belongs to electrode",
                        "con": _role_individual_sentence(mat["battery_role"])})
    for cid, cryst in entities["crystals"].items():
        if cryst["structure_family"]:
            rows.append({"Concept": v_cryst[cid], "role": "has structure family",
                        "con": _family_individual_sentence(cryst["structure_family"])})
    return rows


def restrict_to_train(entities: dict, split: dict) -> dict:
    """entities with materials/crystals cut down to the split's TRAIN side (Step 17). Every row
    builder -- and the hard-negative picker -- only ever iterates entities["materials"] /
    entities["crystals"], so this one filter keeps held-out materials and crystals out of every row,
    including as another material's hard negative. Elements are shared schema entities: untouched."""
    train_m, train_c = set(split["train_materials"]), set(split["train_crystals"])
    out = dict(entities)
    out["materials"] = {k: v for k, v in entities["materials"].items() if k in train_m}
    out["crystals"] = {k: v for k, v in entities["crystals"].items() if k in train_c}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/battgpt_abox/entities.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/battgpt_abox/"))
    ap.add_argument("--verbalizations-dir", type=Path, default=None,
                    help="where verbalizations_*.json live (default: --out-dir)")
    ap.add_argument("--split", type=Path, default=None,
                    help="split.json from abox/split.py: generate rows for TRAIN materials only")
    args = ap.parse_args()
    vdir = args.verbalizations_dir or args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entities = json.loads(args.entities.read_text())
    split = None
    if args.split:
        split = json.loads(args.split.read_text())
        entities = restrict_to_train(entities, split)
        logger.info(f"Split {args.split}: generating rows for {len(entities['materials'])} train materials / "
                    f"{len(entities['crystals'])} train crystals ({len(split['test_materials'])} held out)")
    v_elem = json.loads((vdir / "verbalizations_elements.json").read_text())

    meta = {}
    for variant in ("full", "no_geometry"):
        rng = random.Random(SEED)  # fresh per variant, so both variants pick the same hard negs
        v_mat = json.loads((vdir / f"verbalizations_materials_{variant}.json").read_text())
        v_cryst = json.loads((vdir / f"verbalizations_crystals_{variant}.json").read_text())

        type_rows = build_type_rows(entities, v_mat, v_cryst, v_elem, rng)
        exist_rows = build_exist_rows(entities, v_mat, v_cryst)

        type_path = args.out_dir / f"train_abox_type_{variant}.jsonl"
        with open(type_path, "w") as f:
            for row in type_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        exist_path = args.out_dir / f"train_abox_exist_{variant}.jsonl"
        with open(exist_path, "w") as f:
            for row in exist_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_hard_neg = sum(1 for r in type_rows if r["negative"][0] not in
                        SIBLING_KIND_LABELS["substance"] + SIBLING_KIND_LABELS["crystal structure"] + SIBLING_KIND_LABELS["chemical element"])
        n_family = sum(1 for r in exist_rows if r["role"] == "has structure family")
        n_role = sum(1 for r in exist_rows if r["role"] == "belongs to electrode")
        n_structure = sum(1 for r in exist_rows if r["role"] == "has structure")
        meta[variant] = {
            "n_type_rows": len(type_rows), "n_type_rows_hard_negative_crystal": n_hard_neg,
            "n_exist_rows": len(exist_rows), "n_exist_hasStructure": n_structure,
            "n_exist_hasStructureFamily": n_family, "n_exist_belongsToElectrode": n_role,
        }
        logger.info(f"[{variant}] wrote {type_path.name} ({len(type_rows)} rows) and "
                   f"{exist_path.name} ({len(exist_rows)} rows)")
        logger.info(f"[{variant}] breakdown: {meta[variant]}")

    if split:
        meta["split"] = {"file": str(args.split), "n_train_materials": len(split["train_materials"]),
                         "n_test_materials": len(split["test_materials"])}
    meta["note"] = ("ABox rows only. TBox class-hierarchy rows (needs DeepOnto + the ontology's "
                    "OWL API + a tiny extracted schema OWL file -- Phase 3, not built yet) and "
                    "TBox oversampling (Phase 4) are NOT included -- these files are not yet "
                    "'train.jsonl' in the sense the trainer would read directly.")
    (args.out_dir / "row_generation_summary.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote row_generation_summary.json")


if __name__ == "__main__":
    main()
