"""Numeric module: entities.json -> the float side-channel `x`, raw + z-scored, two variants.

`x` is kept separate from OnT's text encoding `v`. They're only glued together at export time
(z = concat(v, x)), AFTER OnT is trained on the sentences from verbalize.py -- this module only
builds `x` itself: the raw per-material float vector, the z-scored version, and the scaler
(mean/std) that produced it. See ONT_ABOX_EXTENSION.md SS8 and BUILD_LOG.md for why a language
model needs numbers handed to it this way (it reads "0.00" and "8.00" as similar TEXT, not
different magnitudes -- verified by the probe in SS8.1 of the spec doc).

Same PROPERTY vs GEOMETRY tiering as verbalize.py, same two variants:
  "full"        -- PROPERTY + GEOMETRY columns
  "no_geometry" -- PROPERTY columns only (matches the text variant of the same name exactly)

Missing-data rule (ONT_ABOX_EXTENSION.md SS8.2, applied literally, and MEASURED from the actual
population rather than hand-declared from what we already know about it -- so this stays correct
if the underlying KG's field coverage ever changes):
  - A column missing on MORE than half the population is DROPPED entirely, not zero-filled. A
    linear kg_proj can't tell "really zero" from "we don't know" once a column is on the same
    footing as every other number in the vector -- a mostly-missing column filled with 0 would
    look like a real, confident low value to everything downstream.
  - A column missing on FEWER than half is kept, with a companion missing-flag column ("0 + mask",
    same section) instead of silently imputing.

Applying that rule literally drops open_circuit_voltage and specific_capacity from the main table
-- they're the two values this project cares about most, but only 10/250 materials have them (96%
missing: this KG's electrode data only exists for the original 10-material cathode test set, see
BUILD_LOG.md Step 1 -- the live Materials Project ingestion path never fetches it, only the offline
cache does). Rather than silently lose them, or silently break the doc's own missing-data rule to
keep them, they get written to a SEPARATE, honestly-sparse file (battery_cell_features.json)
instead of being folded into x. Whoever builds the export step decides whether/how to use that
(e.g. only for those 10, or as an auxiliary target) -- this module's job is to report what's really
there, not make that call for them.

Usage:
    .venv/bin/python3 -m abox.numeric --entities data/battgpt_abox/entities.json --out-dir data/battgpt_abox/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MISSING_THRESHOLD = 0.5  # drop a column if more than this fraction of the population lacks it

# Sanity bounds carried over from Step 7 / verbalize.py -- a value outside these is treated as
# missing (not a real number), same "wrong is worse than missing" reasoning as the text side.
_BOUNDED_PROPERTIES = {"bulk_modulus": (0, 500), "shear_modulus": (0, 500), "poisson_ratio": (-1, 0.5)}


def _bool01(value) -> float | None:
    return None if value is None else (1.0 if bool(value) else 0.0)


def _prop(mat: dict, name: str) -> float | None:
    p = mat["properties"].get(name)
    if not p or p["value"] is None:
        return None
    v = float(p["value"])
    if name in _BOUNDED_PROPERTIES:
        lo, hi = _BOUNDED_PROPERTIES[name]
        if not (lo < v <= hi):
            return None
    return v


def _uc_field(mat: dict, entities: dict, field: str) -> float | None:
    crystal = entities["crystals"].get(mat["structure"]) if mat["structure"] else None
    if not crystal:
        return None
    uc = entities["unit_cells"].get(crystal["unit_cell"])
    if not uc:
        return None
    v = uc.get(field)
    return float(v) if v is not None else None


# name -> fn(material_dict, entities) -> float | None. Candidates only -- which ones actually
# survive into `x` is decided empirically below, from measured missingness, not from this list
# order. GEOMETRY_CANDIDATES additionally requires the "full" variant.
PROPERTY_CANDIDATES = {
    "band_gap": lambda m, e: _prop(m, "band_gap"),
    "formation_energy_per_atom": lambda m, e: _prop(m, "formation_energy_per_atom"),
    "energy_above_hull": lambda m, e: _prop(m, "energy_above_hull"),
    "e_fermi": lambda m, e: _prop(m, "e_fermi"),
    "total_magnetization": lambda m, e: _prop(m, "total_magnetization"),
    "bulk_modulus": lambda m, e: _prop(m, "bulk_modulus"),
    "shear_modulus": lambda m, e: _prop(m, "shear_modulus"),
    "poisson_ratio": lambda m, e: _prop(m, "poisson_ratio"),
    "is_metal": lambda m, e: _bool01(m.get("is_metal")),
    "is_stable": lambda m, e: _bool01(m.get("is_stable")),
    "is_gap_direct": lambda m, e: _bool01(m.get("is_gap_direct")),
    "is_theoretical": lambda m, e: _bool01(m.get("is_theoretical")),
    # Included here (not hand-excluded) so the >50%-missing measurement below is the thing that
    # decides their fate, honestly, from real coverage -- not an assumption baked in ahead of time.
    "open_circuit_voltage": lambda m, e: (lambda bc: _prop(bc, "open_circuit_voltage") if bc else None)(
        e["battery_cells"].get(m["battery_cell"]) if m.get("battery_cell") else None),
    "specific_capacity": lambda m, e: (lambda bc: _prop(bc, "specific_capacity") if bc else None)(
        e["battery_cells"].get(m["battery_cell"]) if m.get("battery_cell") else None),
}
GEOMETRY_CANDIDATES = {
    "lattice_a": lambda m, e: _uc_field(m, e, "a"),
    "lattice_b": lambda m, e: _uc_field(m, e, "b"),
    "lattice_c": lambda m, e: _uc_field(m, e, "c"),
    "lattice_alpha": lambda m, e: _uc_field(m, e, "alpha"),
    "lattice_beta": lambda m, e: _uc_field(m, e, "beta"),
    "lattice_gamma": lambda m, e: _uc_field(m, e, "gamma"),
    "volume": lambda m, e: _prop(m, "volume"),
    "density": lambda m, e: _prop(m, "density"),
    "num_sites": lambda m, e: _uc_field(m, e, "num_sites"),
}


def _measure_and_select(candidates: dict, materials: dict, entities: dict) -> tuple[dict, dict, dict]:
    """Run every candidate column over the population, measure its real missing fraction, and
    split into (kept, kept_with_mask, dropped) per the >50% rule. Returns:
      raw_values: {column_name: {material_id: float}}  (kept columns only, mask columns included)
      mask_flags: {column_name: {material_id: 1.0/0.0}}  (only for columns that needed a mask)
      report: {column_name: {"missing_frac": .., "kept": bool, "masked": bool}}
    """
    raw_values, mask_flags, report = {}, {}, {}
    for name, fn in candidates.items():
        values = {mid: fn(mat, entities) for mid, mat in materials.items()}
        missing = sum(1 for v in values.values() if v is None)
        frac = missing / len(values)
        if frac > MISSING_THRESHOLD:
            report[name] = {"missing_frac": round(frac, 3), "kept": False, "masked": False}
            logger.info(f"  DROP  {name}: {100*frac:.0f}% missing (> {100*MISSING_THRESHOLD:.0f}% threshold)")
            continue
        masked = missing > 0
        raw_values[name] = {mid: (v if v is not None else 0.0) for mid, v in values.items()}
        if masked:
            mask_flags[f"{name}_known"] = {mid: (0.0 if v is None else 1.0) for mid, v in values.items()}
        report[name] = {"missing_frac": round(frac, 3), "kept": True, "masked": masked}
        logger.info(f"  KEEP  {name}: {100*frac:.0f}% missing" + (" (+ mask column)" if masked else ""))
    return raw_values, mask_flags, report


class Scaler:
    """z-score per column: scaled = (value - mean) / std, fit once over a population.

    NOTE: ONT_ABOX_EXTENSION.md SS8.2 says to fit on TRAIN materials only. We don't have a
    train/test split yet (that's a Phase 5 concern -- needs held-out materials for evaluation,
    which needs more materials than we're confident classifying today). Fitting on all 250 for now
    is a clearly-logged placeholder, not a silent shortcut: refit this on the train subset only,
    the moment a real split exists, by passing that subset's ids to `fit()`.
    """
    def __init__(self, feature_names: list[str], means: list[float], stds: list[float]):
        self.feature_names = feature_names
        self.means = means
        self.stds = stds

    @classmethod
    def fit(cls, columns: dict[str, dict[str, float]], material_ids: list[str]) -> "Scaler":
        names = list(columns.keys())
        means, stds = [], []
        for name in names:
            vals = [columns[name][mid] for mid in material_ids]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            if std < 1e-9:
                logger.warning(f"  column '{name}' has ~zero variance (std={std:.2e}) over the fit "
                               f"population -- scaling will only center it, not spread it (dividing "
                               f"by a near-zero std would blow the value up arbitrarily).")
                std = 1.0
            means.append(mean); stds.append(std)
        return cls(names, means, stds)

    def transform(self, columns: dict[str, dict[str, float]], mid: str) -> list[float]:
        return [(columns[name][mid] - mean) / std
                for name, mean, std in zip(self.feature_names, self.means, self.stds)]

    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names, "means": self.means, "stds": self.stds}


def build_variant(materials: dict, entities: dict, include_geometry: bool) -> dict:
    logger.info(f"Measuring PROPERTY columns ({'full' if include_geometry else 'no_geometry'} variant):")
    prop_raw, prop_mask, prop_report = _measure_and_select(PROPERTY_CANDIDATES, materials, entities)
    all_raw, all_report = dict(prop_raw), dict(prop_report)
    all_raw.update(prop_mask)  # mask columns ride along as ordinary 0/1 columns from here on

    if include_geometry:
        logger.info("Measuring GEOMETRY columns:")
        geo_raw, geo_mask, geo_report = _measure_and_select(GEOMETRY_CANDIDATES, materials, entities)
        all_raw.update(geo_raw); all_raw.update(geo_mask)
        all_report.update(geo_report)

    material_ids = list(materials.keys())
    scaler = Scaler.fit(all_raw, material_ids)
    logger.info(f"Fit scaler on all {len(material_ids)} materials (placeholder -- see Scaler docstring): "
               f"{len(scaler.feature_names)} columns kept.")

    per_material = {}
    for mid in material_ids:
        raw_vec = [all_raw[name][mid] for name in scaler.feature_names]
        per_material[mid] = {"raw": raw_vec, "scaled": scaler.transform(all_raw, mid)}

    return {
        "feature_names": scaler.feature_names,
        "scaler": scaler.to_dict(),
        "column_report": all_report,
        "materials": per_material,
    }


def build_battery_cell_features(materials: dict, entities: dict) -> dict:
    """The sparse, dropped-from-x-but-not-discarded OCV/capacity block -- see module docstring."""
    out = {}
    for mid, mat in materials.items():
        bc = entities["battery_cells"].get(mat.get("battery_cell")) if mat.get("battery_cell") else None
        if not bc:
            continue
        ocv = _prop(bc, "open_circuit_voltage")
        cap = _prop(bc, "specific_capacity")
        if ocv is not None or cap is not None:
            out[mid] = {"open_circuit_voltage_V": ocv, "specific_capacity_mAh_g": cap}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/battgpt_abox/entities.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/battgpt_abox/"))
    args = ap.parse_args()

    entities = json.loads(args.entities.read_text())
    materials = entities["materials"]
    logger.info(f"Loaded {len(materials)} materials from {args.entities}")

    for variant, include_geometry in [("full", True), ("no_geometry", False)]:
        result = build_variant(materials, entities, include_geometry)
        out_path = args.out_dir / f"numeric_features_{variant}.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote {out_path}: {len(result['feature_names'])} columns x {len(materials)} materials\n")

    bc_features = build_battery_cell_features(materials, entities)
    bc_path = args.out_dir / "battery_cell_features.json"
    bc_path.write_text(json.dumps(bc_features, indent=2))
    logger.info(f"Wrote {bc_path}: {len(bc_features)}/{len(materials)} materials have real electrode data")


if __name__ == "__main__":
    main()
