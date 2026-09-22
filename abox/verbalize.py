"""Verbalizer: entities.json -> V(a) sentences, in two variants.

Two variants, both built from the same entities.json, differing only in which fields are allowed
into the sentence:

  "full"         -- every field, including 3D geometry (lattice, volume, density, per-site
                    element counts).
  "no_geometry"  -- drops the geometry fields listed below, from the TEXT (not just the numeric
                    channel, which doesn't exist yet -- see BUILD_LOG.md Step 3/9). Geometry is
                    what CrystaLLM is asked to *predict*; training OnT to read it as an input would
                    let a downstream lookup hand CrystaLLM the answer to its own question, and an
                    unseen formula (the real use case) has no geometry to look up anyway, so a
                    model that leans on it would get a mismatched input at the moment it matters.

Field tiers (this tiering IS the leakage decision -- not left configurable elsewhere):

  IDENTITY / PROPERTY / SYMMETRY  -> both variants
      formula, chemical system, structure family, battery role, metallic, stable, band gap,
      formation energy, energy above hull, Fermi energy, total magnetization, bulk/shear modulus
      (sanity-bounded, see below), Poisson ratio, open-circuit voltage, specific capacity, space
      group, crystal system. Space group / crystal system are symmetry labels, not coordinates --
      CrystaLLM's own prompt format already accepts a target space group as an input (see
      CrystaLLM's make_prompt_file.py), so stating it isn't handing over anything CrystaLLM
      couldn't already be told directly.
  GEOMETRY  -> "full" variant only
      unit cell lattice lengths/angles, volume, density, per-site element counts. Bond distances
      are left out of BOTH variants -- with ~80 bonds on some crystals (Step 7's crystal-bond
      count / 250 crystals), any per-bond listing blows the 256-token budget long before reaching
      the properties that matter; the design doc allows omitting bonds entirely when they'd be
      long (ONT_ABOX_EXTENSION.md SS7), so we do.

Sanity bounds carried over directly from Step 7's data-quality finding: bulk_modulus/shear_modulus
outside (0, 500] GPa and poisson_ratio outside [-1, 0.5] are DROPPED from the sentence (not just
flagged) -- writing "Bulk modulus: -4729.5 GPa" into training text would be actively wrong, worse
than the missing-column problem the original spec worried about.

Crystals carry no properties of their own in this KG (verified in extract.py / Step 7) -- band
gap, formation energy etc. for V(crystal) are looked up from the crystal's 1:1 owning material
(same mp- id suffix, verified with 0 mismatches across all 250 in Step 7).

Usage:
    .venv/bin/python3 -m abox.verbalize --entities data/battgpt_abox/entities.json --out-dir data/battgpt_abox/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STRUCTURE_FAMILY_READABLE = {
    "LayeredOxideStructure": "layered oxide",
    "SpinelStructure": "spinel",
    "RockSaltStructure": "rock salt",
    "OlivineStructure": "olivine",
    "NASICONStructure": "NASICON",
    "GarnetStructure": "garnet",
    "PerovskiteStructure": "perovskite",
    "LGPSTypeStructure": "LGPS-type",
    "ArgyroditeStructure": "argyrodite",
}
BATTERY_ROLE_READABLE = {
    "PositiveElectrode": "positive electrode (cathode)",
    "NegativeElectrode": "negative electrode (anode)",
    "Electrolyte": "electrolyte",
    "Separator": "separator",
}
# (label, property key, decimals, unit override-or-None) for the PROPERTY-tier lines that are a
# plain "print this number" job. band_gap/e_fermi/total_magnetization/OCV/capacity are handled
# separately because they need extra logic (directness suffix, or come from a different entity).
_SIMPLE_PROPERTY_LINES = [
    ("Formation energy", "formation_energy_per_atom", 3, "/atom"),
    ("Energy above hull", "energy_above_hull", 3, "/atom"),
]
# Sanity bounds from Step 7: (low, high) exclusive-low/inclusive-high range a value must fall in
# to be trusted. Matches the same scan extract.py's spot-check ran across all 250 materials.
_ELASTIC_BOUNDS = {"bulk_modulus": (0, 500), "shear_modulus": (0, 500)}
_POISSON_BOUNDS = (-1, 0.5)


def _site_element_counts(uc: dict, entities: dict) -> str:
    """'14 sites' -> a compact per-element count summary, e.g. 'Li:2, Ti:4, O:8', instead of
    listing all 14 (mostly repeated) symbols -- more informative (frequency) and shorter."""
    counts: dict[str, int] = {}
    for site_key in uc["sites"]:
        species_key = entities["sites"][site_key]["species"]
        elem_key = entities["species"][species_key]["element"]
        symbol = entities["elements"][elem_key]["symbol"]
        counts[symbol] = counts.get(symbol, 0) + 1
    return ", ".join(f"{sym}:{n}" for sym, n in counts.items())


def verbalize_material(mat: dict, entities: dict, variant: str) -> str:
    lines = [
        f"Material: {mat['formula']} ({mat['material_project_id']})",
        "Type: substance",
        f"Formula: {mat['formula']}",
        f"Chemical system: {mat['chemsys']}",
    ]
    crystal = entities["crystals"].get(mat["structure"]) if mat["structure"] else None

    if crystal and crystal["structure_family"]:
        lines.append(f"Structure family: {STRUCTURE_FAMILY_READABLE[crystal['structure_family']]}")
    if mat["battery_role"]:
        lines.append(f"Battery role: {BATTERY_ROLE_READABLE[mat['battery_role']]}")
    if mat["is_metal"] is not None:
        lines.append(f"Metallic: {str(bool(mat['is_metal'])).lower()}")
    if mat["is_stable"] is not None:
        lines.append(f"Stable: {str(bool(mat['is_stable'])).lower()}")

    props = mat["properties"]
    if "band_gap" in props:
        gap = props["band_gap"]
        directness = ""
        if gap["value"] and mat.get("is_gap_direct") is not None:
            directness = " (direct)" if mat["is_gap_direct"] else " (indirect)"
        lines.append(f"Band gap: {gap['value']:.2f} {gap['unit']}{directness}")
    for label, key, decimals, unit_suffix in _SIMPLE_PROPERTY_LINES:
        if key in props and props[key]["value"] is not None:
            v = props[key]
            lines.append(f"{label}: {v['value']:.{decimals}f} {v['unit']}{unit_suffix}")
    if "e_fermi" in props and props["e_fermi"]["value"] is not None:
        v = props["e_fermi"]
        lines.append(f"Fermi energy: {v['value']:.2f} {v['unit']}")
    if "total_magnetization" in props and props["total_magnetization"]["value"] is not None:
        v = props["total_magnetization"]
        lines.append(f"Total magnetization: {v['value']:.2f} {v['unit']}")
    for key, (lo, hi) in _ELASTIC_BOUNDS.items():
        v = props.get(key)
        if v and v["value"] is not None and lo < v["value"] <= hi:
            label = key.replace("_", " ").capitalize()
            lines.append(f"{label}: {v['value']:.1f} {v['unit']}")
    v = props.get("poisson_ratio")
    if v and v["value"] is not None and _POISSON_BOUNDS[0] <= v["value"] <= _POISSON_BOUNDS[1]:
        lines.append(f"Poisson ratio: {v['value']:.3f}")

    bc = entities["battery_cells"].get(mat["battery_cell"]) if mat["battery_cell"] else None
    if bc:
        ocv = bc["properties"].get("open_circuit_voltage")
        cap = bc["properties"].get("specific_capacity")
        if ocv and ocv["value"] is not None:
            lines.append(f"Open-circuit voltage: {ocv['value']:.2f} {ocv['unit']}")
        if cap and cap["value"] is not None:
            lines.append(f"Specific capacity: {cap['value']:.1f} {cap['unit']}")

    if crystal:
        sg = entities["space_groups"].get(crystal["space_group"]) if crystal["space_group"] else None
        if sg:
            lines.append(f"Space group: {sg['symbol']} (#{sg['number']})")
        cs = entities["crystal_systems"].get(crystal["crystal_system"]) if crystal["crystal_system"] else None
        if cs:
            lines.append(f"Crystal system: {cs['label']}")
        lines.append(f"Structure: {crystal['label']}")

    if variant == "full" and crystal:
        uc = entities["unit_cells"].get(crystal["unit_cell"])
        if uc:
            lines.append(f"Lattice: a={uc['a']}, b={uc['b']}, c={uc['c']} Å; "
                         f"α={uc['alpha']}, β={uc['beta']}, γ={uc['gamma']}°")
            vol = props.get("volume")
            if vol and vol["value"] is not None:
                lines.append(f"Volume: {vol['value']:.2f} {vol['unit']}")
            dens = props.get("density")
            if dens and dens["value"] is not None:
                lines.append(f"Density: {dens['value']:.2f} {dens['unit']}")
            lines.append(f"Sites: {uc['num_sites']} ({_site_element_counts(uc, entities)})")

    return "\n".join(lines)


def verbalize_crystal(crystal: dict, entities: dict, variant: str) -> str:
    material = entities["materials"].get(f"material/{crystal['material_project_id']}")
    formula = material["formula"] if material else crystal["material_project_id"]
    lines = [
        f"Crystal: {formula} ({crystal['material_project_id']})",
        "Type: crystal structure",
    ]
    if crystal["structure_family"]:
        lines.append(f"Structure family: {STRUCTURE_FAMILY_READABLE[crystal['structure_family']]}")
    sg = entities["space_groups"].get(crystal["space_group"]) if crystal["space_group"] else None
    if sg:
        lines.append(f"Space group: {sg['symbol']} (#{sg['number']})")
    cs = entities["crystal_systems"].get(crystal["crystal_system"]) if crystal["crystal_system"] else None
    if cs:
        lines.append(f"Crystal system: {cs['label']}")

    # Property-tier facts borrowed from the owning material -- crystals carry none of their own
    # (Step 7). Deliberately a SMALLER set than the material's own sentence (just enough to
    # distinguish crystals from each other for the hasStructure ranking task -- see BUILD_LOG.md
    # Step 8 -- not a full restatement of the material).
    props = material["properties"] if material else {}
    if "band_gap" in props and props["band_gap"]["value"] is not None:
        v = props["band_gap"]
        lines.append(f"Band gap: {v['value']:.2f} {v['unit']}")
    if "formation_energy_per_atom" in props and props["formation_energy_per_atom"]["value"] is not None:
        v = props["formation_energy_per_atom"]
        lines.append(f"Formation energy: {v['value']:.3f} {v['unit']}/atom")

    if variant == "full":
        uc = entities["unit_cells"].get(crystal["unit_cell"])
        if uc:
            lines.append(f"Lattice: a={uc['a']}, b={uc['b']}, c={uc['c']} Å; "
                         f"α={uc['alpha']}, β={uc['beta']}, γ={uc['gamma']}°")
            vol = props.get("volume")
            if vol and vol["value"] is not None:
                lines.append(f"Volume: {vol['value']:.2f} {vol['unit']}")
            lines.append(f"Sites: {uc['num_sites']} ({_site_element_counts(uc, entities)})")

    return "\n".join(lines)


def verbalize_element(el: dict) -> str:
    """Same for both variants -- elements carry no geometry to omit. Only the 6 fields that
    actually exist in this KG (see BUILD_LOG.md Step 8: the ontology has a hasAtomicNumber
    predicate available via its EMMO import, but the population pipeline never writes it, so
    'atomic number' -- present in ONT_ABOX_EXTENSION.md's own template -- doesn't exist in real
    data here; we don't invent it)."""
    lines = [f"Element: {el['symbol']}"]
    if el["group"] is not None:
        lines.append(f"Group: {el['group']}")
    if el["period"] is not None:
        lines.append(f"Period: {el['period']}")
    if el["electronegativity"] is not None:
        lines.append(f"Electronegativity: {el['electronegativity']}")
    if el["valence_electrons"] is not None:
        lines.append(f"Valence electrons: {el['valence_electrons']}")
    if el["atomic_mass"] is not None:
        lines.append(f"Atomic mass: {el['atomic_mass']} u")
    if el["covalent_radius"] is not None:
        lines.append(f"Covalent radius: {el['covalent_radius']} Å")
    return "\n".join(lines)


def verbalize_all(entities: dict) -> dict:
    result = {
        "elements": {k: verbalize_element(v) for k, v in entities["elements"].items()},
    }
    for variant in ("full", "no_geometry"):
        result[f"materials_{variant}"] = {
            k: verbalize_material(v, entities, variant) for k, v in entities["materials"].items()
        }
        result[f"crystals_{variant}"] = {
            k: verbalize_crystal(v, entities, variant) for k, v in entities["crystals"].items()
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/battgpt_abox/entities.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/battgpt_abox/"))
    args = ap.parse_args()

    entities = json.loads(args.entities.read_text())
    logger.info(f"Loaded {args.entities}: {entities['meta']['counts']}")

    verbalizations = verbalize_all(entities)
    for key, mapping in verbalizations.items():
        out_path = args.out_dir / f"verbalizations_{key}.json"
        out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
        lengths = [len(s) for s in mapping.values()]
        logger.info(f"Wrote {out_path} ({len(mapping)} entries, {min(lengths)}-{max(lengths)} chars)")


if __name__ == "__main__":
    main()
