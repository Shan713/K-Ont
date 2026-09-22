"""Extractor: populated KG (battery_kg.ttl) -> entities.json.

This is the first stage of the ABox pipeline. It reads the whole KG once and writes out a plain,
structured JSON file that the next stages (verbalize.py, numeric.py) read instead of touching RDF
again. Nothing here decides what OnT will be trained on -- this stage is deliberately "get
everything out faithfully", not "get what we need". The verbalizer is what later decides which
fields go into a sentence and which stay out (see BUILD_LOG.md for why, e.g., lattice geometry
stays OUT of the material's sentence but IS extracted here -- extraction and use are different
decisions, and keeping them separate means a future variant can use fields this one doesn't).

Deviation from ONT_ABOX_EXTENSION.md worth noting: the original spec put this script at
OnT/ont/data/extract_abox.py -- inside the vendored OnT/ checkout. We deliberately did NOT do
that: OnT/ is a gitignored, third-party checkout (see BUILD_LOG.md Step 1), and our own code has
no business living inside someone else's, gitignored, repo where it would never be committed. Our
code lives in K-Ont/abox/ instead; only the *data it produces* is meant to end up next to OnT's
training scripts, later, at training time.

Also deviates on file size handling: the spec worried about streaming a 9.6M-line TTL "block-wise"
because DeepOnto/rdflib can't load a file that size. Our actual KG is 325K triples / 21MB --
rdflib.Graph().parse() loads it directly in a few seconds. If the KG grows enough that this stops
being true, that's the point to revisit streaming, not before.

Usage:
    .venv/bin/python3 -m abox.extract --ttl ../output/battgpt_kg_ont/battery_kg.ttl --out data/battgpt_abox/entities.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import rdflib
from rdflib import RDF, RDFS, Namespace, URIRef, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Namespaces & the specific EMMO/battery IRIs this KG uses (same constants as
#    battGPT/pipeline/rdf/triple_generator.py -- copied, not imported, because K-Ont is a
#    separate repo/environment from battGPT and shouldn't depend on its package layout). ──
BATTGPT = Namespace("https://w3id.org/battgpt/kg#")
KG = Namespace("https://w3id.org/battgpt/kg/")
EMMO = Namespace("https://w3id.org/emmo#")
BATTERY = Namespace("https://w3id.org/emmo/domain/battery#")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

TYPE_CHEMICAL_SUBSTANCE = EMMO.EMMO_df96cbb6_b5ee_4222_8eab_b3675df24bea
TYPE_CHEMICAL_ELEMENT = EMMO.EMMO_4f40def1_3cd7_4067_9596_541e9a5134cf
TYPE_PROPERTY = EMMO.EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba
TYPE_BATTERY_CELL = BATTERY.battery_68ed592a_7924_45d0_a108_94d6275d57f0
PRED_HAS_PROPERTY = EMMO.EMMO_e1097637_70d2_4895_973f_2396f04fa204
PRED_HAS_VALUE = EMMO.EMMO_faf79f53_749d_40b2_807c_d34244c192f4
PRED_HAS_UNIT = EMMO.EMMO_bed1d005_b04e_4a90_94cf_02bc678a8569

# Human-readable symbols for the QUDT unit IRIs this KG actually uses (checked against the real
# graph -- see BUILD_LOG.md -- not the full QUDT vocabulary, just what's here).
UNIT_SYMBOLS = {
    "http://qudt.org/vocab/unit/EV": "eV",
    "http://qudt.org/vocab/unit/GigaPA": "GPa",
    "http://qudt.org/vocab/unit/ANGSTROM3": "Å³",
    "http://qudt.org/vocab/unit/G-PER-CentiM3": "g/cm³",
    "http://qudt.org/vocab/unit/BohrMagneton": "µB",
    "http://qudt.org/vocab/unit/UNITLESS": "",
    "http://qudt.org/vocab/unit/V": "V",
    "http://qudt.org/vocab/unit/MilliA-HR-PER-GM": "mAh/g",
}


def local_key(uri: URIRef) -> str:
    """URI -> the short key we use in entities.json, e.g.
    'https://w3id.org/battgpt/kg/material/mp-10178' -> 'material/mp-10178'.
    Deliberately just strips the KG's own resource namespace, so the key is directly
    reconstructible back into the real URI (KG + key) -- no separate id scheme invented here."""
    s = str(uri)
    return s[len(str(KG)):] if s.startswith(str(KG)) else s


def label_of(g: rdflib.Graph, uri: URIRef) -> str | None:
    vals = list(g.objects(uri, RDFS.label))
    return str(vals[0]) if vals else None


def literal_or_none(g: rdflib.Graph, s: URIRef, p: URIRef):
    vals = list(g.objects(s, p))
    if not vals:
        return None
    v = vals[0]
    if isinstance(v, Literal):
        return v.toPython()
    return str(v)


def extract_properties(g: rdflib.Graph, subject: URIRef) -> dict:
    """Every quantity (band gap, formation energy, ...) hangs off `subject` via the generic
    emmo:hasProperty predicate, regardless of whether a specific battgpt:hasXxx predicate also
    points at the same node. Reading the generic predicate once, and naming each property from
    the LAST PATH SEGMENT of its own URI (".../property/mp-10178/band_gap" -> "band_gap", which is
    exactly the name battGPT's PropertyData/URIScheme gave it when building the KG), means we
    don't have to hand-enumerate every possible battgpt:hasXxx predicate here and keep that list
    in sync with the ontology by hand."""
    props = {}
    for prop_uri in g.objects(subject, PRED_HAS_PROPERTY):
        name = str(prop_uri).rsplit("/", 1)[-1]
        value = literal_or_none(g, prop_uri, PRED_HAS_VALUE)
        unit_uri = next(g.objects(prop_uri, PRED_HAS_UNIT), None)
        unit_symbol = UNIT_SYMBOLS.get(str(unit_uri), str(unit_uri)) if unit_uri else None
        if value is not None:
            props[name] = {"value": value, "unit": unit_symbol}
    return props


def extract(ttl_path: Path) -> dict:
    logger.info(f"Parsing {ttl_path} ...")
    g = rdflib.Graph()
    g.parse(str(ttl_path), format="turtle")
    logger.info(f"Parsed {len(g):,} triples.")

    out = {
        "materials": {}, "crystals": {}, "unit_cells": {}, "sites": {}, "species": {},
        "elements": {}, "space_groups": {}, "crystal_systems": {}, "battery_cells": {},
    }

    # ── Elements (static chemistry -- extracted once, referenced by species) ──
    for uri in g.subjects(RDF.type, TYPE_CHEMICAL_ELEMENT):
        key = local_key(uri)
        out["elements"][key] = {
            "symbol": label_of(g, uri),
            "atomic_mass": literal_or_none(g, uri, BATTGPT.hasAtomicMass),
            "covalent_radius": literal_or_none(g, uri, BATTGPT.hasCovalentRadius),
            "electronegativity": literal_or_none(g, uri, BATTGPT.hasElectronegativity),
            "group": literal_or_none(g, uri, BATTGPT.hasGroup),
            "period": literal_or_none(g, uri, BATTGPT.hasPeriod),
            "valence_electrons": literal_or_none(g, uri, BATTGPT.hasValenceElectrons),
        }

    # ── Species (element + oxidation state, referenced by sites) ──
    for uri in g.subjects(RDF.type, BATTGPT.Species):
        key = local_key(uri)
        elem_uri = next(g.objects(uri, BATTGPT.hasElement), None)
        out["species"][key] = {
            "label": label_of(g, uri),
            "element": local_key(elem_uri) if elem_uri else None,
        }

    # ── Space groups & crystal systems (small closed vocabularies) ──
    for uri in g.subjects(RDF.type, BATTGPT.SpaceGroup):
        key = local_key(uri)
        out["space_groups"][key] = {
            "label": label_of(g, uri),
            "number": literal_or_none(g, uri, BATTGPT.hasSpaceGroupNumber),
            "symbol": literal_or_none(g, uri, BATTGPT.hasSymmetrySymbol),
        }
    for uri in g.subjects(RDF.type, BATTGPT.CrystalSystem):
        out["crystal_systems"][local_key(uri)] = {"label": label_of(g, uri)}

    # ── Sites (+ their bonds, read off CrystalBond individuals) ──
    for uri in g.subjects(RDF.type, BATTGPT.Site):
        key = local_key(uri)
        species_uri = next(g.objects(uri, BATTGPT.hasSpecies), None)
        geom_uri = next(g.objects(uri, BATTGPT.hasCoordinationGeometry), None)
        bonds = []
        for bond_uri in g.objects(uri, BATTGPT.hasBond):
            target = next(g.objects(bond_uri, BATTGPT.hasTargetSite), None)
            bonds.append({
                "target_site": local_key(target) if target else None,
                "distance_angstrom": literal_or_none(g, bond_uri, BATTGPT.hasBondDistance),
                "coordination_method": literal_or_none(g, bond_uri, BATTGPT.hasCoordinationMethod),
            })
        out["sites"][key] = {
            "label": label_of(g, uri),
            "species": local_key(species_uri) if species_uri else None,
            "fractional_x": literal_or_none(g, uri, BATTGPT.hasFractionalX),
            "fractional_y": literal_or_none(g, uri, BATTGPT.hasFractionalY),
            "fractional_z": literal_or_none(g, uri, BATTGPT.hasFractionalZ),
            # CoordinationGeometry individuals are labelled e.g. "TetrahedralGeometryIndividual"
            # in the ontology itself (same "Individual" suffix convention as StructureFamily and
            # BatteryRole individuals) -- strip it to a clean "Tetrahedral", same as those two.
            "coordination_geometry": label_of(g, geom_uri).rsplit("GeometryIndividual", 1)[0] if geom_uri else None,
            "bonds": bonds,
        }

    # ── Unit cells ──
    for uri in g.subjects(RDF.type, BATTGPT.UnitCell):
        key = local_key(uri)
        sites = [local_key(s) for s in g.objects(uri, BATTGPT.hasSite)]
        out["unit_cells"][key] = {
            "label": label_of(g, uri),
            "a": literal_or_none(g, uri, BATTGPT.hasLatticeA),
            "b": literal_or_none(g, uri, BATTGPT.hasLatticeB),
            "c": literal_or_none(g, uri, BATTGPT.hasLatticeC),
            "alpha": literal_or_none(g, uri, BATTGPT.hasAlpha),
            "beta": literal_or_none(g, uri, BATTGPT.hasBeta),
            "gamma": literal_or_none(g, uri, BATTGPT.hasGamma),
            "sites": sites,
            "num_sites": len(sites),
        }

    # ── Crystals ──
    for uri in g.subjects(RDF.type, BATTGPT.CrystalStructure):
        key = local_key(uri)
        sg = next(g.objects(uri, BATTGPT.hasSpaceGroup), None)
        cs = next(g.objects(uri, BATTGPT.hasCrystalSystem), None)
        uc = next(g.objects(uri, BATTGPT.hasUnitCell), None)
        family = next(g.objects(uri, BATTGPT.hasStructureFamily), None)
        # rdfs:comment is used for two unrelated purposes in this KG (structure-family evidence
        # and, on materials, battery-role evidence) -- filter to the one written with our
        # "StructureFamily Evidence: ..." prefix (see triple_generator.py), not just "first
        # comment", so we don't grab the wrong one if a node ever carries both.
        family_evidence = None
        for c in g.objects(uri, RDFS.comment):
            if str(c).startswith("StructureFamily Evidence:"):
                family_evidence = str(c)[len("StructureFamily Evidence: "):]
        out["crystals"][key] = {
            "label": label_of(g, uri),
            "material_project_id": literal_or_none(g, uri, BATTGPT.hasMaterialProjectId),
            "cif": literal_or_none(g, uri, BATTGPT.hasCif),
            "space_group": local_key(sg) if sg else None,
            "crystal_system": local_key(cs) if cs else None,
            "unit_cell": local_key(uc) if uc else None,
            "structure_family": local_key(family).rsplit("Individual", 1)[0].rsplit("#", 1)[-1] if family else None,
            "structure_family_evidence": family_evidence,
            "properties": extract_properties(g, uri),
        }

    # ── Battery cells (OCV / specific capacity -- linked to a material by dcterms:relation) ──
    for uri in g.subjects(RDF.type, TYPE_BATTERY_CELL):
        key = local_key(uri)
        material = next(g.objects(uri, DCTERMS.relation), None)
        out["battery_cells"][key] = {
            "label": label_of(g, uri),
            "material": local_key(material) if material else None,
            "properties": extract_properties(g, uri),
        }
    # index battery cells by material so materials can look theirs up directly
    cell_by_material = {v["material"]: k for k, v in out["battery_cells"].items() if v["material"]}

    # ── Materials (the main entity -- everything else exists to describe one of these) ──
    for uri in g.subjects(RDF.type, TYPE_CHEMICAL_SUBSTANCE):
        key = local_key(uri)
        structure = next(g.objects(uri, BATTGPT.hasStructure), None)
        role = next(g.objects(uri, BATTGPT.belongsToElectrode), None)
        role_evidence = None
        for c in g.objects(uri, RDFS.comment):
            if str(c).startswith("BattINFO Role Evidence:"):
                role_evidence = str(c)[len("BattINFO Role Evidence: "):]
        out["materials"][key] = {
            "label": label_of(g, uri),
            "formula": literal_or_none(g, uri, BATTGPT.hasFormula),
            "chemsys": literal_or_none(g, uri, BATTGPT.hasChemsys),
            "material_project_id": literal_or_none(g, uri, BATTGPT.hasMaterialProjectId),
            "is_metal": literal_or_none(g, uri, BATTGPT.isMetal),
            "is_stable": literal_or_none(g, uri, BATTGPT.isStable),
            "is_gap_direct": literal_or_none(g, uri, BATTGPT.isGapDirect),
            "is_theoretical": literal_or_none(g, uri, BATTGPT.isTheoretical),
            "structure": local_key(structure) if structure else None,
            "battery_role": local_key(role).rsplit("RoleIndividual", 1)[0].rsplit("#", 1)[-1] if role else None,
            "battery_role_evidence": role_evidence,
            "battery_cell": cell_by_material.get(key),
            "properties": extract_properties(g, uri),
        }

    counts = {k: len(v) for k, v in out.items()}
    out["meta"] = {
        "source_ttl": str(ttl_path),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_triple_count": len(g),
        "counts": counts,
    }
    logger.info(f"Extracted: {counts}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", type=Path, required=True, help="Path to the populated KG's battery_kg.ttl")
    ap.add_argument("--out", type=Path, default=Path("data/battgpt_abox/entities.json"))
    args = ap.parse_args()

    entities = extract(args.ttl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entities, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
