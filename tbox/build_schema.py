"""Builds the tiny TBox schema OWL file (ONT_ABOX_EXTENSION.md Phase 3) that OnT's own
prepare_ontology_data() reads via DeepOnto -- a small, self-contained set of class/property
declarations, NOT the full battgpt.ttl (which owl:imports EMMO/BattINFO/etc., ~60k+ axioms we do
not want DeepOnto walking -- see the "why no owl:imports" note below).

Reuses the REAL IRIs the populated KG already types its individuals with (checked directly against
BattGpt-Ontology/battgpt.ttl and pipeline/rdf/triple_generator.py, not re-derived), so a class name
here and an rdf:type in the ABox are provably the same concept, not just similarly-named. Every
class/property is declared bare (rdf:type owl:Class / owl:ObjectProperty + rdfs:label), no
owl:imports at all.

Why no owl:imports, precisely: OnT's own data-prep code (OnT/ont/data/prepare.py,
ELNormalizedData.create_dataset) does `for ont_in_closure in ont.owl_onto.getImportsClosure():
axioms.extend(...)` -- it walks every axiom in the FULL imports closure, not just the file we hand
it. An owl:imports of even one of battgpt.ttl's own imports (EMMO alone is tens of thousands of
axioms) would make DeepOnto verbalise and OnT try to train on all of it. There is no scope
argument that skips this -- the only way to keep the TBox tiny is to not import anything, which is
exactly the design doc's own instruction (SS13 pitfall #1).

Two axiom shapes end up in the file, and ONLY these two are ever processed by prepare.py's
create_dataset (axiom_type == "SubClassOf" or "EquivalentClasses" -- everything else, including
plain rdfs:domain/rdfs:range declarations, is silently ignored by that code):

  1. Plain class hierarchy: `A rdfs:subClassOf B .`               -> nf1 ("A is a B")
  2. Existential restriction: `A rdfs:subClassOf [ owl:onProperty r ; owl:someValuesFrom B ] .`
                                                                    -> nf3 ("A has some r that is B")

Synthesized existential restrictions (#2) for hasStructure / hasStructureFamily / belongsToElectrode
-- battgpt.ttl only has these as plain rdfs:domain/rdfs:range, which produces NOTHING for OnT
(domain/range are ObjectPropertyDomain/Range axioms, not SubClassOf -- prepare.py never looks at
them). Without adding the restriction explicitly, none of these three relations would produce a
single training row, silently.

One deliberate simplification, not an oversight: `belongsToElectrode`'s REAL range in battgpt.ttl
is `owl:unionOf(NegativeElectrodeRole, PositiveElectrodeRole)` -- a complex class expression.
prepare.py's nf3 handling explicitly requires the someValuesFrom filler to be a plain named class
(`if parent["class"]["type"] != "IRI": return  # Skip if filler is complex`), so using the real
union would produce a restriction DeepOnto verbalises fine but prepare.py then silently drops.
We use `battgpt:BatteryRole` (the atomic ancestor of both) as the filler instead -- a real class
in the file, correctly broader than the true range (also covers Electrolyte/Separator roles,
which belongsToElectrode can never actually point at), but the only way to get a training row out
of the relation at all given this constraint. Documented here and in BUILD_LOG.md, not hidden.

Usage:
    .venv/bin/python3 -m tbox.build_schema --out data/battgpt_tbox.owl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rdflib
from rdflib import RDF, RDFS, OWL, Namespace, URIRef, BNode, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATTGPT = Namespace("https://w3id.org/battgpt/kg#")
EMMO = Namespace("https://w3id.org/emmo#")

# (IRI, local label) -- the two EMMO classes our ABox individuals are actually typed with, but
# which carry no rdfs:label of their own inside battgpt.ttl (their labels live in the EMMO import
# we're deliberately not pulling in) -- so we give them one here, matching exactly what
# verbalize.py already calls them ("substance", not the full EMMO term "ChemicalSubstance").
EMMO_ANCHOR_CLASSES = {
    EMMO.EMMO_df96cbb6_b5ee_4222_8eab_b3675df24bea: "substance",
    EMMO.EMMO_4f40def1_3cd7_4067_9596_541e9a5134cf: "chemical element",
}


def camel_case_to_spaced(phrase: str) -> str:
    """"LayeredOxideStructure" -> "layered oxide structure", "NASICONStructure" -> "NASICON
    structure" (preserves all-caps acronyms instead of mangling them -- unlike a naive
    `.capitalize()`, the bug we already hit once in abox/rows.py). Copied verbatim from OnT's own
    `OnT/ont/data/prepare.py::camel_case_to_spaced` rather than imported: importing anything under
    `ont.*` triggers `ont/__init__.py`'s eager import chain (ont.model -> ont.hit -> torch +
    geoopt + sentence_transformers), which this pure-rdflib schema-building script has no other
    reason to need. Kept identical to the original so it produces the exact same text OnT's own
    prepare.py would fall back to -- verified against a battgpt-specific set of names, see
    BUILD_LOG.md Step 12."""
    import re
    phrase = phrase.split("#")[-1]
    segments = re.findall(r"[A-Z]+|[^A-Z]+", phrase)
    second_phrase = ""
    numbers_romain = {"II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
    for segment in segments:
        if segment[-1].isupper():
            if len(segment) == 1:
                segment = segment.lower()
            elif len(segments) > 1 and segment not in numbers_romain:
                segment = segment[:-1] + " " + segment[-1].lower()
        else:
            segment = segment + " "
        second_phrase += segment
    return second_phrase.strip()
# Classes declared directly in battgpt.ttl -- reuses the real IRI, our own spaced label (see
# camel_case_to_spaced below).
BATTGPT_BARE_CLASSES = ["CrystalStructure", "SpaceGroup", "CrystalSystem"]

STRUCTURE_FAMILY_HIERARCHY = {
    # child -> parent, exactly as declared in battgpt.ttl (verified via rdflib query, not by hand)
    "StructureFamily": None,  # its own real parent is an EMMO Property class we're not importing;
                              # root it locally instead of dangling a reference to an undeclared class
    "OxideStructureFamily": "StructureFamily",
    "PolyanionStructureFamily": "StructureFamily",
    "SulfideStructureFamily": "StructureFamily",
    "LayeredOxideStructure": "OxideStructureFamily",
    "SpinelStructure": "OxideStructureFamily",
    "RockSaltStructure": "OxideStructureFamily",
    "GarnetStructure": "OxideStructureFamily",
    "PerovskiteStructure": "OxideStructureFamily",
    "OlivineStructure": "PolyanionStructureFamily",
    "NASICONStructure": "PolyanionStructureFamily",
    "LGPSTypeStructure": "SulfideStructureFamily",
    "ArgyroditeStructure": "SulfideStructureFamily",
}
BATTERY_ROLE_HIERARCHY = {
    "BatteryRole": None,  # same reasoning: root locally, real parent is an unimported EMMO class
    "PositiveElectrodeRole": "BatteryRole",
    "NegativeElectrodeRole": "BatteryRole",
    "ElectrolyteRole": "BatteryRole",
    "SeparatorRole": "BatteryRole",
}
# (label, object_property_local_name, domain_iri, range_iri) for the 3 synthesized existential
# restrictions. Domain/range copied from battgpt.ttl's real rdfs:domain/rdfs:range (verified by
# query), except belongsToElectrode's range -- see module docstring.
EXISTENTIAL_RESTRICTIONS = [
    ("hasStructure", EMMO.EMMO_df96cbb6_b5ee_4222_8eab_b3675df24bea, BATTGPT.CrystalStructure),
    ("hasStructureFamily", BATTGPT.CrystalStructure, BATTGPT.StructureFamily),
    ("belongsToElectrode", EMMO.EMMO_df96cbb6_b5ee_4222_8eab_b3675df24bea, BATTGPT.BatteryRole),
]


def _resolve(name: str) -> URIRef:
    """Local battgpt: class name -> its real URI. Class names here are always battgpt: local
    names (the two EMMO anchor classes are referenced by full IRI directly, not through this)."""
    return BATTGPT[name]


def build_schema() -> rdflib.Graph:
    g = rdflib.Graph()
    g.bind("battgpt", BATTGPT)
    g.bind("emmo", EMMO)
    g.bind("owl", OWL)

    # Ontology header -- deliberately NO owl:imports (see module docstring).
    onto_iri = URIRef("https://w3id.org/battgpt/kg/tbox-schema")
    g.add((onto_iri, RDF.type, OWL.Ontology))
    g.add((onto_iri, RDFS.comment, Literal(
        "Minimal TBox schema extracted from BattGpt-Ontology/battgpt.ttl for OnT training data "
        "prep (K-Ont/tbox/build_schema.py). Deliberately carries no owl:imports -- see that "
        "script's docstring for why. Not a substitute for the real ontology; do not use this for "
        "anything except generating OnT training rows.", lang="en")))

    # ── The two EMMO anchor classes (substance, chemical element) ──
    for iri, lbl in EMMO_ANCHOR_CLASSES.items():
        g.add((iri, RDF.type, OWL.Class))
        g.add((iri, RDFS.label, Literal(lbl, lang="en")))

    # ── Bare battgpt: classes ──
    for name in BATTGPT_BARE_CLASSES:
        uri = _resolve(name)
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.label, Literal(camel_case_to_spaced(name), lang="en")))

    # ── StructureFamily + BatteryRole hierarchies ──
    for hierarchy in (STRUCTURE_FAMILY_HIERARCHY, BATTERY_ROLE_HIERARCHY):
        for name, parent in hierarchy.items():
            uri = _resolve(name)
            g.add((uri, RDF.type, OWL.Class))
            g.add((uri, RDFS.label, Literal(camel_case_to_spaced(name), lang="en")))
            if parent:
                g.add((uri, RDFS.subClassOf, _resolve(parent)))

    # ── Object properties + synthesized existential restrictions ──
    for prop_name, domain, range_ in EXISTENTIAL_RESTRICTIONS:
        prop_uri = _resolve(prop_name)
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(camel_case_to_spaced(prop_name), lang="en")))
        restriction = BNode()
        g.add((restriction, RDF.type, OWL.Restriction))
        g.add((restriction, OWL.onProperty, prop_uri))
        g.add((restriction, OWL.someValuesFrom, range_))
        g.add((domain, RDFS.subClassOf, restriction))

    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/battgpt_tbox.owl"))
    args = ap.parse_args()

    g = build_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(args.out), format="xml")

    n_classes = len(set(g.subjects(RDF.type, OWL.Class)))
    n_props = len(set(g.subjects(RDF.type, OWL.ObjectProperty)))
    n_subclass = len(list(g.subject_objects(RDFS.subClassOf)))
    logger.info(f"Wrote {args.out}: {len(g)} triples, {n_classes} classes, {n_props} object "
               f"properties, {n_subclass} subClassOf axioms (incl. the 3 existential ones)")
    imports = list(g.objects(None, OWL.imports))
    assert not imports, f"BUG: schema has owl:imports ({imports}) -- must never happen, see docstring"
    logger.info("Confirmed: zero owl:imports in the generated schema.")


if __name__ == "__main__":
    main()
