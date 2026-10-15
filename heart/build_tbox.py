"""Heart-disease K-Ont check, step 1: the TBox-only OWL from the heart KG.

Removes every individual and assertion (patients, value individuals) and the label property hasDiagnosis
from heart_disease_knowledge_graph.owl, keeping classes, subclass/disjointness axioms and the 5 object
properties (gender, chest pain, resting ECG, ST slope, thallium scan). This is what standard OnT trains on.

    python heart/build_tbox.py --kg <heart_disease_knowledge_graph.owl> --out data/heart/heart_tbox.owl
"""
import argparse
import re

DROP_BLOCKS = ("ClassAssertion", "ObjectPropertyAssertion", "DataPropertyAssertion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = open(a.kg, encoding="utf-8").read()
    for b in DROP_BLOCKS:
        s = re.sub(rf"\s*<{b}>.*?</{b}>", "", s, flags=re.S)
    s = re.sub(r"\s*<Declaration>\s*<NamedIndividual [^>]*/>\s*</Declaration>", "", s)
    # the label: no declaration, domain or range axiom of hasDiagnosis
    s = re.sub(r"\s*<(\w+)>\s*<ObjectProperty IRI=\"#hasDiagnosis\"/>.*?</\1>", "", s, flags=re.S)
    open(a.out, "w", encoding="utf-8").write(s)
    print("individuals left:", s.count("NamedIndividual"), "| hasDiagnosis left:", s.count("hasDiagnosis"),
          "| classes:", len(set(re.findall(r'<Class IRI="#(\w+)"', s))),
          "| object properties:", sorted(set(re.findall(r'<ObjectProperty IRI="#(\w+)"', s))))


if __name__ == "__main__":
    main()
