"""Phase C sanity set: well-known battery materials CrystaLLM most likely saw in training.

If the pipeline (prompting, post-processing, StructureMatcher) is sound, CrystaLLM should reproduce
structures it was trained on often. A near-zero match rate here would point at our pipeline, not at
how hard unseen cathodes are. References come from our 5,000 KG (the Materials Project structure of
the lowest-energy entry per formula); any formula present in CrystaLLM's held-out test set is
excluded, so every reference is from the data CrystaLLM could have trained on.

Run with the CrystaLLM venv (needs pymatgen):
    .venv-crystallm/Scripts/python.exe abox/phaseC_make_sanity.py
"""
import glob
import json
import os
import re

from pymatgen.core import Composition, Structure

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
FAMOUS = ["LiCoO2", "LiFePO4", "LiMn2O4", "LiNiO2", "LiMnPO4", "LiCoPO4", "Li4Ti5O12", "LiTi2O4",
          "Li2MnO3", "LiVO2", "LiCrO2", "LiFeO2", "LiMnO2", "NaCoO2", "NaFePO4", "NaCrO2", "NaFeO2",
          "NaMnO2", "Na3V2(PO4)3", "LiNbO3", "Li2FeSiO4", "LiVPO4F", "Li3V2(PO4)3", "NaVPO4F", "Li2S"]


def key(formula):
    return tuple(sorted(Composition(formula).reduced_composition.as_dict().items()))


def main():
    ents = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000", "entities.json"), encoding="utf-8"))
    test_keys = set()
    for f in glob.glob(os.path.join(DATA, "test_cifs", "*.cif")):
        cell = re.search(r"^data_(\S+)", open(f, encoding="utf-8").read(), re.M).group(1)
        test_keys.add(key(cell))
    out, skipped = [], []
    for formula in FAMOUS:
        mats = [(m.get("properties", {}).get("energy_above_hull", {}).get("value", 9.9), k, m)
                for k, m in ents["materials"].items() if m["formula"] == formula]
        if not mats:
            skipped.append((formula, "not in the 5,000 KG")); continue
        if key(formula) in test_keys:
            skipped.append((formula, "in CrystaLLM's test set")); continue
        _, k, m = min(mats, key=lambda x: (x[0], x[1]))
        c = ents["crystals"][m["structure"]]
        s = Structure.from_str(c["cif"], fmt="cif")
        if len(s) > 60:  # very large cells are slow and near CrystaLLM's context limit
            skipped.append((formula, f"{len(s)} atoms in the cell")); continue
        out.append({"id": "KG_" + k.split("/")[-1], "formula": formula,
                    "cell": s.composition.formula.replace(" ", ""),
                    "sg": int(c["space_group"].split("/")[-1]), "cathode_like": True,
                    "sg_pred_top5": [], "n_atoms": len(s), "cif": c["cif"]})
    json.dump(out, open(os.path.join(DATA, "phaseC_sanity_candidates.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(out)} sanity materials:", [(x["formula"], x["cell"], x["sg"]) for x in out])
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
