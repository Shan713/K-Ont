"""Task D / D2: MatterGen fine-tuning data from the KG (data/crystallm/kg_training_source.json.gz).

Keeps materials with at most 20 atoms in the primitive standard cell (MatterGen's training range),
ordered, and made only of elements MatterGen supports (no noble gases, Z > 84, Tc, Pm). Uses the KG's
own train / val splits as given and writes only those; `test` and `excluded` are counted for the
statistics but never written, so nothing can train on them.

Output (data/crystallm/taskD/, gitignored): train.csv, val.csv with the columns MatterGen's
`csv-to-dataset` reads (material_id, cif, battery_role) plus structure_family and a few KG fields.
Statistics: data/crystallm/taskD_data_stats.json.

    .venv-crystallm/Scripts/python.exe -I abox/taskd_prepare_data.py
"""
import argparse
import collections
import gzip
import json
import os
import warnings

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
UNSUPPORTED_SYMBOLS = {"Tc", "Pm"}  # MatterGen model card: radioactive elements it excludes
WRITE_SPLITS = ("train", "val")


def element_ok(el):
    return not (el.Z > 84 or el.is_noble_gas or el.symbol in UNSUPPORTED_SYMBOLS)


def primitive_cell(rec, max_atoms):
    """Return (primitive standard structure or None, drop reason or None)."""
    try:
        s = Structure.from_str(rec["cif"], fmt="cif")
    except Exception:
        return None, "cif_parse_failed"
    if not s.is_ordered:
        return None, "disordered"
    if not all(element_ok(e) for e in s.composition.elements):
        return None, "unsupported_element"
    try:
        prim = SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()
    except Exception:
        return None, "symmetry_failed"
    if len(prim) > max_atoms:
        return None, f"gt{max_atoms}_atoms_primitive"
    return prim, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-atoms", type=int, default=20)
    ap.add_argument("--source", default=os.path.join(DATA, "kg_training_source.json.gz"))
    ap.add_argument("--out-dir", default=os.path.join(DATA, "taskD"))
    ap.add_argument("--stats", default=os.path.join(DATA, "taskD_data_stats.json"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    recs = json.load(gzip.open(a.source, "rt", encoding="utf-8"))
    rows = collections.defaultdict(list)
    st = collections.defaultdict(lambda: {"total": 0, "kept": 0, "dropped": collections.Counter(),
                                          "battery_role_kept": collections.Counter(),
                                          "structure_family_kept": collections.Counter(),
                                          "n_atoms_kept": collections.Counter(),
                                          "role_labelled_total": 0, "family_labelled_total": 0})
    for r in recs:
        sp = r["split"]
        d = st[sp]
        d["total"] += 1
        d["role_labelled_total"] += bool(r.get("battery_role"))
        d["family_labelled_total"] += bool(r.get("structure_family"))
        prim, why = primitive_cell(r, a.max_atoms)
        if why:
            d["dropped"][why] += 1
            continue
        role = r.get("battery_role") or "none"
        fam = r.get("structure_family") or "none"
        d["kept"] += 1
        d["battery_role_kept"][role] += 1
        d["structure_family_kept"][fam] += 1
        d["n_atoms_kept"][len(prim)] += 1
        if sp in WRITE_SPLITS:
            rows[sp].append({"material_id": r["id"], "formula_pretty": r["formula"], "battery_role": role,
                             "structure_family": fam, "cif": str(CifWriter(prim)), "nsites": len(prim),
                             "energy_above_hull": r.get("e_above_hull"), "space_group": r.get("space_group")})
    for sp in WRITE_SPLITS:
        pd.DataFrame(rows[sp]).to_csv(os.path.join(a.out_dir, f"{sp}.csv"), index=False)
    out = {"max_atoms": a.max_atoms, "cell": "primitive standard (SpacegroupAnalyzer symprec 0.1)",
           "written_splits": list(WRITE_SPLITS), "never_written": ["test", "excluded"],
           "splits": {sp: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in d.items()}
                      for sp, d in st.items()}}
    json.dump(out, open(a.stats, "w"), indent=1, sort_keys=True)
    for sp in ("train", "val", "test", "excluded"):
        d = st[sp]
        print(f"{sp:9s} total {d['total']:5d}  kept {d['kept']:5d}  dropped {dict(d['dropped'])}")
        print(f"          battery_role kept {dict(d['battery_role_kept'])}")
        print(f"          structure_family kept {dict(d['structure_family_kept'])}")


if __name__ == "__main__":
    main()
