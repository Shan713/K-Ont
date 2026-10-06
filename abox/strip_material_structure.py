"""Step 23: material sentences that do not restate their crystal.

Step 22 showed hasStructure (and polymorph identification) is solved only by matching the lines a
material sentence copies from its crystal -- space group, crystal system, lattice, volume, sites.
This writes a verbalization directory where every material sentence drops those lines (the same
STRUCT_LINES phase5_checks.py check F strips), so the only route from a material to its crystal is
through the material's properties. Crystal and element sentences are copied unchanged, and so is
entities.json, so abox/rows.py and phase5_checks.py work on the new directory as on any other.

Density is dropped too: with the formula it encodes volume per formula unit.

    python -m abox.strip_material_structure --src data/battgpt_abox_grow_1000 --out data/battgpt_abox_grow_1000_nostruct
"""
import argparse
import json
import shutil
from pathlib import Path

from .phase5_checks import STRUCT_LINES


def strip(sentence: str) -> str:
    return "\n".join(l for l in sentence.split("\n") if not l.startswith(STRUCT_LINES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.src.glob("*.json")):
        if f.name.startswith("verbalizations_materials_"):
            v = json.loads(f.read_text(encoding="utf-8"))
            out = {k: strip(s) for k, s in v.items()}
            (a.out / f.name).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            lens = sorted(len(s) for s in out.values())
            print(f"{f.name}: {len(out)} sentences, {lens[0]}-{lens[-1]} chars (was "
                  f"{min(map(len, v.values()))}-{max(map(len, v.values()))})")
        else:
            shutil.copy2(f, a.out / f.name)


if __name__ == "__main__":
    main()
