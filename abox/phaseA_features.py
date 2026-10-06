"""Phase A, step 1: composition-only features + structure labels (run with battGPT's venv: needs pymatgen).

Question Phase A asks: from what a NOVEL composition has before any structure exists -- its formula,
elements, plausible oxidation states -- can the crystal system / space group of its ground-state
structure be predicted better than simple baselines? (Goal: novel cathode CIFs via CrystaLLM, which
accepts a space group in its prompt.)

One example per formula: its ground state (lowest energy above hull), the structure we'd want to
generate. Nothing here uses DFT properties or structure: a novel composition has neither.

    ../battGPT/.venv/Scripts/python.exe abox/phaseA_features.py \
        --entities data/battgpt_abox_grow_5000/entities.json --out data/phaseA/examples.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element

PROPS = ("Z", "X", "atomic_radius", "row", "group", "mendeleev_no", "atomic_mass")


def element_props(el: Element) -> list[float]:
    out = []
    for p in PROPS:
        v = getattr(el, p, None)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def featurize(formula: str) -> dict:
    comp = Composition(formula)
    fr = comp.fractional_composition
    els = sorted(fr.elements, key=lambda e: e.Z)
    w = np.array([fr[e] for e in els])
    P = np.array([element_props(e) for e in els])
    stats = {}
    for j, p in enumerate(PROPS):
        col = P[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            col, wk = np.zeros(1), np.ones(1)
        else:
            col, wk = col[ok], w[ok] / w[ok].sum()
        mean = float((wk * col).sum())
        stats.update({f"{p}_mean": mean, f"{p}_min": float(col.min()), f"{p}_max": float(col.max()),
                      f"{p}_range": float(col.max() - col.min()),
                      f"{p}_std": float(np.sqrt((wk * (col - mean) ** 2).sum()))})
    # most likely oxidation-state assignment (pymatgen's composition-only guess, no structure)
    oxi = {}
    try:
        guesses = comp.oxi_state_guesses(max_sites=-20)
        if guesses:
            oxi = {str(k): float(v) for k, v in guesses[0].items()}
    except Exception:
        pass
    cations = [v for k, v in oxi.items() if v > 0 and k not in ("Li", "Na")]
    stats.update({"n_elements": len(els), "oxi_found": float(bool(oxi)),
                  "tm_oxi_mean": float(np.mean(cations)) if cations else 0.0,
                  "tm_oxi_max": float(max(cations)) if cations else 0.0,
                  "anion_oxi_min": float(min(oxi.values())) if oxi else 0.0})
    return {"element_fractions": {str(e): float(fr[e]) for e in els}, "stats": stats,
            "oxidation_states": oxi, "anonymous_formula": comp.reduced_composition.anonymized_formula}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    e = json.loads(a.entities.read_text(encoding="utf-8"))
    M, C = e["materials"], e["crystals"]
    by_formula = defaultdict(list)
    for key, m in M.items():
        eh = (m.get("properties") or {}).get("energy_above_hull", {}).get("value")
        by_formula[m["formula"]].append((eh if eh is not None else 9.9, key))
    examples = []
    for i, (formula, ms) in enumerate(sorted(by_formula.items())):
        ms.sort()
        key = ms[0][1]
        m, c = M[key], C[M[key]["structure"]]
        examples.append({
            "formula": formula, "material": key, "n_polymorphs_in_kg": len(ms),
            "battery_role": m["battery_role"],
            "label": {"crystal_system": c["crystal_system"].split("/")[-1],
                      "space_group": int(c["space_group"].split("/")[-1]),
                      "structure_family": (c["structure_family"] or "None").split("/")[-1]},
            **featurize(formula)})
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(by_formula)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(examples, indent=1), encoding="utf-8")
    print(f"wrote {len(examples)} ground-state examples to {a.out}")


if __name__ == "__main__":
    main()
