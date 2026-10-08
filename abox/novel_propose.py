"""Novel-cathode pilot, step 1: propose new cathode compositions by element substitution.

Parents: cathode-role materials of our 5,000 KG (lowest-energy entry per formula, <= 0.05 eV/atom
above hull, conventional standard cell <= 40 atoms, containing a redox-active transition metal).
Children: one transition metal swapped for another that can take the same oxidation state, or
Li <-> Na. Parents with precious or toxic elements are skipped. Kept only if SMACT-valid and NOT present in Materials Project (any energy, checked
live), ranked by ionic-radius mismatch of the swap (smaller = more likely to keep the parent's
structure), at most 2 per parent. Each child keeps the parent's conventional cell (so the CrystaLLM
prompt asks for a cell consistent with the parent's space group) and the parent's space group as a
KG-analogue hint ("parent_sg" prompt).

Theoretical capacity is a rough upper bound: every alkali ion per formula unit extracted, one
electron each (n F / 3.6 M, mAh/g); real capacity also needs the transition metal to have the
redox range, which this does not check.

Run with battGPT's venv (pymatgen, smact, mp_api) and the MP key in battGPT/.env:
    ../battGPT/.venv/Scripts/python.exe abox/novel_propose.py --n 200
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "battGPT"))

from pymatgen.core import Composition, Element, Species, Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402
from smact.screening import smact_validity  # noqa: E402

TMS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Nb", "Mo"]
FARADAY_MAH = 26801.48  # F / 3.6, mAh per mol of electrons
# precious or toxic: not practical in a cathode, so parents or children containing them are skipped
EXCLUDE = {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au", "Cd", "Hg", "Tl", "Pb", "As", "Be"}


def radius(el, ox):
    try:
        return Species(el, ox).ionic_radius
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-parent", type=int, default=2)
    ap.add_argument("--max-atoms", type=int, default=40)
    a = ap.parse_args()
    ents = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000", "entities.json"), encoding="utf-8"))
    by_formula = defaultdict(list)
    for k, m in ents["materials"].items():
        if m["battery_role"] != "PositiveElectrode":
            continue
        eh = (m.get("properties") or {}).get("energy_above_hull", {}).get("value")
        if eh is None or eh > 0.05:
            continue
        by_formula[m["formula"]].append((eh, k, m))
    known = {Composition(f).reduced_formula for f in {m["formula"] for m in ents["materials"].values()}}
    children = {}
    n_parents = 0
    for formula, ms in sorted(by_formula.items()):
        eh, k, m = min(ms, key=lambda x: (x[0], x[1]))
        c = ents["crystals"][m["structure"]]
        try:
            s = Structure.from_str(c["cif"], fmt="cif")
            conv = SpacegroupAnalyzer(s, symprec=0.1).get_conventional_standard_structure()
        except Exception:
            continue
        if len(conv) > a.max_atoms:
            continue
        comp = conv.composition
        els = [str(e) for e in comp.elements]
        if EXCLUDE & set(els):
            continue
        tms_here = [e for e in els if e in TMS]
        if not tms_here:
            continue
        try:
            guesses = comp.reduced_composition.oxi_state_guesses(max_sites=-20)
        except Exception:  # pymatgen refuses very large formula units
            guesses = []
        if not guesses:
            continue
        oxi = guesses[0]
        n_parents += 1
        swaps = [(A, B) for A in tms_here for B in TMS if B != A and B not in els]
        alk = [e for e in els if e in ("Li", "Na")]
        if len(alk) == 1:
            swaps.append((alk[0], "Na" if alk[0] == "Li" else "Li"))
        for A, B in swaps:
            ox = round(oxi.get(A, 0))
            if ox == 0 or ox not in Element(B).oxidation_states:
                continue
            rA, rB = radius(A, ox), radius(B, ox)
            if not rA or not rB:
                continue
            new = Composition({(B if str(e) == A else str(e)): amt for e, amt in comp.items()})
            red = new.reduced_formula
            if red in known or red in children:
                continue
            try:
                if not smact_validity(red):
                    continue
            except Exception:
                continue
            n_alk = sum(new.reduced_composition[x] for x in ("Li", "Na") if x in new.reduced_composition)
            children[red] = {
                "formula": red, "cell": new.formula.replace(" ", ""), "n_atoms": int(new.num_atoms),
                "parent_id": k, "parent_formula": formula, "parent_sg": int(c["space_group"].split("/")[-1]),
                "parent_e_above_hull": eh, "substitution": f"{A}->{B} ({ox:+d})",
                "radius_mismatch": round(abs(rA - rB) / rA, 3),
                "theoretical_capacity_mAh_g": round(n_alk * FARADAY_MAH / new.reduced_composition.weight, 1)}
    print(f"{n_parents} parents -> {len(children)} SMACT-valid children not in our KG")

    from mp_api.client import MPRester
    from pipeline.config import PipelineConfig
    names = sorted(children)
    in_mp = set()
    with MPRester(PipelineConfig().mp_api_key, mute_progress_bars=True) as mpr:
        for i in range(0, len(names), 100):
            docs = mpr.materials.summary.search(formula=names[i:i + 100], fields=["formula_pretty"])
            in_mp |= {Composition(d.formula_pretty).reduced_formula for d in docs}
    novel = [children[n] for n in names if n not in in_mp]
    print(f"{len(in_mp)} already in Materials Project -> {len(novel)} novel")

    novel.sort(key=lambda x: (x["radius_mismatch"], x["parent_e_above_hull"], x["formula"]))
    picked, per_parent = [], defaultdict(int)
    for x in novel:
        if per_parent[x["parent_id"]] >= a.per_parent:
            continue
        per_parent[x["parent_id"]] += 1
        picked.append(x)
        if len(picked) >= a.n:
            break
    for i, x in enumerate(picked):
        x.update({"id": f"NOVEL_{i:03d}", "novel": True, "cathode_like": True, "sg_pred_top5": []})
    json.dump(picked, open(os.path.join(DATA, "novel_candidates.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {len(picked)} candidates from {len(per_parent)} parents; examples:",
          [(x["formula"], x["substitution"], x["parent_formula"], x["parent_sg"]) for x in picked[:8]])


if __name__ == "__main__":
    main()
