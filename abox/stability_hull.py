"""Stability screening: CHGNet energy placed on the Materials Project phase diagram.

  e_hull(predicted) = energy above the MP convex hull of a structure whose energy comes from CHGNet
                      (relaxed; CHGNet's raw energy is already on MP's corrected scale), against MP's own
                      corrected entries for the same chemical system.

Two stages, two venvs (the MP client lives in battGPT's venv, CHGNet in the CrystaLLM venv):
  fetch    (battGPT venv, needs the MP key) -> data/crystallm/mp_entries_<tag>.json
  compute  (CrystaLLM venv)                 -> data/crystallm/stability_<tag>.json

Sets:
  calib    60 known cathodes from the 5,000 KG (seeded random, <= 40 atoms): predicted vs MP's real
           e_above_hull -- how far off is this screening on materials whose answer is known?
  novel    the novel shortlist's best structures (data/crystallm/novel_best/<id>.cif)

    ../battGPT/.venv/Scripts/python.exe abox/stability_hull.py fetch --set calib
    .venv-crystallm/Scripts/python.exe abox/stability_hull.py compute --set calib
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")


def calib_set(n=60, seed=7):
    e = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000", "entities.json"), encoding="utf-8"))
    from pymatgen.core import Structure
    cands = []
    for k, m in sorted(e["materials"].items()):
        if m["battery_role"] != "PositiveElectrode":
            continue
        eh = (m.get("properties") or {}).get("energy_above_hull", {}).get("value")
        if eh is None:
            continue
        cands.append((k, m, eh))
    random.Random(seed).shuffle(cands)
    out = []
    for k, m, eh in cands:
        cif = e["crystals"][m["structure"]]["cif"]
        if len(Structure.from_str(cif, fmt="cif")) > 40:
            continue
        out.append({"id": k.split("/")[-1], "formula": m["formula"], "mp_e_above_hull": eh, "cif": cif})
        if len(out) >= n:
            break
    return out


def novel_set():
    a = json.load(open(os.path.join(DATA, "novel_steer_analysis.json")))
    out = []
    for s in a["shortlist"]:
        p = os.path.join(DATA, "novel_best", s["id"] + ".cif")
        if os.path.exists(p):
            out.append({"id": s["id"], "formula": s["formula"], "cif": open(p).read()})
    return out


def strkeys(o):
    """JSON needs string keys; some MP entry dicts are keyed by Element objects."""
    if isinstance(o, dict):
        return {str(k): strkeys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strkeys(v) for v in o]
    return o


def fetch(tag):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "battGPT"))
    from mp_api.client import MPRester
    from pipeline.config import PipelineConfig
    from monty.json import jsanitize
    from pymatgen.core import Composition
    items = calib_set() if tag == "calib" else novel_set()
    systems = sorted({"-".join(sorted(str(el) for el in Composition(x["formula"]).elements)) for x in items})
    entries = {}
    with MPRester(PipelineConfig().mp_api_key, mute_progress_bars=True) as mpr:
        for i, cs in enumerate(systems):
            for attempt in range(4):  # the MP API drops connections now and then
                try:
                    ents = mpr.get_entries_in_chemsys(cs.split("-"), additional_criteria={"thermo_types": ["GGA_GGA+U"]})
                    break
                except Exception as ex:
                    if attempt == 3:
                        raise
                    print(f"  {cs}: retry after {ex.__class__.__name__}", flush=True)
                    time.sleep(10 * (attempt + 1))
            # composition + MP2020-corrected energy is all a phase diagram needs (entry.as_dict() itself
            # fails in this pymatgen version on Element-keyed data)
            entries[cs] = [{"composition": {str(el): float(a) for el, a in en.composition.items()},
                            "energy": float(en.energy), "entry_id": str(en.entry_id)} for en in ents]
            print(f"{i + 1}/{len(systems)} {cs}: {len(ents)} entries", flush=True)
    json.dump({"items": items, "entries": entries}, open(os.path.join(DATA, f"mp_entries_{tag}.json"), "w"))


def compute(tag, threads):
    import torch
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.core import Structure
    if threads:
        torch.set_num_threads(threads)
    d = json.load(open(os.path.join(DATA, f"mp_entries_{tag}.json")))
    model = CHGNet.load()
    opt = StructOptimizer(model=model, use_device="cpu")
    out = []
    for i, x in enumerate(d["items"]):
        s = Structure.from_str(x["cif"], fmt="cif")
        cs = "-".join(sorted(str(el) for el in s.composition.elements))
        mp_entries = [PDEntry(Composition(en["composition"]), en["energy"], name=en["entry_id"])
                      for en in d["entries"].get(cs, [])]
        try:
            r = opt.relax(s, fmax=0.1, steps=300, verbose=False)
            final = r["final_structure"]
            e_atom = float(model.predict_structure(final)["e"])
        except Exception as ex:
            out.append({**{k: v for k, v in x.items() if k != "cif"}, "error": str(ex)[:150]}); continue
        # CHGNet is trained on MPtrj energies that already include MP2020 corrections: its raw energy
        # is directly comparable with MP's corrected entries (checked: median CHGNet - MP = +0.026
        # eV/atom on 12 known cathodes). Applying MP2020 again double-corrects and puts everything on
        # the hull (the first calibration run: 58/60 predicted 0.0).
        entry = PDEntry(final.composition, e_atom * len(final), name="chgnet")
        row = {k: v for k, v in x.items() if k != "cif"}
        if not mp_entries:
            row["error"] = "no MP reference phases"
        else:
            mine = entry
            pd = PhaseDiagram(mp_entries + [mine])
            row["e_above_hull_pred"] = round(float(pd.get_e_above_hull(mine)), 4)
            row["chgnet_e_per_atom"] = round(e_atom, 4)
        out.append(row)
        print(f"{i + 1}/{len(d['items'])} {x['formula']}: pred {row.get('e_above_hull_pred')}"
              + (f" | MP {x['mp_e_above_hull']}" if "mp_e_above_hull" in x else ""), flush=True)
    res = {"items": out}
    pairs = [(r["e_above_hull_pred"], r["mp_e_above_hull"]) for r in out if "e_above_hull_pred" in r and "mp_e_above_hull" in r]
    if pairs:
        err = [p - m for p, m in pairs]
        res["calibration"] = {"n": len(pairs), "mae": round(statistics.mean(abs(e) for e in err), 4),
                              "median_signed_error": round(statistics.median(err), 4),
                              "within_0.05": round(sum(abs(e) <= 0.05 for e in err) / len(err), 3),
                              "mp_stable_predicted_below_0.05": f"{sum(p <= 0.05 for p, m in pairs if m == 0)}/{sum(m == 0 for p, m in pairs)}"}
        print(res["calibration"])
    json.dump(res, open(os.path.join(DATA, f"stability_{tag}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "compute"])
    ap.add_argument("--set", default="calib", choices=["calib", "novel"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    fetch(a.set) if a.stage == "fetch" else compute(a.set, a.threads)


if __name__ == "__main__":
    main()
