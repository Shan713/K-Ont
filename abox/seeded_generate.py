"""Task A: KG-seeded generation -- does seeding CrystaLLM with a KG analogue's framework help?

For each target composition with a KG analogue (one element swapped, TM<->TM or Li<->Na):
  target  = the analogue's conventional standard cell with the element swapped, volume scaled by the KG
            volume model: V_target = V_analogue x V_KG(target) / V_KG(analogue),
            V_KG = sum(n_element x volume_per_atom[element]) from data/crystallm/kg_priors.json
Prompt levels (same seeds per composition):
  L0 composition      data_<cell>
  L1 parent_sg        + the analogue's space group
  L2 parent_lattice   + the analogue's lattice scaled to the target (CrystaLLM then writes the atom sites)
  L3 substitution     no generation: the scaled, swapped analogue itself (the control)
Per CIF: valid, right composition, KG score/plausible, keeps the analogue's framework
(StructureMatcher.fit_anonymous vs the analogue), strict match vs the true structure where known.
CHGNet: up to --chgnet-per-level valid right-composition candidates per level, plus L3 and the truth, are
relaxed; reported: energy, match after relaxation, framework after relaxation.

Sets: --set test  -> data/crystallm/test_analogues.json (22 test cathodes, truth known)
      --set novel -> novel_candidates.json + novel_parents.json (200 compositions, no truth)

    python -u abox/seeded_generate.py --set test --model crystallm_v1_large --batch 5
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

import phaseC_generate as gen  # noqa: E402

DATA = gen.DATA
LEVELS = ("composition", "parent_sg", "parent_lattice")


def kg_volume(comp, vpa):
    return sum(n * vpa.get(str(el), 0.0) for el, n in comp.items())


def build_target(analogue_cif, swap, vpa):
    """(conventional analogue, scaled swapped target, analogue space-group number)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    a, b = swap
    conv = SpacegroupAnalyzer(Structure.from_str(analogue_cif, fmt="cif"), symprec=0.1).get_conventional_standard_structure()
    sg = SpacegroupAnalyzer(conv, symprec=0.1).get_space_group_number()
    target = conv.copy()
    target.replace_species({a: b})
    va, vt = kg_volume(conv.composition, vpa), kg_volume(target.composition, vpa)
    if va > 0 and vt > 0:
        target.scale_lattice(conv.volume * vt / va)
    return conv, target, sg


def lattice_prompt(cell, sg_sym, lat):
    p = gen.prompt_for(cell, sg_sym)
    return p + "".join(f"{k} {v:.4f}\n" for k, v in (
        ("_cell_length_a", lat.a), ("_cell_length_b", lat.b), ("_cell_length_c", lat.c),
        ("_cell_angle_alpha", lat.alpha), ("_cell_angle_beta", lat.beta), ("_cell_angle_gamma", lat.gamma)))


def score(args):
    """Worker: one CIF (generated text, or a ready structure as CIF for L3)."""
    raw, is_generated, target_formula, analogue_cif, truth_cif = args
    import kg_steer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(raw) if is_generated else raw, fmt="cif")
    except Exception:
        return {"valid": False}
    row = {"valid": True, "composition_ok": s.composition.reduced_formula == target_formula}
    if not row["composition_ok"]:
        return row
    row.update(kg_steer.kg_score(s))
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    try:
        row["keeps_framework"] = bool(sm.fit_anonymous(s, Structure.from_str(analogue_cif, fmt="cif")))
    except Exception:
        row["keeps_framework"] = False
    if truth_cif:
        try:
            row["strict"] = bool(sm.fit(s, Structure.from_str(truth_cif, fmt="cif")))
        except Exception:
            row["strict"] = False
    return row


def items_for(set_name):
    if set_name == "test":
        out = []
        for x in json.load(open(os.path.join(DATA, "test_analogues.json"), encoding="utf-8")):
            a, b = x["swap"].split("->")
            out.append({"id": x["test_id"], "formula": x["formula"], "analogue_cif": x["analogue_cif"],
                        "swap": (a, b), "truth": True})
        return out
    parents = json.load(open(os.path.join(DATA, "novel_parents.json"), encoding="utf-8"))
    out = []
    for c in json.load(open(os.path.join(DATA, "novel_candidates.json"), encoding="utf-8")):
        a, b = re.match(r"(\w+)->(\w+)", c["substitution"]).groups()
        out.append({"id": c["id"], "formula": c["formula"], "analogue_cif": parents[c["parent_id"]]["cif"],
                    "swap": (a, b), "truth": False})
    return out


def summarise(materials):
    summ = {}
    for lvl in LEVELS + ("substitution",):
        rows = [r for m in materials for r in m["levels"].get(lvl, {}).get("rows", [])]
        ok = [r for r in rows if r.get("composition_ok") and "total" in r]
        relaxed = [m["levels"][lvl]["best"] for m in materials if m["levels"].get(lvl, {}).get("best")]
        s = {"cifs": len(rows),
             "valid": round(sum(r["valid"] for r in rows) / max(len(rows), 1), 3),
             "right_composition": round(len(ok) / max(len(rows), 1), 3),
             "kg_plausible": round(sum(r["plausible"] for r in ok) / max(len(ok), 1), 3),
             "keeps_framework": round(sum(r["keeps_framework"] for r in ok) / max(len(ok), 1), 3),
             "compositions_with_relaxed": len(relaxed),
             "best_keeps_framework_after_relax": round(sum(b["keeps_framework_after_relax"] for b in relaxed) / max(len(relaxed), 1), 3)}
        if ok and "strict" in ok[0]:
            s["strict_per_cif"] = round(sum(r["strict"] for r in ok) / len(ok), 3)
            s["compositions_found"] = sum(any(r.get("strict") for r in m["levels"].get(lvl, {}).get("rows", [])) for m in materials)
            s["best_energy_pick_correct_after_relax"] = sum(b.get("match_after_relax", False) for b in relaxed)
        de = [m["levels"][lvl]["best"]["e"] - m["levels"]["substitution"]["best"]["e"] for m in materials
              if m["levels"].get(lvl, {}).get("best") and m["levels"].get("substitution", {}).get("best")]
        if de and lvl != "substitution":
            s["median_best_E_minus_substitution_eV"] = round(statistics.median(de), 4)
            s["compositions_lower_than_substitution"] = sum(d < -0.005 for d in de)
        summ[lvl] = s
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test", choices=["test", "novel"])
    ap.add_argument("--model", default="crystallm_v1_large")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--chgnet-per-level", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8, help="CHGNet CPU threads")
    ap.add_argument("--workers", type=int, default=8, help="KG-scoring processes")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 6)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    tag = f"seeded_{a.set}_{a.model.replace('crystallm_v1_', '')}"
    gen_dir = os.path.join(DATA, tag + "_gen")
    out_path = os.path.join(DATA, tag + ".json")
    vpa = json.load(open(os.path.join(DATA, "kg_priors.json")))["volume_per_atom"]
    items = items_for(a.set)[a.start: a.end]
    torch.set_num_threads(a.threads)
    model, tok = gen.load_model("cuda", a.model), gen.CIFTokenizer()
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    chg = CHGNet.load()
    opt = StructOptimizer(model=chg, use_device="cpu")
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def relax(s):
        r = opt.relax(s, fmax=0.1, steps=300, verbose=False)
        return r["final_structure"], float(chg.predict_structure(r["final_structure"])["e"])

    out, t0 = {"settings": vars(a), "materials": []}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, it in enumerate(items):
            conv, target, sg = build_target(it["analogue_cif"], it["swap"], vpa)
            cell = target.composition.formula.replace(" ", "")
            tf = target.composition.reduced_formula
            truth = gen.load_truth({"id": it["id"]}) if it["truth"] else None
            truth_cif = truth.to(fmt="cif") if truth is not None else None
            conv_cif = conv.to(fmt="cif")
            sg_sym = gen.sg_symbol(sg)
            prompts = {"composition": gen.prompt_for(cell)}
            if sg_sym:
                prompts["parent_sg"] = gen.prompt_for(cell, sg_sym)
                prompts["parent_lattice"] = lattice_prompt(cell, sg_sym, target.lattice)
            os.makedirs(os.path.join(gen_dir, it["id"]), exist_ok=True)
            rec = {"id": it["id"], "formula": it["formula"], "target_cell": cell, "analogue_sg": sg,
                   "swap": "->".join(it["swap"]), "levels": {}}
            raw = {}
            for lvl, prompt in prompts.items():
                torch.manual_seed(a.seed + sum(map(ord, it["id"] + lvl)))
                cifs = []
                while len(cifs) < a.samples:
                    cifs += gen.sample_batch(model, tok, prompt, min(a.batch, a.samples - len(cifs)), "cuda")
                for k, c in enumerate(cifs):
                    open(os.path.join(gen_dir, it["id"], f"{lvl}_{k}.cif"), "w").write(c)
                raw[lvl] = cifs
            sub_cif = target.to(fmt="cif")
            open(os.path.join(gen_dir, it["id"], "substitution_0.cif"), "w").write(sub_cif)
            jobs = [(lvl, k, (c, True, tf, conv_cif, truth_cif)) for lvl, cifs in raw.items() for k, c in enumerate(cifs)]
            jobs.append(("substitution", 0, (sub_cif, False, tf, conv_cif, truth_cif)))
            scored = list(ex.map(score, [j[2] for j in jobs]))
            for (lvl, k, _), r in zip(jobs, scored):
                rec["levels"].setdefault(lvl, {"rows": []})["rows"].append({"k": k, **r})
            t_rel = relax(truth)[0] if truth is not None else None
            for lvl, d in rec["levels"].items():
                cands = [r for r in d["rows"] if r.get("composition_ok") and "total" in r][: a.chgnet_per_level]
                best = None
                for r in cands:
                    path = os.path.join(gen_dir, it["id"], f"{lvl}_{r['k']}.cif")
                    try:
                        txt = open(path).read()
                        s = Structure.from_str(gen.postprocess(txt) if lvl != "substitution" else txt, fmt="cif")
                        final, e = relax(s)
                    except Exception:
                        continue
                    r["e"] = round(e, 5)
                    if best is None or e < best[0]:
                        best = (e, r["k"], final)
                if best:
                    e, k, final = best
                    b = {"e": round(e, 5), "k": k, "keeps_framework_after_relax": bool(sm.fit_anonymous(final, conv))}
                    if t_rel is not None:
                        b["match_after_relax"] = bool(sm.fit(final, t_rel))
                    final.to(filename=os.path.join(gen_dir, it["id"], f"best_{lvl}.cif"))
                    d["best"] = b
            out["materials"].append(rec)
            out["summary"] = summarise(out["materials"])
            json.dump(out, open(out_path, "w"), indent=1)
            print(f"{i + 1}/{len(items)} {it['formula']} (analogue sg {sg}): " + " | ".join(
                f"{lvl} plaus {sum(r.get('plausible', False) for r in d['rows'])}/{len(d['rows'])}"
                + (f" strict {sum(r.get('strict', False) for r in d['rows'])}" if it["truth"] else "")
                for lvl, d in rec["levels"].items()) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for k, v in out.get("summary", {}).items():
        print(k, v)


if __name__ == "__main__":
    main()
