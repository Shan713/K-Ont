"""30k generation, P2: K-Ont-ranked templates for the P1 formulas (Step 64).

For every selected formula of data/gen30k/compositions.json:
  cands  candidate frameworks = template-pool materials (all 5,000 KG + 5,267 Materials Project extension) with the
         same anonymous formula and O on the same-count site ("oxide templates"); a formula with no oxide template
         falls back to any same-pattern template, flagged `non_oxide` (e.g. a phosphate framework for an oxide).
         Formulas with neither (all senary / septenary, a few ternary / quaternary) are left to P3 and CrystaLLM.
         Query sentence = what a new formula has: name, type, formula, chemical system, cell sites (formula x Z,
         Z the largest of 1, 2, 4 with <= 40 atoms) -- the composition-only form of Step 44.
  score  K-Ont retriever fine-tuned on composition-only queries (kg_retriever_ft.py --minimal --save, Step 64):
         cosine between the query and each candidate's full sentence; top --keep kept per formula.
  build  element substitution exactly as Steps 41 / 61 (elements matched by amount then electronegativity, cell
         scaled by atomic volume, no relaxation), in score order until --per-formula distinct structures pass:
         right reduced composition, O on the template's O sites, no two atoms closer than 0.5 A, and not a
         duplicate of an earlier template for the formula (StructureMatcher 0.2 / 0.3 / 5 deg).
         -> data/gen30k/templates/<system>/<formula>__t<k>__<template id>.cif  + templates_manifest.json

    .venv-crystallm/Scripts/python.exe abox/gen30k_templates.py --stage cands
    .venv/Scripts/python.exe           abox/gen30k_templates.py --stage score
    .venv-crystallm/Scripts/python.exe abox/gen30k_templates.py --stage build
"""
import argparse
import gzip
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(HERE, "data", "gen30k")
CANDS = os.path.join(OUT, "template_candidates.json")
SCORES = os.path.join(OUT, "template_scores.json")
# composition-only K-Ont retriever without retention: 61.1% top-1 right framework on the 902 test materials, vs 56.2% for
# the one-model (retention) version kont_fw_minimal -- for generation the better ranker wins (Step 64)
MODEL = os.path.join(HERE, "data", "runs", "kont_fw_minimal_plain", "final")


def pool_rows():
    rows = {r["id"]: r for s in ("train", "val", "test")
            for r in json.load(gzip.open(os.path.join(DATA, f"kg_corpus_{s}.json.gz"), "rt", encoding="utf-8"))}
    for r in json.load(gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "rt", encoding="utf-8")):
        rows.setdefault(r["id"], r)
    return rows


def cell_for(formula, max_atoms=40):
    from pymatgen.core import Composition
    c = Composition(formula)
    z = max([z for z in (1, 2, 4) if c.num_atoms * z <= max_atoms] or [1])
    return c * z, z


def stage_cands(a):
    from pymatgen.core import Composition
    from gen30k_compositions import oxide_key
    rows = pool_rows()
    by_anon, by_oxide = {}, {}
    for i, r in rows.items():
        c = Composition(r["formula"])
        by_anon.setdefault(c.anonymized_formula, []).append(i)
        if "O" in c:
            by_oxide.setdefault(oxide_key(c), []).append(i)
    comps = json.load(open(os.path.join(OUT, "compositions.json")))
    out, stats = {}, {}
    for sysname, s in comps["systems"].items():
        n_ox = n_any = 0
        for f in s["selected"]:
            comp = Composition(f["reduced"])
            ox = [i for i in by_oxide.get(oxide_key(comp), []) if Composition(rows[i]["formula"]).reduced_formula != f["reduced"]]
            anyp = [i for i in by_anon.get(comp.anonymized_formula, []) if Composition(rows[i]["formula"]).reduced_formula != f["reduced"]]
            cands, kind = (ox, "oxide") if ox else (anyp, "non_oxide") if anyp else ([], None)
            if not cands:
                continue
            n_ox += kind == "oxide"
            n_any += kind == "non_oxide"
            cell, z = cell_for(f["reduced"])
            sites = ", ".join(f"{e.symbol}:{int(v)}" for e, v in cell.items())
            text = "\n".join([f"Material: {f['reduced']} (gen30k)", "Type: substance", f"Formula: {f['reduced']}",
                              "Chemical system: " + "-".join(sorted(e.symbol for e in comp.elements)),
                              f"Sites: {int(cell.num_atoms)} ({sites})"])
            out[f"{sysname}|{f['reduced']}"] = {"system": sysname, "formula": f["reduced"], "text": text, "kind": kind,
                                                "candidates": cands}
        stats[sysname] = {"selected": len(s["selected"]), "oxide": n_ox, "non_oxide": n_any}
        print(f"{sysname:20s} {stats[sysname]}", flush=True)
    json.dump({"stats": stats, "formulas": out}, open(CANDS, "w"))
    print(f"{len(out)} formulas with candidates; {sum(len(v['candidates']) for v in out.values())} (formula, template) pairs")


def stage_score(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from ont.hit import HierarchyTransformer
    d = json.load(open(CANDS))["formulas"]
    vm = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json"), encoding="utf-8"))
    full = {k.split("/")[-1]: v for k, v in vm.items()}
    for r in json.load(gzip.open(os.path.join(DATA, "kg_ext_pool.json.gz"), "rt", encoding="utf-8")):
        full.setdefault(r["id"], r["sentence"])
    m = HierarchyTransformer.from_pretrained(a.model).to("cuda").eval()

    def emb(texts):
        out = []
        for i in range(0, len(texts), 64):
            f = m.tokenize(texts[i:i + 64])
            f = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in f.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out.append(F.normalize(m(f)["sentence_embedding"].float(), dim=-1))
        return torch.cat(out)
    cs = sorted({c for v in d.values() for c in v["candidates"]})
    missing = [c for c in cs if c not in full]
    cs = [c for c in cs if c in full]
    ce = dict(zip(cs, emb([full[c] for c in cs])))
    keys = sorted(d)
    qe = dict(zip(keys, emb([d[k]["text"] for k in keys])))
    scores = {}
    for k in keys:
        sc = {c: float(ce[c] @ qe[k]) for c in d[k]["candidates"] if c in ce}
        scores[k] = sorted(sc.items(), key=lambda kv: -kv[1])[:a.keep]
    json.dump({"model": a.model, "keep": a.keep, "missing_sentence": missing, "scores": scores}, open(SCORES, "w"))
    print(f"scored {len(keys)} formulas over {len(cs)} candidate templates ({len(missing)} without a sentence)")


def substitute(target, cand_formula, cand_cif):
    """Steps 41 / 61 substitution; returns (structure, mapping) or (None, reason)"""
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.core import Composition, Element
    try:
        s = gen.reference_from_text(cand_cif)
    except Exception:
        return None, "template does not parse"
    mapping = {x.symbol: y.symbol for x, y in zip(k2._ordered(cand_formula), k2._ordered(target))}
    old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.replace_species(mapping)
    if s.composition.reduced_formula != Composition(target).reduced_formula:
        return None, "composition"
    new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.scale_lattice(s.volume * new / old)
    return s, mapping


def stage_build(a):
    import numpy as np
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.cif import CifWriter
    rows = pool_rows()
    d = json.load(open(CANDS))["formulas"]
    sc = json.load(open(SCORES))["scores"]
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    man, reasons, t0 = [], {}, time.time()
    for n, k in enumerate(sorted(sc)):
        v = d[k]
        kept = []
        for rank, (cid, score) in enumerate(sc[k], 1):
            if len(kept) >= a.per_formula:
                break
            r = rows[cid]
            s, mapping = substitute(v["formula"], r["formula"], r["cif"])
            if s is None:
                reasons[mapping] = reasons.get(mapping, 0) + 1
                continue
            if v["kind"] == "oxide" and mapping.get("O") != "O":
                reasons["O not on O sites"] = reasons.get("O not on O sites", 0) + 1
                continue
            if len(s) > 1 and (s.distance_matrix + 10 * np.eye(len(s))).min() < 0.5:
                reasons["clash < 0.5 A"] = reasons.get("clash < 0.5 A", 0) + 1
                continue
            if any(sm.fit(s, q) for q in kept):
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue
            kept.append(s)
            sub = os.path.join(OUT, "templates", v["system"])
            os.makedirs(sub, exist_ok=True)
            fn = f"{v['formula']}__t{len(kept)}__{cid}.cif"
            CifWriter(s).write_file(os.path.join(sub, fn))
            man.append({"file": f"templates/{v['system']}/{fn}", "system": v["system"], "formula": v["formula"],
                        "source": "kont_template", "template_id": cid, "template_formula": r["formula"],
                        "template_space_group": r.get("space_group"), "kont_rank": rank, "kont_score": round(score, 4),
                        "template_kind": v["kind"], "mapping": mapping, "n_atoms": len(s)})
        if (n + 1) % 250 == 0:
            print(f"{n + 1}/{len(sc)} formulas, {len(man)} templates ({(time.time() - t0) / 60:.1f} min)", flush=True)
    per_sys = {}
    for x in man:
        per_sys[x["system"]] = per_sys.get(x["system"], 0) + 1
    json.dump({"per_formula": a.per_formula, "rejected": reasons, "per_system": per_sys, "rows": man},
              open(os.path.join(OUT, "templates_manifest.json"), "w"), indent=1)
    print(json.dumps({"templates": len(man), "rejected": reasons, "per_system": per_sys}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cands", "score", "build"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--keep", type=int, default=12, help="score: candidates kept per formula, in K-Ont order")
    ap.add_argument("--per-formula", type=int, default=5, help="build: distinct valid templates per formula")
    a = ap.parse_args()
    {"cands": stage_cands, "score": stage_score, "build": stage_build}[a.stage](a)


if __name__ == "__main__":
    main()
