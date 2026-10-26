"""K-Ont as a checker of CrystaLLM samples (BUILD_LOG Step 60).

The one-model K-Ont (Step 52c) was trained on "has the framework of": a material's structure-free description
should be closest to the FULL description (structure lines included) of a material with its framework. Here the
same judgement is applied to CrystaLLM's own samples: for each of the 8 samples of a target (Step 40 plain
fine-tune, `const`), the target's KG description gets the SAMPLE's structure lines (space group, crystal system,
structure, lattice, volume, density, sites, written as the battGPT verbalizer writes them), and K-Ont scores it
against the target's structure-free description. Samples are re-ordered by that score.
Arms (902 K-Ont-unseen targets, sample right = Step 40's strict match):
  CrystaLLM order (s_0 first) | K-Ont checker | untrained encoder checker (OnT-galen, same text) |
  space-group rule: samples whose space group equals the K-Ont top-1 template's come first (no embedding)
  + on the 300-target subset with CHGNet energies (Step 48): CHGNet pick among all 8 vs among the checker's top 4.

    .venv-crystallm/Scripts/python.exe abox/kont_checker.py --stage describe
    .venv/Scripts/python.exe abox/kont_checker.py --stage score
    .venv-crystallm/Scripts/python.exe abox/kont_checker.py --stage eval
"""
import argparse
import json
import os
import random
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
GEN = os.path.join(DATA, "kgcond3_gen", "const")
DESC = os.path.join(DATA, "kont_checker_desc.json")
SCORES = os.path.join(DATA, "kont_checker_scores.json")
OUT = os.path.join(DATA, "kont_checker.json")
VM = os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_materials_full.json")
STRUCT = ("Space group:", "Crystal system:", "Structure:", "Lattice:", "Volume:", "Density:", "Sites:")
MODELS = {"kont": os.path.join(HERE, "data", "runs", "kont_fw_retain_galen_anchor", "final"),
          "base": "Hui97/OnT-MiniLM-L12-galen"}


def struct_lines(s, formula):
    """The battGPT verbalizer's structure lines for a pymatgen Structure."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(s, symprec=0.1)
    p = s.get_primitive_structure()
    L = p.lattice
    counts = {}
    for site in p:
        counts[site.specie.symbol] = counts.get(site.specie.symbol, 0) + 1
    return [f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})",
            f"Crystal system: {sga.get_crystal_system().capitalize()}",
            f"Structure: Crystal structure of {formula}",
            f"Lattice: a={L.a:.4f}, b={L.b:.4f}, c={L.c:.4f} Å; α={L.alpha:.1f}, β={L.beta:.1f}, γ={L.gamma:.1f}°",
            f"Volume: {p.volume:.2f} Å³",
            f"Density: {p.density:.2f} g/cm³",
            f"Sites: {len(p)} ({', '.join(f'{e}:{n}' for e, n in counts.items())})"]


def with_structure(full, lines):
    keep = [l for l in full.split("\n") if not l.startswith(STRUCT)]
    return "\n".join(keep + lines)


def stage_describe(a):
    import kg_pool
    import phaseC_generate as gen
    from pymatgen.core import Structure
    pool, rows = kg_pool.load()
    vm = {k.split("/")[-1]: v for k, v in json.load(open(VM, encoding="utf-8")).items()}
    top8 = json.load(open(os.path.join(DATA, "kg_rerank_top8.json")))["prompt+kg+ft_kont"]
    out = {}
    for n, t in enumerate(pool["targets"]):
        full = vm[t]
        rec = {"query": "\n".join(l for l in full.split("\n") if not l.startswith(STRUCT) or l.startswith("Sites:")),
               "samples": [], "sg": [], "template_sg": None}
        try:
            c = top8[t][0][0]
            rec["template_sg"] = gen.reference_from_text(rows[c]["cif"]).get_space_group_info(symprec=0.1)[1]
        except Exception:
            pass
        for k in range(8):
            try:
                s = Structure.from_str(gen.postprocess(open(os.path.join(GEN, t, f"s_{k}.cif")).read()), fmt="cif")
                rec["samples"].append(with_structure(full, struct_lines(s, rows[t]["formula"])))
                rec["sg"].append(s.get_space_group_info(symprec=0.1)[1])
            except Exception:
                rec["samples"].append(None)
                rec["sg"].append(None)
        out[t] = rec
        if n % 100 == 0:
            print(f"{n + 1}/{len(pool['targets'])}", flush=True)
    json.dump(out, open(DESC, "w", encoding="utf-8"))
    print("described", len(out), "targets")


def stage_score(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "OnT"))
    from ont.hit import HierarchyTransformer
    D = json.load(open(DESC, encoding="utf-8"))
    res = {}
    for name, path in MODELS.items():
        m = HierarchyTransformer.from_pretrained(path).to("cuda").eval()

        def emb(texts):
            out = []
            for i in range(0, len(texts), 64):
                f = m.tokenize(texts[i:i + 64])
                f = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in f.items()}
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out.append(F.normalize(m(f)["sentence_embedding"].float(), dim=-1))
            return torch.cat(out)
        ts = sorted(D)
        q = dict(zip(ts, emb([D[t]["query"] for t in ts])))
        flat = [(t, k, s) for t in ts for k, s in enumerate(D[t]["samples"]) if s]
        e = emb([s for _, _, s in flat])
        sc = {t: [None] * 8 for t in ts}
        for (t, k, _), v in zip(flat, e):
            sc[t][k] = float(v @ q[t])
        res[name] = sc
        del m
        torch.cuda.empty_cache()
        print("scored", name, flush=True)
    json.dump(res, open(SCORES, "w"))


def stage_eval(a):
    D = json.load(open(DESC, encoding="utf-8"))
    S = json.load(open(SCORES))
    right = {x["id"]: [bool(s.get("strict")) for s in x["samples"]]
             for x in json.load(open(os.path.join(DATA, "kgcond3_eval_const.json")))}
    targets = sorted(D)

    def order(t, how):
        ks = [k for k in range(8)]
        if how == "crystallm":
            return ks
        if how == "sg_rule":
            return sorted(ks, key=lambda k: (D[t]["sg"][k] != D[t]["template_sg"], k))
        sc = S[how][t]
        return sorted(ks, key=lambda k: (-(sc[k] if sc[k] is not None else -9), k))
    arms = {"CrystaLLM order": "crystallm", "K-Ont checker": "kont", "untrained-encoder checker": "base",
            "space-group rule (K-Ont template)": "sg_rule"}
    found = {n: {kk: {t: float(any(right[t][k] for k in order(t, h)[:kk])) for t in targets} for kk in (1, 2, 3, 5)}
             for n, h in arms.items()}
    rng = random.Random(0)
    B = [[rng.choice(targets) for _ in targets] for _ in range(5000)]

    def diff(x, y, kk, T=targets, F=found):
        d = {t: F[x][kk][t] - F[y][kk][t] for t in T}
        bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B) if T is targets else None
        if bs is None:
            r2 = random.Random(1)
            bs = sorted(100 * sum(d[t] for t in (r2.choice(T) for _ in T)) / len(T) for _ in range(5000))
        return [round(100 * sum(d.values()) / len(T), 1), round(bs[125], 1), round(bs[4874], 1)]
    rep = {"n_targets": len(targets),
           "found@k": {n: {f"@{kk}": round(100 * sum(v.values()) / len(targets), 1) for kk, v in f.items()}
                       for n, f in found.items()},
           "any of 8 (ceiling)": round(100 * sum(any(right[t]) for t in targets) / len(targets), 1)}
    for n in arms:
        if n != "CrystaLLM order":
            rep[f"{n} - CrystaLLM order"] = {f"@{kk}": diff(n, "CrystaLLM order", kk) for kk in (1, 3)}
    # CHGNet single answer on the Step 48 subset
    E = json.load(open(os.path.join(DATA, "kg_pipeline_energies_s40.json")))
    sub = sorted({k.split("|")[0] for k in E})
    pick = {}

    def chg(t, ks):
        got = [(E[f"{t}|C|{k}"][0], E[f"{t}|C|{k}"][1]) for k in ks if E.get(f"{t}|C|{k}")]
        return float(min(got)[1]) if got else 0.0
    pick["CHGNet among 8 CrystaLLM"] = {t: chg(t, range(8)) for t in sub}
    for n, h in (("CHGNet among K-Ont checker's top 4", "kont"), ("CHGNet among untrained checker's top 4", "base"),
                 ("CHGNet among space-group rule's top 4", "sg_rule")):
        pick[n] = {t: chg(t, order(t, h)[:4]) for t in sub}
    rep["single answer (300 targets)"] = {n: round(100 * sum(v.values()) / len(sub), 1) for n, v in pick.items()}
    r2 = random.Random(0)
    B2 = [[r2.choice(sub) for _ in sub] for _ in range(5000)]
    for n in pick:
        if n != "CHGNet among 8 CrystaLLM":
            d = {t: pick[n][t] - pick["CHGNet among 8 CrystaLLM"][t] for t in sub}
            bs = sorted(100 * sum(d[t] for t in b) / len(b) for b in B2)
            rep[f"{n} - CHGNet among 8"] = [round(100 * sum(d.values()) / len(sub), 1), round(bs[125], 1), round(bs[4874], 1)]
    json.dump({"report": rep}, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["describe", "score", "eval"])
    a = ap.parse_args()
    {"describe": stage_describe, "score": stage_score, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
