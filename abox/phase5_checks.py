"""Design spec Phase 5 sanity checks on a trained K-Ont OnT model (ONT_ABOX_EXTENSION.md §11).

    .venv\\Scripts\\python abox/phase5_checks.py --run data/runs/full_galen_b32_e3 --variant full

Compares the trained model ("after") against its starting point ("before" = the base model the
run started from, read from step_times.json) on:

  A. class separation   -- are the 23 TBox class names still distinct points? (spec: encode("substance") != encode("crystal"))
  B. typing             -- is each material/crystal/element's nearest class (OnT's score_hierarchy) its true type?
  C. hasStructure       -- rank each material's true crystal among all 250 via the learned role rotation
  D. crowding           -- have same-family materials collapsed into one blob near the ball boundary?
  E. confusable pairs   -- same-chemical-system materials (stand-in for the spec's LiMgP/LiZnP, which aren't in this KG)

IMPORTANT: every material here was also in training -- there is no held-out split yet (BUILD_LOG.md
Step 9's scaler note). B and C therefore measure *training fit*, not generalization. This is NOT
the Phase 5b held-out eval; it's the "did training do anything sensible" check that comes before it.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))

import torch  # noqa: E402
from ont.hit import HierarchyTransformer  # noqa: E402
from ont.model import OntologyTransformer  # noqa: E402

ABOX = os.path.join(HERE, "data", "battgpt_abox")


def load(name):
    with open(os.path.join(ABOX, name), encoding="utf-8") as f:
        return json.load(f)


def enc(model, sents):
    return model.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()


def ranks_of_true(scores, true_idx):
    """scores: [n, k] higher = better; true_idx: [n]. 1-based rank of the true column per row."""
    true = scores.gather(1, true_idx.view(-1, 1))
    return (scores > true).sum(1) + 1


def rank_summary(r):
    r = r.float()
    return {"n": len(r), "MRR": round((1 / r).mean().item(), 4), "H@1": round((r == 1).float().mean().item(), 4),
            "H@10": round((r <= 10).float().mean().item(), 4), "median_rank": int(r.median().item())}


def pct(x, q):
    s = sorted(x)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["full", "no_geometry"], required=True)
    a = ap.parse_args()

    run_cfg = json.load(open(os.path.join(a.run, "step_times.json")))
    after = OntologyTransformer.from_pretrained(os.path.join(a.run, "final"))
    after.hit_model.eval()
    lam = after.best_lambda or 0.0
    base = run_cfg.get("base_model", "sentence-transformers/all-MiniLM-L12-v2")  # older runs: fit() default
    before = HierarchyTransformer.from_pretrained(base,
                                                  device=next(after.hit_model.parameters()).device)
    before.eval()
    man = after.manifold
    radius = 1 / math.sqrt(float(man.c))

    ents = load("entities.json")
    vm = load(f"verbalizations_materials_{a.variant}.json")
    vc = load(f"verbalizations_crystals_{a.variant}.json")
    ve = load("verbalizations_elements.json")
    classes = [v for _, v in sorted(json.load(open(os.path.join(a.run, "data", "concept_names.json"))).items(),
                                    key=lambda kv: int(kv[0]))]
    mats = sorted(vm)
    crys = [ents["materials"][m]["structure"] for m in mats]  # crystal key for each material, same order
    crys_all = sorted(vc)

    out = {"run": a.run, "variant": a.variant, "base_model": base, "best_lambda": lam,
           "ball_radius": round(radius, 3),
           "NOTE": "training-set fit only: no held-out split exists yet; this is not the Phase 5b eval"}

    with torch.no_grad():
        for tag, model in (("before", before), ("after", after)):
            res = {}
            C = enc(model, classes)
            # A. class separation
            dC = man.dist(C.unsqueeze(1), C.unsqueeze(0))
            off = dC + torch.eye(len(classes), device=dC.device) * 1e9
            i, j = divmod(int(off.argmin()), len(classes))
            res["A_class_separation"] = {
                "substance_vs_crystal_structure": round(man.dist(C[classes.index("substance")],
                                                                 C[classes.index("crystal structure")]).item(), 3),
                "closest_class_pair": [classes[i], classes[j], round(off.min().item(), 3)],
                "median_pairwise": round(dC[~torch.eye(len(classes), dtype=bool, device=dC.device)].median().item(), 3)}

            # B. typing: nearest class by OnT's own hierarchy score (distance + lambda * norm gap)
            def score(child, parents):
                return -(man.dist(child.unsqueeze(1), parents.unsqueeze(0))
                         + lam * (man.dist0(parents).unsqueeze(0) - man.dist0(child).unsqueeze(1)))
            M = enc(model, [vm[m] for m in mats])
            X = enc(model, [vc[c] for c in crys_all])
            E = enc(model, [ve[e] for e in sorted(ve)])
            typing = {}
            for name, emb, true in (("materials", M, "substance"), ("crystals", X, "crystal structure"),
                                    ("elements", E, "chemical element")):
                s = score(emb, C)
                r = ranks_of_true(s, torch.full((len(emb),), classes.index(true), device=s.device))
                top = defaultdict(int)
                for k in s.argmax(1).tolist():
                    top[classes[k]] += 1
                typing[name] = {**rank_summary(r), "true_type": true,
                                "top1_class_counts": dict(sorted(top.items(), key=lambda kv: -kv[1])[:4])}
            res["B_typing"] = typing

            # C. hasStructure ranking over all 250 crystals
            true_idx = torch.tensor([crys_all.index(c) for c in crys], device=M.device)
            plain = score(M, X)  # no role at all: is V(m) simply close to V(its crystal)?
            hs = {"plain_distance_no_role": rank_summary(ranks_of_true(plain, true_idx)),
                  "random_baseline_MRR": round(sum(1 / k for k in range(1, len(crys_all) + 1)) / len(crys_all), 4)}
            if tag == "after":  # the role rotation only exists after training (fit() initializes it fresh)
                R = torch.tensor(after.encode_existence(["has structure"] * len(crys_all),
                                                        [vc[c] for c in crys_all]), device=M.device)
                hs["via_role_rotation"] = rank_summary(ranks_of_true(score(M, R), true_idx))
            res["C_hasStructure"] = hs

            # D. crowding
            fam = defaultdict(list)
            for n, m in enumerate(mats):
                f = ents["crystals"][ents["materials"][m]["structure"]].get("structure_family")
                if f and f != "None":
                    fam[f].append(n)
            dM = man.dist(M.unsqueeze(1), M.unsqueeze(0))
            triu = torch.triu(torch.ones_like(dM, dtype=bool), 1)
            allpairs = dM[triu].tolist()
            within = [dM[p, q].item() for idx in fam.values() for x, p in enumerate(idx) for q in idx[x + 1:]]
            nn = (dM + torch.eye(len(mats), device=dM.device) * 1e9).min(1).values.tolist()
            eu = (M.norm(dim=1) / radius).tolist()
            res["D_crowding"] = {
                "material_pairwise_median": round(pct(allpairs, .5), 3), "material_pairwise_p5": round(pct(allpairs, .05), 3),
                "within_family_median": round(pct(within, .5), 3), "within_family_n_pairs": len(within),
                "nearest_neighbour_min": round(min(nn), 4), "nearest_neighbour_median": round(pct(nn, .5), 3),
                "euclid_norm_over_radius_median": round(pct(eu, .5), 4), "euclid_norm_over_radius_max": round(max(eu), 4),
                "hyperbolic_norm_median": round(pct(man.dist0(M).tolist(), .5), 3)}

            # E. confusable (same chemical system) pairs
            by_sys = defaultdict(list)
            for n, m in enumerate(mats):
                by_sys[ents["materials"][m]["chemsys"]].append(n)
            pairs = [(p, q) for idx in by_sys.values() for x, p in enumerate(idx) for q in idx[x + 1:]]
            pd = [dM[p, q].item() for p, q in pairs]
            ex = next(((p, q) for p, q in pairs if {ents["materials"][mats[p]]["formula"], ents["materials"][mats[q]]["formula"]}
                       == {"LiTi2O4", "LiTiO2"}), pairs[0] if pairs else None)
            res["E_same_chemsys_pairs"] = {
                "n_pairs": len(pairs), "median_dist": round(pct(pd, .5), 3) if pd else None,
                "min_dist": round(min(pd), 4) if pd else None, "n_collapsed_lt_1e-3": sum(d < 1e-3 for d in pd),
                "example": [ents["materials"][mats[ex[0]]]["formula"], ents["materials"][mats[ex[1]]]["formula"],
                            round(dM[ex[0], ex[1]].item(), 3)] if ex else None}
            out[tag] = res

    with open(os.path.join(a.run, "phase5_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
