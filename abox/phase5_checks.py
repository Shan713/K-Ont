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
# every line of a material sentence that restates its crystal (check F strips these)
STRUCT_LINES = ("Space group:", "Crystal system:", "Structure family:", "Structure:", "Lattice:", "Volume:",
                "Density:", "Sites:")


LABEL_LINES = ("Structure family:", "Battery role:")  # Step 36: the targets of hasStructureFamily / belongsToElectrode
ROLE_SENT = {"PositiveElectrode": "Battery role: positive electrode (cathode)",
             "NegativeElectrode": "Battery role: negative electrode (anode)"}  # belongsToElectrode's range
FAMILY_READABLE = {"LayeredOxideStructure": "layered oxide", "SpinelStructure": "spinel", "RockSaltStructure": "rock salt",
                   "OlivineStructure": "olivine", "NASICONStructure": "NASICON", "GarnetStructure": "garnet",
                   "PerovskiteStructure": "perovskite", "LGPSTypeStructure": "LGPS-type", "ArgyroditeStructure": "argyrodite"}
FAMILY_CLASS = {"layered oxide": "layered oxide structure", "spinel": "spinel structure", "rock salt": "rock salt structure",
                "olivine": "olivine structure", "NASICON": "NASICON structure", "garnet": "garnet structure",
                "perovskite": "perovskite structure", "LGPS-type": "LGPS type structure", "argyrodite": "argyrodite structure"}


def strip_lines(sentence, prefixes):
    return "\n".join(l for l in sentence.split("\n") if not l.startswith(prefixes))


def load(name, abox_dir=ABOX):
    with open(os.path.join(abox_dir, name), encoding="utf-8") as f:
        return json.load(f)


def enc(model, sents):
    return model.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()


def ranks_of_true(scores, true_idx):
    """scores: [n, k] higher = better; true_idx: [n]. 1-based rank of the true column per row.

    Ties count as a random tie-break (expected rank), not in the true column's favour: since Step 16,
    no_geometry crystals with the same symmetry have identical sentences -> identical embeddings, and
    a strict ">" count would silently rank the true one first among its exact duplicates."""
    true = scores.gather(1, true_idx.view(-1, 1))
    ties = (scores == true).sum(1) - 1  # other columns scoring exactly the same
    return (scores > true).sum(1) + 1 + ties.float() / 2


def rank_summary(r):
    r = r.float()
    return {"n": len(r), "MRR": round((1 / r).mean().item(), 4), "H@1": round((r == 1).float().mean().item(), 4),
            "H@10": round((r <= 10).float().mean().item(), 4), "median_rank": float(r.median().item())}


def hard_negative_ranks(scores, true_idx, groups):
    """Rank of the true crystal among ONLY itself + the crystals of other materials in the same
    chemical system (the confusable set, Step 10). groups[i] = candidate column indices for row i,
    true one included; rows with no same-chemsys alternative are skipped."""
    out = []
    for i, cand in enumerate(groups):
        if len(cand) < 2:
            continue
        s = scores[i, cand]
        t = s[cand.index(int(true_idx[i]))]
        out.append((s > t).sum().item() + 1 + ((s == t).sum().item() - 1) / 2)
    return torch.tensor(out)


def pct(x, q):
    s = sorted(x)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["full", "no_geometry"], required=True)
    ap.add_argument("--split", default=None,
                    help="split.json (abox/split.py): also report typing + hasStructure per side (train / held-out)")
    ap.add_argument("--abox-dir", default=ABOX,
                    help="verbalizations the run was TRAINED on (e.g. the pre-Step-16 backup for older runs)")
    a = ap.parse_args()

    run_cfg = json.load(open(os.path.join(a.run, "step_times.json")))
    split = json.load(open(a.split)) if a.split else None
    after = OntologyTransformer.from_pretrained(os.path.join(a.run, "final"))
    after.hit_model.eval()
    lam = after.best_lambda or 0.0
    base = run_cfg.get("base_model", "sentence-transformers/all-MiniLM-L12-v2")  # older runs: fit() default
    before = HierarchyTransformer.from_pretrained(base,
                                                  device=next(after.hit_model.parameters()).device)
    before.eval()
    man = after.manifold
    radius = 1 / math.sqrt(float(man.c))

    ents = load("entities.json", a.abox_dir)
    vm = load(f"verbalizations_materials_{a.variant}.json", a.abox_dir)
    vc = load(f"verbalizations_crystals_{a.variant}.json", a.abox_dir)
    ve = load("verbalizations_elements.json", a.abox_dir)
    classes = [v for _, v in sorted(json.load(open(os.path.join(a.run, "data", "concept_names.json"))).items(),
                                    key=lambda kv: int(kv[0]))]
    mats = sorted(vm)
    crys = [ents["materials"][m]["structure"] for m in mats]  # crystal key for each material, same order
    crys_all = sorted(vc)

    out = {"run": a.run, "variant": a.variant, "abox_dir": a.abox_dir, "base_model": base, "best_lambda": lam,
           "ball_radius": round(radius, 3),
           "split": a.split,
           "NOTE": ("top-level B/C numbers pool ALL materials; with --split, see <before|after>.split.test for "
                    "held-out materials (in no training row). Without --split everything here is training fit.")}

    def pdist(x, y, chunk=128):
        """man.dist for every pair, in row chunks: broadcasting [n, m, 384] at once needs ~38 GB at 5,000"""
        return torch.cat([man.dist(x[i:i + chunk].unsqueeze(1), y.unsqueeze(0)) for i in range(0, len(x), chunk)])

    with torch.no_grad():
        for tag, model in (("before", before), ("after", after)):
            res = {}
            C = enc(model, classes)
            # A. class separation
            dC = pdist(C, C)
            off = dC + torch.eye(len(classes), device=dC.device) * 1e9
            i, j = divmod(int(off.argmin()), len(classes))
            res["A_class_separation"] = {
                "substance_vs_crystal_structure": round(man.dist(C[classes.index("substance")],
                                                                 C[classes.index("crystal structure")]).item(), 3),
                "closest_class_pair": [classes[i], classes[j], round(off.min().item(), 3)],
                "median_pairwise": round(dC[~torch.eye(len(classes), dtype=bool, device=dC.device)].median().item(), 3)}

            # B. typing: nearest class by OnT's own hierarchy score (distance + lambda * norm gap)
            def score(child, parents):
                return -(pdist(child, parents)
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
            groups = [[crys_all.index(ents["materials"][o]["structure"]) for o in mats
                       if ents["materials"][o]["chemsys"] == ents["materials"][m]["chemsys"]] for m in mats]
            n_hn = sum(len(g) >= 2 for g in groups)
            # references, so a number can be read without context: random ranking, random among the
            # same-chemsys confusable set, and a perfect ranker that only sees symmetry (family, space
            # group, crystal system) -- the ceiling for no_geometry crystals since Step 16
            sig = [(ents["crystals"][c].get("structure_family"), ents["crystals"][c].get("space_group"),
                    ents["crystals"][c].get("crystal_system")) for c in crys_all]
            H = lambda n: sum(1 / k for k in range(1, n + 1))
            hs = {"random_baseline_MRR": round(H(len(crys_all)) / len(crys_all), 4),
                  "symmetry_only_oracle_MRR": round(sum(H(sig.count(sig[t])) / sig.count(sig[t]) for t in true_idx.tolist()) / len(mats), 4),
                  "hard_negative_n_materials": n_hn,
                  "hard_negative_random_MRR": round(sum(H(len(g)) / len(g) for g in groups if len(g) >= 2) / max(n_hn, 1), 4)}
            plain = score(M, X)  # no role at all: is V(m) simply close to V(its crystal)?
            hs["plain_distance_no_role"] = rank_summary(ranks_of_true(plain, true_idx))
            hs["plain_distance_no_role_hard_negative"] = rank_summary(hard_negative_ranks(plain, true_idx, groups))
            sr = None
            if tag == "after":  # the role rotation only exists after training (fit() initializes it fresh)
                R = torch.tensor(after.encode_existence(["has structure"] * len(crys_all),
                                                        [vc[c] for c in crys_all]), device=M.device)
                sr = score(M, R)
                hs["via_role_rotation"] = rank_summary(ranks_of_true(sr, true_idx))
                hs["via_role_rotation_hard_negative"] = rank_summary(hard_negative_ranks(sr, true_idx, groups))
            res["C_hasStructure"] = hs

            # Per-side numbers for a held-out split (Step 17). "test" rows are materials/crystals that
            # appear in NO training row, so these are the first generalization numbers in this log.
            if split:
                cls_idx = {n: classes.index(t) for n, t in (("materials", "substance"), ("crystals", "crystal structure"))}
                sides = {}
                for side in ("train", "test"):
                    mi = [n for n, m in enumerate(mats) if m in split[f"{side}_materials"]]
                    ci = [crys_all.index(c) for c in split[f"{side}_crystals"]]
                    d = {"n_materials": len(mi)}
                    for name, emb, rows_ in (("materials", M, mi), ("crystals", X, ci)):
                        sc = score(emb[rows_], C)
                        d[f"typing_{name}_H@1"] = round((sc.argmax(1) == cls_idx[name]).float().mean().item(), 4)
                    mats_score = {"plain_no_role": plain}
                    if tag == "after":
                        mats_score["via_role"] = sr
                    for k, sc in mats_score.items():
                        ti = true_idx[mi]
                        d[f"hasStructure_{k}_all250"] = rank_summary(ranks_of_true(sc[mi], ti))
                        # candidates = only this side's crystals (for test: 51 crystals never seen in training)
                        cols = torch.tensor(ci, device=sc.device)
                        pos = torch.tensor([ci.index(int(t)) for t in ti.tolist()], device=sc.device)
                        d[f"hasStructure_{k}_among_{side}_crystals"] = rank_summary(ranks_of_true(sc[mi][:, cols], pos))
                        d[f"hasStructure_{k}_hard_negative"] = rank_summary(
                            hard_negative_ranks(sc[mi], ti, [groups[n] for n in mi]))
                    H_ = lambda n: sum(1 / k for k in range(1, n + 1))
                    d["random_MRR_among_side_crystals"] = round(H_(len(ci)) / len(ci), 4)
                    # F. polymorphs (Step 22): rank the true crystal among ONLY the crystals of materials
                    # with the same formula. The material sentence repeats its crystal's symmetry and
                    # geometry lines verbatim, so "as_is" is string matching; "structure_lines_removed"
                    # drops them from the material sentence (test-time only), leaving properties alone.
                    poly = [[crys_all.index(ents["materials"][o]["structure"]) for o in mats
                             if ents["materials"][o]["formula"] == ents["materials"][mats[n]]["formula"]] for n in mi]
                    keep = [x for x, g in enumerate(poly) if len(g) >= 2]
                    if keep:
                        rows_k = [mi[x] for x in keep]
                        stripped = enc(model, ["\n".join(l for l in vm[mats[n]].split("\n") if not l.startswith(STRUCT_LINES))
                                               for n in rows_k])
                        if tag == "after":
                            target, sr_rows = R, sr[rows_k]
                        else:
                            target, sr_rows = X, plain[rows_k]
                        sc_strip = score(stripped, target)
                        ti = true_idx[rows_k]
                        groups_k = [poly[x] for x in keep]
                        d["F_polymorphs"] = {
                            "scored_by": "role rotation" if tag == "after" else "plain distance",
                            "n_materials": len(keep),
                            "random_MRR": round(sum(H_(len(g)) / len(g) for g in groups_k) / len(keep), 4),
                            "as_is": rank_summary(hard_negative_ranks(sr_rows, ti, groups_k)),
                            "structure_lines_removed": rank_summary(hard_negative_ranks(sc_strip, ti, groups_k))}
                    sides[side] = d
                res["split"] = sides

            # G/H. Step 36: the two ontology relations whose targets are concept individuals --
            # belongsToElectrode (material -> role) and hasStructureFamily (crystal -> family) -- ranked
            # among all candidate individuals, per split side, "as_is" and with the label lines removed
            # from the child sentence (a model trained with them in the text may just copy them).
            # Baselines: majority class (accuracy) and 1/k (balanced accuracy = mean per-class recall).
            def label_task(role, cand, items, sent_of, true_of, train_items):
                keys = sorted({true_of(i) for i in train_items + items if true_of(i)} & set(cand))
                if not keys:
                    return None
                T = enc(model, [cand[k] for k in keys])
                if tag == "after":
                    T = torch.tensor(after.encode_existence([role] * len(keys), [cand[k] for k in keys]), device=T.device)
                out_ = {}
                maj = max(keys, key=lambda k: sum(true_of(i) == k for i in train_items))
                for side_name, its in (("train", train_items), ("test", items)):
                    its = [i for i in its if true_of(i) in keys]
                    if not its:
                        continue
                    t = torch.tensor([keys.index(true_of(i)) for i in its], device=T.device)
                    row = {"n": len(its), "majority_class": maj,
                           "majority_baseline_acc": round(sum(true_of(i) == maj for i in its) / len(its), 4),
                           "chance_balanced_acc": round(1 / len(keys), 4)}
                    for how, sents in (("as_is", [sent_of(i) for i in its]),
                                       ("labels_removed", [strip_lines(sent_of(i), LABEL_LINES) for i in its])):
                        sc = score(enc(model, sents), T)
                        pred = sc.argmax(1)
                        recalls = {keys[k]: round(((pred == k) & (t == k)).sum().item() / max((t == k).sum().item(), 1), 4)
                                   for k in range(len(keys)) if (t == k).any()}
                        row[how] = {"acc": round((pred == t).float().mean().item(), 4),
                                    "balanced_acc": round(sum(recalls.values()) / len(recalls), 4),
                                    "recall_per_class": recalls, "MRR": rank_summary(ranks_of_true(sc, t))["MRR"]}
                    out_[side_name] = row
                return out_

            tr_m = [m for m in mats if not split or m in split["train_materials"]]
            te_m = [m for m in mats if split and m in split["test_materials"]]
            res["G_belongsToElectrode"] = label_task(
                "belongs to electrode", ROLE_SENT, te_m, lambda m: vm[m],
                lambda m: ents["materials"][m].get("battery_role"), tr_m)
            fam_sent = {k: f"Structure family: {v}" for k, v in FAMILY_READABLE.items()}
            tr_c = [c for c in crys_all if not split or c in split["train_crystals"]]
            te_c = [c for c in crys_all if split and c in split["test_crystals"]]
            res["H_hasStructureFamily"] = label_task(
                "has structure family", fam_sent, te_c, lambda c: vc[c],
                lambda c: ents["crystals"][c].get("structure_family"), tr_c)

            # I. Step 36: is the ontology itself still encoded? Every TBox subsumption axiom (child ⊑ parent,
            # from DeepOnto's train.jsonl): rank the parent among all class names; and the canonical
            # family / role individuals typed under their class.
            tbox = []
            with open(os.path.join(HERE, "data", "battgpt_ont", "train.jsonl"), encoding="utf-8") as f:
                for line in f:
                    r_ = json.loads(line)
                    # atomic axioms only (DeepOnto also writes "C ⊑ r some D" restrictions)
                    if r_["parent"] in classes and (r_["child"], r_["parent"]) not in tbox:
                        tbox.append((r_["child"], r_["parent"]))
            indiv = [(f"Structure family: {k}", v) for k, v in FAMILY_CLASS.items()] + \
                    [(ROLE_SENT["PositiveElectrode"], "positive electrode role"), (ROLE_SENT["NegativeElectrode"], "negative electrode role"),
                     ("Battery role: electrolyte", "electrolyte role"), ("Battery role: separator", "separator role")]
            hier = {}
            for name, pairs_ in (("tbox_axioms", tbox), ("label_individuals", indiv)):
                sc = score(enc(model, [c for c, _ in pairs_]), C)
                for n, (c, _) in enumerate(pairs_):  # a class may not answer with itself
                    if c in classes:
                        sc[n, classes.index(c)] = -1e9
                t = torch.tensor([classes.index(p) for _, p in pairs_], device=sc.device)
                hier[name] = {**rank_summary(ranks_of_true(sc, t)),
                              "random_MRR": round(sum(1 / k for k in range(1, len(classes))) / (len(classes) - 1), 4),
                              "wrong_top1": [[c[:40], p, classes[int(sc[n].argmax())]] for n, (c, p) in enumerate(pairs_)
                                             if classes[int(sc[n].argmax())] != p][:10]}
            res["I_hierarchy"] = hier

            # D. crowding (on at most 1,500 materials: all pairs of 5,000 is 12.5 M distances)
            fam = defaultdict(list)
            for n, m in enumerate(mats):
                f = ents["crystals"][ents["materials"][m]["structure"]].get("structure_family")
                if f and f != "None":
                    fam[f].append(n)
            mats_d = mats
            if len(mats) > 1500:  # fixed random subset; family index lists remapped onto it
                import random as _r
                sub = sorted(_r.Random(0).sample(range(len(mats)), 1500))
                pos_ = {n: x for x, n in enumerate(sub)}
                fam = {f: [pos_[n] for n in idx if n in pos_] for f, idx in fam.items()}
                M, mats_d = M[sub], [mats[n] for n in sub]
            dM = pdist(M, M)
            triu = torch.triu(torch.ones_like(dM, dtype=bool), 1)
            allpairs = dM[triu].tolist()
            within = [dM[p, q].item() for idx in fam.values() for x, p in enumerate(idx) for q in idx[x + 1:]]
            nn = (dM + torch.eye(len(mats_d), device=dM.device) * 1e9).min(1).values.tolist()
            eu = (M.norm(dim=1) / radius).tolist()
            res["D_crowding"] = {
                "material_pairwise_median": round(pct(allpairs, .5), 3), "material_pairwise_p5": round(pct(allpairs, .05), 3),
                "within_family_median": round(pct(within, .5), 3), "within_family_n_pairs": len(within),
                "nearest_neighbour_min": round(min(nn), 4), "nearest_neighbour_median": round(pct(nn, .5), 3),
                "euclid_norm_over_radius_median": round(pct(eu, .5), 4), "euclid_norm_over_radius_max": round(max(eu), 4),
                "hyperbolic_norm_median": round(pct(man.dist0(M).tolist(), .5), 3)}

            # E. confusable (same chemical system) pairs
            by_sys = defaultdict(list)
            for n, m in enumerate(mats_d):
                by_sys[ents["materials"][m]["chemsys"]].append(n)
            pairs = [(p, q) for idx in by_sys.values() for x, p in enumerate(idx) for q in idx[x + 1:]]
            pd = [dM[p, q].item() for p, q in pairs]
            ex = next(((p, q) for p, q in pairs if {ents["materials"][mats_d[p]]["formula"], ents["materials"][mats_d[q]]["formula"]}
                       == {"LiTi2O4", "LiTiO2"}), pairs[0] if pairs else None)
            res["E_same_chemsys_pairs"] = {
                "n_pairs": len(pairs), "median_dist": round(pct(pd, .5), 3) if pd else None,
                "min_dist": round(min(pd), 4) if pd else None, "n_collapsed_lt_1e-3": sum(d < 1e-3 for d in pd),
                "example": [ents["materials"][mats_d[ex[0]]]["formula"], ents["materials"][mats_d[ex[1]]]["formula"],
                            round(dM[ex[0], ex[1]].item(), 3)] if ex else None}
            out[tag] = res

    with open(os.path.join(a.run, "phase5_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
