"""Step 21 head-to-head: score every hasStructure training method the same way, on held-out data.

For each run (method x variant), on the 51 held-out materials of data/split.json, rank the 51
held-out crystals three ways, so no method is favoured by the scoring rule it was trained for:

  rotation      d(V(m), R_r . V(c))                           -- our relation rows' geometry
  text concept  d(V(m), encode("has structure some " + V(c)))  -- the teammate's is-a rows' geometry
  plain         d(V(m), V(c))

each as MRR over all 51 test crystals and over only the same-chemical-system crystals (hard
negatives, ties broken at random), plus held-out typing and two leak probes that change the TEST
sentences only: no_geometry -- remove the space group / crystal system / structure family lines
from the material (Step 19); full -- shuffle the crystals' "Sites:" (unit-cell contents) lines
(Step 17). The untrained base model is scored too. CPU is fine (~30 s per run).

    python abox/h2h_eval.py --out data/runs/h2h_results.json
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))

import torch  # noqa: E402
from ont.hit import HierarchyTransformer  # noqa: E402
from ont.model import OntologyTransformer  # noqa: E402

ABOX = os.path.join(HERE, "data", "battgpt_abox")
RUNS = os.path.join(HERE, "data", "runs")
METHODS = {  # label -> run-folder suffix
    "standard OnT": "base",
    "M1 ours (in-batch)": "inbatch",
    "M2 his (is-a rows)": "isa",
    "M3 both": "both",
}
SYM = ("Space group:", "Crystal system:", "Structure family:")
HS = "has structure some "


def load(name):
    with open(os.path.join(ABOX, name), encoding="utf-8") as f:
        return json.load(f)


def ranks(scores, true_idx):
    t = scores.gather(1, true_idx.view(-1, 1))
    return (scores > t).sum(1) + 1 + ((scores == t).sum(1) - 1).float() / 2


def mrr(r):
    return round((1 / r.float()).mean().item(), 4)


def hard_mrr(scores, true_idx, groups):
    out = []
    for i, cand in enumerate(groups):
        if len(cand) < 2:
            continue
        s = scores[i, cand]
        t = s[cand.index(int(true_idx[i]))]
        out.append(1.0 / ((s > t).sum().item() + 1 + ((s == t).sum().item() - 1) / 2))
    return round(sum(out) / len(out), 4), len(out)


def evaluate(model, variant, ents, split, sites_perm):
    vm = load(f"verbalizations_materials_{variant}.json")
    vc = load(f"verbalizations_crystals_{variant}.json")
    mats, crys = split["test_materials"], split["test_crystals"]
    true_idx = torch.tensor([crys.index(ents["materials"][m]["structure"]) for m in mats])
    # hard negatives: the true crystal vs every OTHER crystal (train or test) whose material shares
    # the chemical system -- as in phase5_checks.py; within the 51 test crystals alone only 4
    # held-out materials have such a partner, too few to mean anything
    all_mats = sorted(ents["materials"])
    all_crys = [ents["materials"][m]["structure"] for m in all_mats]
    true_all = torch.tensor([all_crys.index(ents["materials"][m]["structure"]) for m in mats])
    groups = [[j for j, o in enumerate(all_mats) if ents["materials"][o]["chemsys"] == ents["materials"][m]["chemsys"]]
              for m in mats]
    man = model.manifold if hasattr(model, "manifold") else model.hit_model.manifold
    enc = lambda s: model.encode(s, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float().cpu()
    dist = lambda A, B: -man.dist(A.unsqueeze(1), B.unsqueeze(0))

    def score_all(mat_txt, cry_txt):
        M = enc(mat_txt)
        out = {"plain": dist(M, enc(cry_txt)), "text concept": dist(M, enc([HS + c for c in cry_txt]))}
        if isinstance(model, OntologyTransformer):
            R = torch.tensor(model.encode_existence(["has structure"] * len(cry_txt), cry_txt))
            out["rotation"] = dist(M, R)
        return out

    res = {}
    with torch.no_grad():
        base_txt = [vm[m] for m in mats]
        cry_txt = [vc[c] for c in crys]
        hard = {how: hard_mrr(sc, true_all, groups) for how, sc in score_all(base_txt, [vc[c] for c in all_crys]).items()}
        for how, sc in score_all(base_txt, cry_txt).items():
            h, n_h = hard[how]
            res[how] = {"MRR": mrr(ranks(sc, true_idx)), "hard_neg_MRR": h}
        res["n_hard_neg_materials"] = n_h
        # typing: nearest of the 23 TBox classes is "substance"
        classes = [v for _, v in sorted(json.load(open(os.path.join(HERE, "data", "battgpt_ont", "concept_names.json"))).items(), key=lambda kv: int(kv[0]))]
        d = dist(enc(base_txt), enc(classes))
        res["typing_H@1"] = round((d.argmax(1) == classes.index("substance")).float().mean().item(), 4)
        # leak probes (test sentences only)
        if variant == "no_geometry":
            probe_m = ["\n".join(l for l in vm[m].split("\n") if not l.startswith(SYM)) for m in mats]
            probe = score_all(probe_m, cry_txt)
            res["probe"] = {"what": "material symmetry lines removed", **{k: mrr(ranks(v, true_idx)) for k, v in probe.items()}}
        else:
            sites = [next((l for l in c.split("\n") if l.startswith("Sites:")), "") for c in cry_txt]
            shuf = ["\n".join([l for l in c.split("\n") if not l.startswith("Sites:")] + [sites[sites_perm[i]]])
                    for i, c in enumerate(cry_txt)]
            probe = score_all(base_txt, shuf)
            res["probe"] = {"what": "crystal Sites lines shuffled", **{k: mrr(ranks(v, true_idx)) for k, v in probe.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "h2h_results.json"))
    a = ap.parse_args()
    ents = load("entities.json")
    split = json.load(open(os.path.join(HERE, "data", "split.json")))
    n = len(split["test_crystals"])
    perm = list(range(n)); random.Random(0).shuffle(perm)
    H = lambda k: sum(1 / i for i in range(1, k + 1)) / k
    out = {"random_MRR_among_51": round(H(n), 4), "runs": {}}
    base = HierarchyTransformer.from_pretrained("Hui97/OnT-MiniLM-L12-galen", device="cpu"); base.eval()
    for variant in ("full", "no_geometry"):
        out["runs"][f"untrained galen | {variant}"] = evaluate(base, variant, ents, split, perm)
        for label, suffix in METHODS.items():
            path = os.path.join(RUNS, f"{variant}_split_{suffix}", "final")
            if not os.path.isdir(path):
                out["runs"][f"{label} | {variant}"] = "not trained"
                continue
            m = OntologyTransformer.from_pretrained(path); m.hit_model.eval()
            out["runs"][f"{label} | {variant}"] = evaluate(m, variant, ents, split, perm)
            print(f"done: {label} | {variant}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
