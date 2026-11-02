"""Is K-Ont's hyperbolic space meaningful? Checks + 2-D map data for the guide and the paper (Step 62).

Compares the trained K-Ont (default: the 5,000-material v2 run, family/role lines not in any sentence)
with the model it started from (OnT-MiniLM-L12-galen, never saw the battery graph). Everything is
measured with the model's own Poincare-ball geometry (curvature c, distance d, distance from origin d0).

  1 depth      ontology level vs d0: general classes near the centre, leaves further out, individuals at the edge
  2 order      every child is farther from the centre than its parent (TBox axioms; individual -> its type)
  3 typing     held-out individuals: the top class by OnT's own subsumption score is their true type
  4 branch     held-out crystals: their own structure family ranks first among the 9 family classes
               (a) plain distance to the class, (b) via the learned "has structure family" rotation;
               held-out materials: their battery role (positive / negative electrode) likewise
  5 neighbours the 10 nearest labelled crystals (materials) share the family (role) more often than chance
  6 faithful   ontology graph vs hyperbolic distance: Spearman of shortest-path vs d on the concept trees,
               and mean average precision of reconstructing the graph edges from distances (Nickel & Kiela 2017)

Map: tangent vectors at the origin (logmap0) -> 2-D by an uncentred SVD (the origin stays the origin) ->
each point's radius reset to its exact hyperbolic distance from the origin -> expmap0 into the unit disk.
So the RADIUS in the figure is exact, only the ANGLE is a projection.

    .venv\\Scripts\\python abox/hyperbolic_proof.py
    .venv-crystallm\\Scripts\\python abox/hyperbolic_plots.py      (figures; matplotlib lives there)
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
sys.path.insert(0, os.path.join(HERE, "abox"))

import torch  # noqa: E402
from ont.hit import HierarchyTransformer  # noqa: E402
from ont.model import OntologyTransformer  # noqa: E402
from phase5_checks import FAMILY_CLASS, FAMILY_READABLE, ROLE_SENT  # noqa: E402

ROLE_CLASS = {"PositiveElectrode": "positive electrode role", "NegativeElectrode": "negative electrode role"}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/runs/full_grow5000_v2")
    ap.add_argument("--abox-dir", default="data/battgpt_abox_grow_5000_v2")
    ap.add_argument("--split", default="data/split_grow_5000.json")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--out", default="data/hyperbolic_proof")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load = lambda f: json.load(open(os.path.join(a.abox_dir, f), encoding="utf-8"))
    ents, vm, vc, ve = load("entities.json"), load(f"verbalizations_materials_{a.variant}.json"), \
        load(f"verbalizations_crystals_{a.variant}.json"), load("verbalizations_elements.json")
    split = json.load(open(a.split))
    test_m, test_c = set(split["test_materials"]), set(split["test_crystals"])
    classes = [v for _, v in sorted(json.load(open(os.path.join(a.run, "data", "concept_names.json"))).items(),
                                    key=lambda kv: int(kv[0]))]
    cfg = json.load(open(os.path.join(a.run, "step_times.json")))

    after = OntologyTransformer.from_pretrained(os.path.join(a.run, "final"))
    after.hit_model.eval()
    lam = after.best_lambda or 0.0
    dev = next(after.hit_model.parameters()).device
    before = HierarchyTransformer.from_pretrained(cfg.get("base_model", "Hui97/OnT-MiniLM-L12-galen"), device=dev)
    before.eval()
    man = after.manifold
    c = float(man.c)

    # TBox: atomic child -> parent among the class names (DeepOnto's train.jsonl, as phase5 check I)
    tbox = []
    for line in open(os.path.join(HERE, "data", "battgpt_ont", "train.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["child"] in classes and r["parent"] in classes and (r["child"], r["parent"]) not in tbox:
            tbox.append((r["child"], r["parent"]))
    parent_of = defaultdict(list)
    for ch, pa in tbox:
        parent_of[ch].append(pa)

    def level(cl):  # 0 = root class (no parent), 1 = family / role, 2 = leaf family
        return 0 if not parent_of[cl] else 1 + max(level(p) for p in parent_of[cl])

    lvl = {cl: level(cl) for cl in classes}

    mats, crys, els = sorted(vm), sorted(vc), sorted(ve)
    fam_of = {x: ents["crystals"][x].get("structure_family") for x in crys}
    role_of = {m: ents["materials"][m].get("battery_role") for m in mats}

    def enc(model, sents):
        with torch.no_grad():
            return model.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()

    def pdist(x, y, chunk=256):
        return torch.cat([man.dist(x[i:i + chunk].unsqueeze(1), y.unsqueeze(0)) for i in range(0, len(x), chunk)])

    def score(child, parents):  # OnT's subsumption score: -(d + lambda * (|parent| - |child|))
        return -(pdist(child, parents) + lam * (man.dist0(parents).unsqueeze(0) - man.dist0(child).unsqueeze(1)))

    report, maps = {"run": a.run, "base_model": cfg.get("base_model"), "curvature": c,
                    "ball_radius": 1 / math.sqrt(c), "lambda": lam, "k": a.k}, {}
    for tag, model in (("base", before), ("kont", after)):
        torch.manual_seed(0)
        C, M, X, E = enc(model, classes), enc(model, [vm[m] for m in mats]), enc(model, [vc[x] for x in crys]), \
            enc(model, [ve[e] for e in els])
        d0 = {k: man.dist0(v).cpu().numpy() for k, v in (("C", C), ("M", M), ("X", X), ("E", E))}
        r = {}

        # 1 depth
        by_level = defaultdict(list)
        for i, cl in enumerate(classes):
            by_level[f"class level {lvl[cl]}"].append(float(d0["C"][i]))
        for k, name in (("E", "elements"), ("M", "materials"), ("X", "crystals")):
            by_level[f"individuals: {name}"] = d0[k].tolist()
        r["depth"] = {k: {"median": round(float(np.median(v)), 3), "min": round(float(np.min(v)), 3),
                          "max": round(float(np.max(v)), 3), "n": len(v)} for k, v in by_level.items()}
        r["depth_per_class"] = {cl: round(float(d0["C"][i]), 3) for i, cl in enumerate(classes)}
        # depth rank correlation over the family / role trees + individuals (level 3)
        tree = [cl for cl in classes if lvl[cl] > 0 or any(cl in p for p in parent_of.values())]
        xs = [lvl[cl] for cl in tree] + [3] * 3
        ys = [float(d0["C"][classes.index(cl)]) for cl in tree] + [float(np.median(d0[k])) for k in ("E", "M", "X")]
        r["depth_spearman_trees_plus_individuals"] = round(spearman(xs, ys), 3)

        # 2 order
        ok = [d0["C"][classes.index(ch)] > d0["C"][classes.index(pa)] for ch, pa in tbox]
        r["order_tbox_child_farther"] = {"held": int(sum(ok)), "of": len(ok),
                                         "violations": [[ch, pa] for (ch, pa), o in zip(tbox, ok) if not o]}
        inst = {}
        for k, items, typ in (("M", mats, "substance"), ("X", crys, "crystal structure"), ("E", els, "chemical element")):
            inst[typ] = round(float((d0[k] > d0["C"][classes.index(typ)]).mean()), 4)
        fam_idx = [(i, FAMILY_CLASS[FAMILY_READABLE[fam_of[x]]]) for i, x in enumerate(crys) if fam_of[x]]
        inst["crystal farther than its family class"] = round(float(np.mean(
            [d0["X"][i] > d0["C"][classes.index(f)] for i, f in fam_idx])), 4)
        r["order_individual_farther_than_type"] = inst

        # 3 typing, held-out
        typing = {}
        for k, emb, items, held, typ in (("M", M, mats, test_m, "substance"), ("X", X, crys, test_c, "crystal structure")):
            idx = [i for i, it in enumerate(items) if it in held]
            top = score(emb[idx], C).argmax(1).cpu().numpy()
            typing[typ] = {"n": len(idx), "top1": round(float(np.mean([classes[t] == typ for t in top])), 4)}
        top = score(E, C).argmax(1).cpu().numpy()
        typing["chemical element"] = {"n": len(els), "top1": round(float(np.mean([classes[t] == "chemical element"
                                                                                   for t in top])), 4)}
        r["typing_heldout_top1"] = typing

        # 4 branch: own family / role first, held-out only
        fam_keys = sorted({fam_of[x] for x in crys if fam_of[x]})
        fam_cls = [FAMILY_CLASS[FAMILY_READABLE[f]] for f in fam_keys]
        Fc = C[[classes.index(f) for f in fam_cls]]
        hx = [i for i, x in enumerate(crys) if x in test_c and fam_of[x]]
        tf = np.array([fam_keys.index(fam_of[crys[i]]) for i in hx])
        branch = {}

        def acc(pred, t, keys):
            rec = [float((pred[t == k] == k).mean()) for k in range(len(keys)) if (t == k).any()]
            maj = np.bincount(t).max() / len(t)
            return {"n": int(len(t)), "acc": round(float((pred == t).mean()), 4), "balanced_acc": round(float(np.mean(rec)), 4),
                    "majority": round(float(maj), 4), "chance_balanced": round(1 / len(rec), 4)}

        pred_plain = (-pdist(X[hx], Fc)).argmax(1).cpu().numpy()
        branch["family_plain_distance_to_class"] = acc(pred_plain, tf, fam_keys)
        conf = np.zeros((len(fam_keys), len(fam_keys)), int)
        if tag == "kont":
            fam_sent = [f"Structure family: {FAMILY_READABLE[f]}" for f in fam_keys]
            Fr = torch.tensor(after.encode_existence(["has structure family"] * len(fam_keys), fam_sent), device=dev)
            pred_rot = score(X[hx], Fr).argmax(1).cpu().numpy()
            branch["family_via_relation_rotation"] = acc(pred_rot, tf, fam_keys)
            for t_, p_ in zip(tf, pred_rot):
                conf[t_, p_] += 1
            role_keys = ["PositiveElectrode", "NegativeElectrode"]
            Rr = torch.tensor(after.encode_existence(["belongs to electrode"] * 2, [ROLE_SENT[k] for k in role_keys]), device=dev)
            hm = [i for i, m in enumerate(mats) if m in test_m and role_of[m] in role_keys]
            tr = np.array([role_keys.index(role_of[mats[i]]) for i in hm])
            branch["role_via_relation_rotation"] = acc(score(M[hm], Rr).argmax(1).cpu().numpy(), tr, role_keys)
        else:
            for t_, p_ in zip(tf, pred_plain):
                conf[t_, p_] += 1
        role_keys = ["PositiveElectrode", "NegativeElectrode"]
        Rc = C[[classes.index(ROLE_CLASS[k]) for k in role_keys]]
        hm = [i for i, m in enumerate(mats) if m in test_m and role_of[m] in role_keys]
        tr = np.array([role_keys.index(role_of[mats[i]]) for i in hm])
        branch["role_plain_distance_to_class"] = acc((-pdist(M[hm], Rc)).argmax(1).cpu().numpy(), tr, role_keys)
        branch["family_confusion"] = {"labels": fam_keys, "matrix": conf.tolist(),
                                      "scored_by": "relation rotation" if tag == "kont" else "plain distance"}
        r["branch_heldout"] = branch

        # 5 neighbours: held-out queries, labelled pool (train + test, self excluded)
        def purity(emb, items, lab, held):
            li = [i for i, it in enumerate(items) if lab[it]]
            q = [n for n, i in enumerate(li) if items[i] in held]
            D = pdist(emb[li][q], emb[li]).cpu().numpy()
            for row, n in enumerate(q):
                D[row, n] = np.inf
            nn = np.argsort(D, 1)[:, :a.k]
            labs = np.array([lab[items[i]] for i in li])
            same = float((labs[nn] == labs[q][:, None]).mean())
            _, cnt = np.unique(labs, return_counts=True)
            return {"n_queries": len(q), "pool": len(li), "same_label_in_kNN": round(same, 4),
                    "chance": round(float(((cnt / cnt.sum()) ** 2).sum()), 4)}

        r["neighbours_heldout"] = {"crystal_family": purity(X, crys, fam_of, test_c),
                                   "material_role": purity(M, mats, role_of, test_m),
                                   "material_chemical_system": purity(M, mats, {m: ents["materials"][m]["chemsys"] for m in mats}, test_m)}

        # 6 graph faithfulness on the concept trees (undirected TBox edges)
        nodes = sorted({x for e in tbox for x in e})
        adj = defaultdict(set)
        for ch, pa in tbox:
            adj[ch].add(pa)
            adj[pa].add(ch)

        def bfs(s):
            dist, dq = {s: 0}, deque([s])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            return dist

        Dn = pdist(C[[classes.index(n) for n in nodes]], C[[classes.index(n) for n in nodes]]).cpu().numpy()
        sp, hd, aps = [], [], []
        for i, u in enumerate(nodes):
            bu = bfs(u)
            for j in range(i + 1, len(nodes)):
                if nodes[j] in bu:
                    sp.append(bu[nodes[j]])
                    hd.append(Dn[i, j])
            order = [nodes[j] for j in np.argsort(Dn[i]) if j != i]
            hits, prec = 0, []
            for rank, v in enumerate(order, 1):
                if v in adj[u]:
                    hits += 1
                    prec.append(hits / rank)
            aps.append(np.mean(prec))
        rng = np.random.default_rng(0)
        rand_ap = []
        for _ in range(200):
            perm = rng.permutation(len(nodes))
            ap_ = []
            for i, u in enumerate(nodes):
                order = [nodes[j] for j in perm if j != i]
                hits, prec = 0, []
                for rank, v in enumerate(order, 1):
                    if v in adj[u]:
                        hits += 1
                        prec.append(hits / rank)
                ap_.append(np.mean(prec))
            rand_ap.append(np.mean(ap_))
        r["graph_faithfulness"] = {"nodes": len(nodes), "edges": len(tbox), "pairs_connected": len(sp),
                                   "spearman_shortest_path_vs_distance": round(spearman(sp, hd), 3),
                                   "MAP_edge_reconstruction": round(float(np.mean(aps)), 3),
                                   "MAP_random": round(float(np.mean(rand_ap)), 3)}
        report[tag] = r

        # 2-D map: each point keeps its exact distance from the origin (rho = d0); only the angles are fitted, so
        # that the 2-D hyperbolic distances (same curvature) match the model's pairwise distances (Sammon stress,
        # weight 1/d^2: neighbourhoods first). Points: all classes and elements, every family-labelled crystal,
        # every role-labelled material and 600 unlabelled materials.
        rs = np.random.default_rng(0)
        pick = {"C": np.arange(len(classes)), "E": np.arange(len(els)),
                "X": np.array([i for i, x in enumerate(crys) if fam_of[x]]),
                "M": np.array(sorted([i for i, m in enumerate(mats) if role_of[m]] +
                                     rs.choice([i for i, m in enumerate(mats) if not role_of[m]], 600, replace=False).tolist()))}
        full = {"C": C, "E": E, "X": X, "M": M}
        P = torch.cat([full[k][pick[k]] for k in "CEXM"])
        Dt = pdist(P, P)
        rho = man.dist0(P)
        sc_ = math.sqrt(c)
        T0 = man.logmap0(P)
        _, _, Vt = torch.linalg.svd(T0 / T0.norm(dim=1, keepdim=True), full_matrices=False)
        th0 = torch.atan2(T0 @ Vt[1], T0 @ Vt[0])
        iu = torch.triu_indices(len(P), len(P), 1, device=dev)
        dt = Dt[iu[0], iu[1]].clamp_min(1e-3)
        ch, sh = torch.cosh(sc_ * rho), torch.sinh(sc_ * rho)

        def d2(th):
            cosd = ch[iu[0]] * ch[iu[1]] - sh[iu[0]] * sh[iu[1]] * torch.cos(th[iu[0]] - th[iu[1]])
            return torch.acosh(cosd.clamp_min(1 + 1e-7)) / sc_

        best = None
        for restart in range(4):
            th = (th0 if restart == 0 else torch.rand(len(P), device=dev) * 2 * math.pi).clone().requires_grad_(True)
            opt = torch.optim.Adam([th], lr=0.05)
            for _ in range(1500):
                opt.zero_grad()
                loss = (((d2(th) - dt) / dt) ** 2).mean()
                loss.backward()
                opt.step()
            if best is None or loss.item() < best[0]:
                best = (loss.item(), th.detach())
        th = best[1]
        with torch.no_grad():
            Dm = torch.zeros_like(Dt)
            Dm[iu[0], iu[1]] = d2(th)
            Dm = Dm + Dm.T
            big = torch.eye(len(P), device=dev) * 1e9
            nn_true = (Dt + big).topk(a.k, largest=False).indices
            nn_map = (Dm + big).topk(a.k, largest=False).indices
            keep = np.mean([len(set(nn_true[i].tolist()) & set(nn_map[i].tolist())) / a.k for i in range(len(P))])
        r["map_quality"] = {"points": len(P), "sammon_stress": round(best[0], 4),
                            "spearman_pairwise_distance": round(spearman(dt.cpu().numpy()[::7], d2(th).cpu().numpy()[::7]), 3),
                            f"{a.k}NN_preserved": round(float(keep), 3), "radius": "exact (distance from the origin)"}
        xy = (torch.tanh(sc_ * rho / 2).unsqueeze(1) * torch.stack([torch.cos(th), torch.sin(th)], 1)).cpu().numpy()
        o, maps[tag] = 0, {}
        for k in "CEXM":
            maps[tag][k], o = xy[o:o + len(pick[k])], o + len(pick[k])
        maps[tag].update({f"{k}_idx": pick[k] for k in "EXM"})
        maps[tag].update({f"d0_{k}": v for k, v in d0.items()})

    np.savez_compressed(os.path.join(a.out, "map2d.npz"), **{f"{t}_{k}": v for t, m in maps.items() for k, v in m.items()})
    json.dump({"classes": classes, "class_level": lvl, "tbox": tbox, "materials": [m.split("/")[-1] for m in mats],
               "material_formula": [ents["materials"][m]["formula"] for m in mats],
               "material_role": [role_of[m] for m in mats], "material_test": [m in test_m for m in mats],
               "crystal_family": [fam_of[x] for x in crys], "crystal_test": [x in test_c for x in crys],
               "elements": [ents["elements"][e]["symbol"] for e in els]},
              open(os.path.join(a.out, "map_labels.json"), "w", encoding="utf-8"))
    json.dump(report, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
