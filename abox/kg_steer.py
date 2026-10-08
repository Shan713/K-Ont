"""KG-steered generation, step 2: score generated structures against KG chemistry, and evaluate
whether picking by that score selects the right structure more often.

KG plausibility score (lower = more plausible; weights fixed a priori, NOT tuned on the test set):
  bond     mean robust z of each CrystalNN bond length vs the KG distribution for that element pair
           (|d - median| / (1.4826 MAD + 0.02 A)); a pair the KG never bonds (< 5 examples) counts z = 3
  coord    mean -log P(coordination number | element) from the KG (add-one smoothing)
  volume   |ln(V / V_KG)|, V_KG from the KG's per-element volumes
  clash    fraction of atoms with a neighbour closer than 0.85 x the KG's 1st-percentile bond for that pair
  total    bond + coord + 3 volume + 5 clash
"KG-plausible" = no clash, volume within +-25%, mean bond z < 2.

Evaluation uses the near-miss files (truth known): per material and prompt, the pool is the valid CIFs
with the right composition; reported: random pick (expected strict-match rate), KG top-1, best of pool,
and -- on the same 5-candidate pool CHGNet relaxed -- KG top-1 vs CHGNet's lowest-energy pick vs the
combination (relax only the KG-plausible candidates, keep the lowest energy).

    python abox/kg_steer.py --runs large,small_seeded,calib_large,calib_small --workers 16
"""
import argparse
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
_P = None


def priors():
    global _P
    if _P is None:
        _P = json.load(open(os.path.join(DATA, "kg_priors.json")))
    return _P


def kg_score(s):
    """KG plausibility of a pymatgen Structure: dict of components + total (lower is better)."""
    from pymatgen.analysis.local_env import CrystalNN
    P = priors()
    cnn = CrystalNN(weighted_cn=False, distance_cutoffs=None)
    zs, nlls, clash_atoms = [], [], 0
    for i, site in enumerate(s):
        a = site.specie.symbol
        try:
            nn = cnn.get_nn_info(s, i)
        except Exception:
            nn = []
        cn_counts = P["coordination"].get(a, {})
        tot = sum(cn_counts.values())
        nlls.append(-math.log((cn_counts.get(str(len(nn)), 0) + 1) / (tot + 13)))
        clash = False
        for n in nn:
            b = n["site"].specie.symbol
            d = site.distance(n["site"])
            st = P["bonds"].get("-".join(sorted((a, b))))
            if not st:
                zs.append(3.0)
                continue
            zs.append(min(abs(d - st["median"]) / (1.4826 * st["mad"] + 0.02), 10.0))
            if d < 0.85 * st["p01"]:
                clash = True
        clash_atoms += clash
    comp = s.composition.as_dict()
    v_kg = sum(n * P["volume_per_atom"].get(el, 0.0) for el, n in comp.items())
    vol = abs(math.log(s.volume / v_kg)) if v_kg > 0 else 1.0
    bond = sum(zs) / len(zs) if zs else 3.0
    coord = sum(nlls) / len(nlls) if nlls else 3.0
    clash_frac = clash_atoms / len(s)
    bond, coord, vol, clash_frac = float(bond), float(coord), float(vol), float(clash_frac)  # plain types for JSON
    return {"bond": round(bond, 3), "coord": round(coord, 3), "volume": round(vol, 3), "clash": round(clash_frac, 3),
            "total": round(bond + coord + 3 * vol + 5 * clash_frac, 3),
            "plausible": bool(clash_frac == 0 and vol < math.log(1.25) and bond < 2.0)}


def _score_file(path):
    import phaseC_generate as gen
    from pymatgen.core import Structure
    try:
        s = Structure.from_str(gen.postprocess(open(path).read()), fmt="cif")
        return path, kg_score(s)
    except Exception as e:
        return path, {"error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="large,small_seeded,calib_large,calib_small")
    ap.add_argument("--conditions", default="composition,oracle")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="kg_steer_eval.json")
    ap.add_argument("--save-scores", action="store_true", help="also write kg_scores_<run>.json (score per CIF)")
    a = ap.parse_args()
    report = {"weights": "bond + coord + 3 volume + 5 clash", "runs": {}}
    for run in a.runs.split(","):
        nm = json.load(open(os.path.join(DATA, f"phaseC_nearmiss_{run}.json")))
        gen_dir = nm["gen_dir"]
        jobs = []
        for m in nm["materials"]:
            for cond, rows in m["conditions"].items():
                if cond in a.conditions.split(","):
                    jobs += [os.path.join(DATA, gen_dir, m["id"], f"{cond}_{r['k']}.cif") for r in rows if r.get("composition_ok")]
        with ProcessPoolExecutor(a.workers) as ex:
            scores = dict(ex.map(_score_file, jobs, chunksize=4))
        if a.save_scores:
            json.dump({os.path.relpath(k, DATA).replace(os.sep, "/"): v for k, v in scores.items()},
                      open(os.path.join(DATA, f"kg_scores_{run}.json"), "w"))
        chg = {}
        cpath = os.path.join(DATA, f"phaseC_chgnet_{run}.json")
        if os.path.exists(cpath):
            for m in json.load(open(cpath))["materials"]:
                for cond, rows in m["conditions"].items():
                    chg[(m["id"], cond)] = [r for r in rows if "e" in r]
        res = {}
        for cond in a.conditions.split(","):
            n = rand = kg1 = best = 0
            plaus_tot = plaus_match = all_tot = all_match = 0
            c_n = c_kg = c_chg = c_kg_relax = c_comb = c_pool = c_relaxed = 0
            for m in nm["materials"]:
                rows = [r for r in m["conditions"].get(cond, []) if r.get("composition_ok")]
                pool = [(scores.get(os.path.join(DATA, gen_dir, m["id"], f"{cond}_{r['k']}.cif"), {}), r) for r in rows]
                pool = [(sc, r) for sc, r in pool if "total" in sc]
                if not pool:
                    continue
                n += 1
                rand += sum(r.get("strict", False) for _, r in pool) / len(pool)
                kg1 += min(pool, key=lambda x: x[0]["total"])[1].get("strict", False)
                best += any(r.get("strict", False) for _, r in pool)
                for sc, r in pool:
                    all_tot += 1; all_match += r.get("strict", False)
                    if sc["plausible"]:
                        plaus_tot += 1; plaus_match += r.get("strict", False)
                crows = chg.get((m["id"], cond), [])
                if crows:
                    by_k = {r["k"]: r for r in crows}
                    cpool = [(sc, r) for sc, r in pool if r["k"] in by_k]
                    if cpool:
                        c_n += 1
                        kpick = min(cpool, key=lambda x: x[0]["total"])[1]["k"]
                        c_kg += next(r.get("strict", False) for _, r in cpool if r["k"] == kpick)
                        c_kg_relax += by_k[kpick]["match_after_relax"]
                        c_chg += min(crows, key=lambda r: r["e"])["match_after_relax"]
                        # combined: relax only the KG-plausible candidates (all, if none pass), keep the lowest energy
                        keep = [by_k[r["k"]] for sc, r in cpool if sc["plausible"]] or [by_k[r["k"]] for _, r in cpool]
                        c_comb += min(keep, key=lambda r: r["e"])["match_after_relax"]
                        c_pool += len(cpool); c_relaxed += len(keep)
            res[cond] = {"materials": n, "random_pick": round(rand / n, 3), "kg_top1": round(kg1 / n, 3),
                         "best_of_pool": round(best / n, 3),
                         "plausible_share": round(plaus_tot / max(all_tot, 1), 3),
                         "match_rate_all": round(all_match / max(all_tot, 1), 3),
                         "match_rate_kg_plausible": round(plaus_match / max(plaus_tot, 1), 3)}
            if c_n:
                res[cond].update({"chgnet_pool_materials": c_n, "kg_top1_on_chgnet_pool_after_relax": round(c_kg_relax / c_n, 3),
                                  "chgnet_lowest_energy_after_relax": round(c_chg / c_n, 3),
                                  "kg_top1_on_chgnet_pool_unrelaxed": round(c_kg / c_n, 3),
                                  "kg_filter_then_chgnet_after_relax": round(c_comb / c_n, 3),
                                  "relaxations_needed_with_kg_filter": round(c_relaxed / max(c_pool, 1), 3)})
            print(run, cond, res[cond], flush=True)
        report["runs"][run] = res
    json.dump(report, open(os.path.join(DATA, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
