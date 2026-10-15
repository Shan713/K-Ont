"""Heart-disease K-Ont check, part 3: the report's datasets (CSV), from results.json, tests_rows/summary.json and
visuals.json. Output: data/heart/report/datasets/*.csv

    python heart/report_datasets.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "heart"))
import kont_heart as kh  # noqa: E402

R = os.path.join(kh.OUT, "report")
D = os.path.join(R, "datasets")
FEAT = {"raw": "Raw table", "raw+ont": "Raw + OnT", "raw+kont": "Raw + K-Ont", "raw+base": "Raw + plain encoder",
        "ont": "OnT only", "kont": "K-Ont only", "base": "Plain encoder only"}


def write(name, header, rows):
    os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(name, len(rows), "rows")


def cmp_name(s):
    a, b = s.split(" - ")
    return f"{FEAT[a]} vs {FEAT[b]}"


def main():
    res = json.load(open(os.path.join(kh.OUT, "results.json")))["summary"]
    write("main_results.csv", ["model", "features", "metric", "mean_pct", "sd_pct"],
          [[m, FEAT[s], met.upper(), round(v[met][0] * 100, 1), round(v[met][1] * 100, 1)]
           for m, sets in res["mean"].items() for s, v in sets.items() for met in ("acc", "auc", "f1")])
    write("main_paired.csv", ["model", "comparison", "metric", "diff_pts", "wins", "losses", "p"],
          [[m, cmp_name(c), met.upper(), round(v[met]["diff"] * 100, 1), v[met]["wins"], v[met]["losses"], v[met]["p"]]
           for m, cs in res["paired"].items() for c, v in cs.items() for met in ("acc", "auc")])

    ts = json.load(open(os.path.join(R, "tests_summary.json")))
    write("fewlabels.csv", ["labels_pct", "model", "features", "auc_pct", "acc_pct"],
          [[int(float(fr) * 100), m, FEAT[s], round(v["auc"] * 100, 1), round(v["acc"] * 100, 1)]
           for fr, ms in ts["mean"]["fewlabels"].items() for m, ss in ms.items() for s, v in ss.items()])
    write("fewlabels_paired.csv", ["labels_pct", "model", "comparison", "auc_diff_pts", "wins", "losses", "p"],
          [[int(float(fr) * 100), m, cmp_name(c), round(v["auc_diff"] * 100, 1), v["wins"], v["losses"], v["p"]]
           for fr, ms in ts["paired"]["fewlabels"].items() for m, cs in ms.items() for c, v in cs.items()])
    write("missing.csv", ["scenario", "model", "features", "auc_pct", "acc_pct"],
          [[sc, m, FEAT[s], round(v["auc"] * 100, 1), round(v["acc"] * 100, 1)]
           for sc, ms in ts["mean"]["missing"].items() for m, ss in ms.items() for s, v in ss.items()])
    write("missing_paired.csv", ["scenario", "model", "comparison", "auc_diff_pts", "wins", "losses", "p"],
          [[sc, m, cmp_name(c), round(v["auc_diff"] * 100, 1), v["wins"], v["losses"], v["p"]]
           for sc, ms in ts["paired"]["missing"].items() for m, cs in ms.items() for c, v in cs.items()])

    v = json.load(open(os.path.join(R, "visuals.json")))
    test = set(v["fold0_test_ids"])
    write("patient_map.csv", ["patient", "diagnosis", "seen_in_training", "representation", "x", "y"],
          [[pid, "Heart disease" if v["labels"][i] else "No heart disease", "No" if pid in test else "Yes",
            rep, mp["tsne"][i][0], mp["tsne"][i][1]]
           for rep, mp in v["maps"].items() for i, pid in enumerate(v["ids"])])
    checks = []
    for m in ("OnT", "K-Ont"):
        mm = v["models"][m]
        checks.append(["Unseen patients typed as \"patient\"", m, round(mm["patient_typed_patient_test"] * 100, 1), ""])
        for role, acc in mm["relation_accuracy_test"].items():
            if not role.endswith("(chance)"):
                checks.append([f"Relation: {role}", m, round(acc * 100, 1),
                               round(mm["relation_accuracy_test"][role + " (chance)"] * 100, 1)])
    for rep, a in v["knn_label_agreement_k10"].items():
        checks.append(["Same diagnosis among 10 nearest patients", rep, round(a * 100, 1), round(v["knn_chance"] * 100, 1)])
    for rep, mp in v["maps"].items():
        checks.append(["Distinct patient points (of 303)", rep, mp["distinct_points"], ""])
    write("checks.csv", ["check", "representation", "value", "chance"], checks)
    typed = [[m, cls, n] for m in ("OnT", "K-Ont") for cls, n in v["models"][m]["patient_typed_as"].items()]
    write("patient_typing.csv", ["model", "nearest_class", "patients"], typed)

    # hierarchy level of each concept, from the TBox's own subclass axioms
    parent = {}
    for r in kh.read_jsonl(os.path.join(kh.TBOX, "train.jsonl")):
        parent.setdefault(r["child"], r["parent"])
    def level(c, seen=()):
        return 0 if c not in parent or c in seen else 1 + level(parent[c], seen + (c,))
    concepts = list(v["models"]["OnT"]["concept_depth"])
    rows = [[c, level(c), v["models"]["OnT"]["concept_depth"][c], v["models"]["K-Ont"]["concept_depth"][c]]
            for c in concepts if c != "thing"]
    rows.append(["(patients, median)", max(r[1] for r in rows) + 1, v["models"]["OnT"]["patient_depth_median"],
                 v["models"]["K-Ont"]["patient_depth_median"]])
    write("depth.csv", ["concept", "hierarchy_level", "ont_distance_from_centre", "kont_distance_from_centre"], rows)

    ex_lines, vec, nb = [], [], []
    for pid, e in v["examples"].items():
        for k, line in enumerate(e["sentence"].split("\n")):
            ex_lines.append([pid, k, line])
        for d, x in enumerate(e["kont_vector_first24"]):
            vec.append([pid, "K-Ont (whole patient)", d, x])
        for d, x in enumerate(e["ont_vector_first24"]):
            vec.append([pid, "OnT (first concept: gender)", d, x])
        for rep, lst in e["neighbours"].items():
            for k, n in enumerate(lst):
                nb.append([pid, rep, k + 1, n["id"], "Heart disease" if n["label"] else "No heart disease",
                           " · ".join(s.split(": ", 1)[1] for s in n["summary"])])
    labs = {pid: ("Heart disease" if e["label"] else "No heart disease") for pid, e in v["examples"].items()}
    write("example_patients.csv", ["patient", "line", "text"], ex_lines)
    write("example_labels.csv", ["patient", "diagnosis", "seen_in_training"],
          [[pid, labs[pid], "No" if e["in_fold0_test"] else "Yes"] for pid, e in v["examples"].items()])
    write("example_vectors.csv", ["patient", "representation", "dim", "value"], vec)
    write("example_neighbours.csv", ["patient", "representation", "rank", "neighbour", "diagnosis", "summary"], nb)


if __name__ == "__main__":
    main()
