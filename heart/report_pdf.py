"""Heart-disease K-Ont check, part 4: the PDF report.

Charts with matplotlib (inline SVG), an A4 print-styled HTML document, printed to PDF by headless Chrome.
Reads data/heart/results.json and data/heart/report/{datasets/*.csv, visuals.json}.

    .venv-crystallm/Scripts/python.exe heart/report_pdf.py
Output: data/heart/report/KOnt_vs_OnT_heart_disease_report.pdf (+ .html)
"""
import csv
import html
import io
import json
import os
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "heart")
R = os.path.join(OUT, "report")
D = os.path.join(R, "datasets")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

C = {"Raw table": "#8a8f98", "Raw + OnT": "#3b6fb6", "Raw + K-Ont": "#d9762b", "Raw + plain encoder": "#4b9b6e",
     "OnT only": "#3b6fb6", "K-Ont only": "#d9762b", "Plain encoder only": "#4b9b6e",
     "OnT": "#3b6fb6", "K-Ont": "#d9762b", "OnT (5 concepts)": "#3b6fb6", "Untrained encoder": "#4b9b6e"}
DIAG = {"No heart disease": "#2a9d8f", "Heart disease": "#c8553d"}
FEATS = ["Raw table", "Raw + OnT", "Raw + K-Ont", "Raw + plain encoder"]
EMBS = ["OnT only", "K-Ont only", "Plain encoder only"]
MODELS = ["LogReg", "RandomForest", "SVM", "GradBoost", "MLP"]
INK, MUTED, GRID = "#1d2430", "#5b6472", "#e3e6ea"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#b9bec6", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK})


def rows(name):
    return list(csv.DictReader(open(os.path.join(D, name), encoding="utf-8")))


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def esc(x):
    return html.escape(str(x))


def signed(v):
    v = float(v)
    return ("+" if v > 0 else "−" if v < 0 else "±") + f"{abs(v):.1f}"


def pfmt(p):
    p = float(p)
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ charts
def dotplot(data, series, key, title, xlabel="ROC-AUC (%)", height=2.15):
    fig, ax = plt.subplots(figsize=(3.4, height))
    vals = [float(r[key]) for r in data if r["features"] in series]
    for i, m in enumerate(MODELS):
        rs = [r for r in data if r["model"] == m and r["features"] in series]
        xs = [float(r[key]) for r in rs]
        if xs:
            ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=2, zorder=1)
        for r in rs:
            j = series.index(r["features"])
            ax.scatter(float(r[key]), i + (j - (len(series) - 1) / 2) * 0.09, s=34, color=C[r["features"]],
                       edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis()
    ax.set_xlim(np.floor(min(vals)) - 0.5, np.ceil(max(vals)) + 0.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return svg(fig)


def legend_html(names):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{C[n]}"></i>{esc(n)}</span>' for n in names) + "</div>"


def chart_typing(typ):
    fig, ax = plt.subplots(figsize=(6.6, 1.35))
    for i, m in enumerate(["OnT", "K-Ont"]):
        acc = 0
        for r in [x for x in typ if x["model"] == m]:
            n = int(r["patients"])
            col = C["K-Ont"] if r["nearest_class"] == "patient" else "#c8553d"
            ax.barh(i, n, left=acc, height=0.55, color=col, edgecolor="white", linewidth=1.5)
            if n > 40:
                ax.text(acc + 6, i, f'{r["nearest_class"]} ({n})', va="center", color="white", fontsize=8.5, fontweight="bold")
            acc += n
    ax.set_yticks([0, 1], ["OnT", "K-Ont"])
    ax.invert_yaxis()
    ax.set_xlim(0, 303)
    ax.set_xlabel("patients (of 303)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg(fig)


def chart_depth(dep):
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    lv = sorted({int(r["hierarchy_level"]) for r in dep})
    top = max(lv)
    for r in dep:
        y = int(r["hierarchy_level"])
        ax.scatter(float(r["ont_distance_from_centre"]), y - 0.12, s=26, color=C["OnT"], edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(float(r["kont_distance_from_centre"]), y + 0.12, s=26, color=C["K-Ont"], edgecolor="white", lw=0.6, zorder=3)
    ax.set_yticks(lv, [("patients (median)" if l == top else f"level {l}") for l in lv])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("hyperbolic distance from the centre of the ball")
    ax.scatter([], [], color=C["OnT"], label="OnT")
    ax.scatter([], [], color=C["K-Ont"], label="K-Ont")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return svg(fig)


def chart_maps(pm):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
    names = {"OnT (5 concepts)": "OnT (5 concepts)", "K-Ont": "K-Ont", "Untrained encoder": "Plain encoder"}
    for ax, rep in zip(axs, names):
        rs = [r for r in pm if r["representation"] == rep]
        for diag, col in DIAG.items():
            seen = [r for r in rs if r["diagnosis"] == diag and r["seen_in_training"] == "Yes"]
            unseen = [r for r in rs if r["diagnosis"] == diag and r["seen_in_training"] == "No"]
            ax.scatter([float(r["x"]) for r in seen], [float(r["y"]) for r in seen], s=7, color=col, alpha=0.8, lw=0)
            ax.scatter([float(r["x"]) for r in unseen], [float(r["y"]) for r in unseen], s=11, facecolor="white",
                       edgecolor=col, lw=0.9)
        ax.set_title(names[rep], loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    return svg(fig)


def chart_vectors(vec, pid):
    reps = ["OnT (first concept: gender)", "K-Ont (whole patient)"]
    fig, ax = plt.subplots(figsize=(6.8, 1.0))
    M = np.array([[float(r["value"]) for r in sorted([x for x in vec if x["patient"] == pid and x["representation"] == rp],
                                                      key=lambda x: int(x["dim"]))] for rp in reps])
    m = np.abs(M).max()
    ax.imshow(M, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto")
    ax.set_yticks([0, 1], ["OnT (1 of 5 concept vectors)", "K-Ont (the patient)"])
    ax.set_xticks(range(0, 24, 4), [str(i + 1) for i in range(0, 24, 4)])
    ax.set_xlabel("first 24 of 384 numbers (red positive, blue negative)")
    for s in ax.spines.values():
        s.set_visible(False)
    return svg(fig)


def chart_relations(checks):
    rel = [r for r in checks if r["check"].startswith("Relation:")]
    roles = list(dict.fromkeys(r["check"].replace("Relation: has ", "") for r in rel))
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    y = np.arange(len(roles))
    for k, (m, off) in enumerate((("OnT", -0.18), ("K-Ont", 0.18))):
        vals = [float(next(r["value"] for r in rel if r["representation"] == m and r["check"].endswith(ro))) for ro in roles]
        ax.barh(y + off, vals, height=0.34, color=C[m], label=m)
    chance = [float(next(r["chance"] for r in rel if r["check"].endswith(ro))) for ro in roles]
    for i, c in enumerate(chance):
        ax.plot([c, c], [i - 0.4, i + 0.4], color=INK, lw=1.2, ls=(0, (2, 2)))
    ax.set_yticks(y, roles)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("unseen patients whose own value is nearest (%) — dashed: chance")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    return svg(fig)


def chart_few(few):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharey=False)
    axs = axs.ravel()
    pts = [10, 20, 50, 100]
    for ax, m in zip(axs, ["All models (mean)"] + MODELS):
        for f in FEATS:
            if m.startswith("All"):
                ys = [np.mean([float(r["auc_pct"]) for r in few if r["features"] == f and int(r["labels_pct"]) == p]) for p in pts]
            else:
                ys = [float(next(r["auc_pct"] for r in few if r["features"] == f and r["model"] == m and int(r["labels_pct"]) == p)) for p in pts]
            ax.plot(range(4), ys, marker="o", ms=3.2, lw=1.8 if f == "Raw + K-Ont" else 1.3, color=C[f], solid_capstyle="round")
        ax.set_xticks(range(4), [str(p) for p in pts])
        ax.set_title(m, loc="left", fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for ax in axs:
        ax.set_ylabel("ROC-AUC (%)")
    for ax in axs[3:]:
        ax.set_xlabel("% of training labels")
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    return svg(fig)


def pipeline_fig():
    fig, ax = plt.subplots(figsize=(7.2, 1.25))
    ax.axis("off")
    boxes = ["Heart KG (OWL)\n303 patients\n+ 32-class ontology", "Remove the\ndiagnosis link\nfrom everything",
             "OnT: classes only\nK-Ont: + training\npatients, per fold", "Patient vectors\n384 → 16 (PCA)\n+ 13-column table",
             "5 classifiers\n10 folds\nAUC, accuracy, F1"]
    w, gap = 0.17, 0.0375
    for i, t in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.12), w, 0.76, fc="#fbeee4" if i == 2 else "#f4f6f8", ec="#b9bec6", lw=0.8))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", fontsize=7.4, color=INK)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.5), xytext=(x + w + 0.004, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.set_xlim(-0.005, 1.0)
    ax.set_ylim(0, 1)
    return svg(fig)


# ------------------------------------------------------------------ tables
def table(head, body, num=()):
    h = "".join(f'<th class="{"num" if i in num else ""}">{esc(x)}</th>' for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>" for r in body)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def sig_cell(diff, p):
    cls = ("good" if float(diff) > 0 else "bad") if float(p) < 0.05 else ""
    return f'<span class="{cls}">{signed(diff)}</span>'


# ------------------------------------------------------------------ document
def main():
    main_r, main_p = rows("main_results.csv"), rows("main_paired.csv")
    few, few_p = rows("fewlabels.csv"), rows("fewlabels_paired.csv")
    mis, mis_p = rows("missing.csv"), rows("missing_paired.csv")
    pm, checks, typ, dep = rows("patient_map.csv"), rows("checks.csv"), rows("patient_typing.csv"), rows("depth.csv")
    exs, exl, vec, nb = rows("example_patients.csv"), rows("example_labels.csv"), rows("example_vectors.csv"), rows("example_neighbours.csv")
    auc = [r for r in main_r if r["metric"] == "AUC"]

    def pmean(rs, comp, metric="AUC", key="diff_pts"):
        x = [float(r[key]) for r in rs if r["comparison"] == comp and r.get("metric", metric) == metric]
        return np.mean(x), x

    k_vs_o, k_vs_o_all = pmean(main_p, "K-Ont only vs OnT only")
    k_vs_o_sig = sum(1 for r in main_p if r["comparison"] == "K-Ont only vs OnT only" and r["metric"] == "AUC"
                     and float(r["p"]) < 0.05 and float(r["diff_pts"]) > 0)
    full = pmean(main_p, "Raw + K-Ont vs Raw table")[0]
    few10 = np.mean([float(r["auc_diff_pts"]) for r in few_p if r["comparison"] == "Raw + K-Ont vs Raw table" and int(r["labels_pct"]) == 10])
    miss3 = [r for r in mis_p if r["comparison"] == "Raw + K-Ont vs Raw table" and r["scenario"] == "hide cp+thal+ca"]
    miss3m = np.mean([float(r["auc_diff_pts"]) for r in miss3])
    miss3sig = sum(1 for r in miss3 if float(r["p"]) < 0.05 and float(r["auc_diff_pts"]) > 0)
    typed_k = next(float(r["value"]) for r in checks if r["check"].startswith("Unseen") and r["representation"] == "K-Ont")
    distinct = {r["representation"]: r["value"] for r in checks if r["check"].startswith("Distinct")}
    ont_typing = ", ".join(f'{r["nearest_class"]} ({r["patients"]})' for r in typ if r["model"] == "OnT")

    # main means table
    main_tab = table(["Model"] + FEATS + EMBS,
                     [[m] + [f'{float(next(r["mean_pct"] for r in auc if r["model"] == m and r["features"] == f)):.1f}' for f in FEATS + EMBS]
                      for m in MODELS], num=range(1, 8))
    acc_tab = table(["Model"] + FEATS + EMBS,
                    [[m] + [f'{float(next(r["mean_pct"] for r in main_r if r["metric"] == "ACC" and r["model"] == m and r["features"] == f)):.1f}' for f in FEATS + EMBS]
                     for m in MODELS], num=range(1, 8))
    comps = ["K-Ont only vs OnT only", "Raw + K-Ont vs Raw + OnT", "Raw + K-Ont vs Raw table", "Raw + OnT vs Raw table",
             "Raw + K-Ont vs Raw + plain encoder", "K-Ont only vs Plain encoder only"]
    paired_tab = table(["Comparison"] + MODELS,
                       [[c] + [sig_cell(r["diff_pts"], r["p"]) + f' <span class="muted">({r["wins"]}/{r["losses"]}, p {pfmt(r["p"])})</span>'
                               for m in MODELS for r in main_p if r["comparison"] == c and r["model"] == m and r["metric"] == "AUC"]
                        for c in comps])
    few_tab = table(["Labels"] + MODELS,
                    [[f"{p}%"] + [sig_cell(r["auc_diff_pts"], r["p"]) + f' <span class="muted">(p {pfmt(r["p"])})</span>'
                                  for m in MODELS for r in few_p if r["comparison"] == "Raw + K-Ont vs Raw table" and r["model"] == m and int(r["labels_pct"]) == p]
                     for p in (10, 20, 50, 100)])
    mis_tab = table(["Hidden columns", "Comparison"] + MODELS,
                    [[{"hide cp+thal": "chest pain, thallium", "hide cp+thal+ca": "chest pain, thallium, vessels"}[sc], c] +
                     [sig_cell(r["auc_diff_pts"], r["p"]) + f' <span class="muted">(p {pfmt(r["p"])})</span>'
                      for m in MODELS for r in mis_p if r["scenario"] == sc and r["comparison"] == c and r["model"] == m]
                     for sc in ("hide cp+thal", "hide cp+thal+ca") for c in ("Raw + K-Ont vs Raw table", "Raw + K-Ont vs Raw + OnT", "Raw + K-Ont vs Raw + plain encoder")])
    checks_tab = table(["Check", "OnT", "K-Ont", "Plain encoder", "Chance"],
                       [[esc(ch)] + [esc(next((r["value"] + ("" if ch.startswith("Distinct") else "%") for r in checks if r["check"] == ch and r["representation"] in reps), "—"))
                                     for reps in (("OnT", "OnT (5 concepts)"), ("K-Ont",), ("Untrained encoder",))] +
                        [esc(next((r["chance"] + "%" for r in checks if r["check"] == ch and r["chance"]), "—"))]
                        for ch in dict.fromkeys(r["check"] for r in checks)], num=(1, 2, 3, 4))

    def example_block(pid):
        lab = next(r for r in exl if r["patient"] == pid)
        sent = "\n".join(r["text"] for r in sorted([r for r in exs if r["patient"] == pid], key=lambda r: int(r["line"])))
        conc = [r["text"].split(": ")[1] for r in exs if r["patient"] == pid and r["text"].split(":")[0] in ("Gender", "Chest pain", "Resting ECG", "ST slope", "Thallium scan")]
        nbt = ""
        for rep, nm in (("OnT (5 concepts)", "OnT"), ("K-Ont", "K-Ont")):
            rs = sorted([r for r in nb if r["patient"] == pid and r["representation"] == rep], key=lambda r: int(r["rank"]))
            same = sum(r["diagnosis"] == lab["diagnosis"] for r in rs)
            nbt += f'<div><h4>Nearest by {nm} <span class="muted">· {same} of 5 share the diagnosis</span></h4>' + table(
                ["#", "Patient", "Diagnosis", "Gender · chest pain · ECG · slope · thallium"],
                [[r["rank"], esc(r["neighbour"].replace("_", " ")), esc(r["diagnosis"]), esc(r["summary"])] for r in rs]) + "</div>"
        return f"""
<div class="example avoid">
  <h3>{esc(pid.replace("_", " "))} <span class="muted">· diagnosis {esc(lab["diagnosis"].lower())} (never shown to the encoders) · seen by K-Ont in training (fold 0): {esc(lab["seen_in_training"].lower())}</span></h3>
  <div class="ex2">
    <div><div class="label">What K-Ont reads: the patient as a knowledge-graph individual</div><pre>{esc(sent)}</pre></div>
    <div><div class="label">What standard OnT can use: five ontology concepts</div><div>{"".join(f'<span class="pill">{esc(c)}</span>' for c in conc)}</div>
      <p class="small">OnT has no entry for a patient, so it embeds each concept separately (5 × 384 numbers). Age, blood pressure, cholesterol, heart rate, ST depression and vessels have no concept and are lost. K-Ont embeds the whole record as one point of 384 numbers.</p>
      <div class="label">The vectors (first 24 numbers)</div>{chart_vectors(vec, pid)}</div>
  </div>
  <div class="two">{nbt}</div>
</div>"""

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>K-Ont vs OnT on heart disease</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; color: {INK}; font-size: 9pt; line-height: 1.42; margin: 0; }}
h1 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 25pt; line-height: 1.15; margin: 0 0 6pt; }}
h2 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 15pt; margin: 16pt 0 6pt; border-bottom: 1px solid #d5d9df; padding-bottom: 3pt; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 10pt 0 4pt; break-after: avoid; }}
h4 {{ font-size: 9pt; margin: 6pt 0 3pt; }}
p {{ margin: 0 0 6pt; }}
.muted {{ color: {MUTED}; font-weight: 400; }}
.small {{ font-size: 8.4pt; color: {MUTED}; }}
.label {{ font-size: 7.8pt; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; margin: 4pt 0 2pt; }}
.cover {{ height: 255mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }}
.kicker {{ font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; color: #a5531d; }}
.summary {{ border: 1px solid #d5d9df; border-radius: 6px; padding: 10pt 12pt; background: #f7f8fa; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 8pt 0; }}
.tile {{ border: 1px solid #d5d9df; border-radius: 6px; padding: 7pt 8pt; }}
.tile .n {{ font-family: Cambria, Georgia, serif; font-size: 19pt; }}
.tile .t {{ font-size: 7.8pt; color: {MUTED}; line-height: 1.3; }}
.good {{ color: #1f7a4d; font-weight: 600; }} .bad {{ color: #b23b2a; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.7pt; margin: 4pt 0 8pt; font-variant-numeric: tabular-nums; break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #e3e6ea; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }}
th {{ color: {MUTED}; font-weight: 600; }}
td.num, th.num {{ text-align: right; }}
pre {{ font-family: Consolas, "Courier New", monospace; font-size: 7.8pt; background: #f4f6f8; border-radius: 4px; padding: 6pt 8pt; margin: 0; white-space: pre-wrap; }}
.pill {{ display: inline-block; border: 1px solid #c9ced6; border-radius: 99px; padding: 1pt 7pt; margin: 2pt 3pt 2pt 0; font-size: 8.2pt; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }}
.figs2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; }}
.fig {{ break-inside: avoid; margin: 4pt 0 8pt; }}
.fig svg, .example svg {{ width: 100%; height: auto; }}
svg text {{ font-family: "Segoe UI", Calibri, Arial, sans-serif !important; }}
.ex2 {{ display: grid; grid-template-columns: 34% 1fr; gap: 12pt; }}
.ex2 > div, .two > div {{ min-width: 0; }}
.cap {{ font-size: 8.1pt; color: {MUTED}; margin-top: 2pt; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4pt 12pt; font-size: 8pt; color: {MUTED}; margin: 2pt 0; }}
.legend i {{ display: inline-block; width: 8pt; height: 8pt; border-radius: 50%; margin-right: 4pt; vertical-align: -1pt; }}
.example {{ border: 1px solid #e3e6ea; border-radius: 6px; padding: 6pt 10pt; margin: 6pt 0 10pt; }}
.avoid {{ break-inside: avoid; }}
ul {{ margin: 0 0 6pt 14pt; padding: 0; }} li {{ margin-bottom: 3pt; }}
.pb {{ break-before: page; }}
</style></head><body>

<section class="cover">
  <div>
    <div class="kicker">K-Ont research update · heart-disease verification</div>
    <h1>Does K-Ont help machine learning more than OnT?</h1>
    <p style="font-size:11pt;color:{MUTED};max-width:150mm">A controlled comparison of standard OnT and K-Ont (OnT extended to knowledge-graph individuals) as patient representations for heart-disease prediction, with four classic machine-learning models and a neural network.</p>
  </div>
  <div class="summary">
    <h3 style="margin-top:0">Summary</h3>
    <div class="tiles">
      <div class="tile"><div class="n">{signed(k_vs_o)}</div><div class="t">AUC points, K-Ont vs OnT as the representation (mean of 5 models; {k_vs_o_sig} of 5 significant)</div></div>
      <div class="tile"><div class="n">{signed(full)}</div><div class="t">AUC points from adding K-Ont to the complete table (mean of 5 models)</div></div>
      <div class="tile"><div class="n">{signed(few10)}</div><div class="t">AUC points from adding K-Ont with only 10% of the labels</div></div>
      <div class="tile"><div class="n">{signed(miss3m)}</div><div class="t">AUC points from adding K-Ont when 3 columns are missing ({miss3sig} of 5 significant)</div></div>
    </div>
    <ul>
      <li><b>K-Ont is a better representation than OnT.</b> It places every patient, including unseen ones, under the ontology's "patient" class ({typed_k:.0f}% of unseen patients; OnT: {esc(ont_typing)}), keeps all {distinct.get("K-Ont")} patients distinct (OnT: {distinct.get("OnT (5 concepts)")}), and its vectors alone predict the diagnosis better than OnT's in every model.</li>
      <li><b>With the complete table, no embedding improves the classifiers.</b> The knowledge graph holds the same 13 facts as the table, and this ontology adds no medical knowledge beyond them.</li>
      <li><b>With few labels, the embeddings hurt.</b> Sixteen extra numbers per patient on 24 to 120 labelled patients make the classifiers overfit.</li>
      <li><b>When columns are missing from the table, the knowledge-graph embedding brings their information back</b> and every model improves significantly; K-Ont edges OnT because OnT cannot hold numeric facts. An untrained encoder reading the same patient sentence does about as well, so the gain comes from embedding each patient as an individual, which K-Ont makes possible, rather than from K-Ont's extra training.</li>
    </ul>
  </div>
  <div class="small">Shantharam P · K-Ont project · data: UCI Heart Disease (Cleveland, 303 patients) with the heart ontology and knowledge graph supplied by the guide · code: K-Ont repository, heart/</div>
</section>

<h2>1. The question</h2>
<p>Our battery-materials work showed that K-Ont, our extension of the Ontology Transformer (OnT) to knowledge-graph individuals, represents a knowledge graph better than standard OnT, which only embeds ontology classes. The guide asked whether this carries over to a practical task: if a machine-learning model is given ontology embeddings, does K-Ont improve its predictions more than OnT does? This report tests that on a second, independent domain: heart-disease diagnosis.</p>

<h2>2. Data</h2>
<ul>
  <li><b>Patients:</b> 303 patients from the UCI Cleveland heart-disease data, 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thallium scan) and a binary label (heart disease or not; 139 / 164).</li>
  <li><b>Ontology:</b> 32 classes (e.g. typical angina ⊑ chest pain ⊑ abstract ⊑ entity) and 5 relations (has gender, has chest pain, has resting ECG result, has ST slope, has thallium scan result).</li>
  <li><b>Knowledge graph:</b> the ontology plus the 303 patients as individuals, each with its 13 facts and a <i>hasDiagnosis</i> link. The graph lists patients in the same order as the table; ages were checked one by one. One ECG value, "ST wave anormality" (4 patients), has no class in the ontology and was mapped to its class "wave abnormality".</li>
</ul>

<h2>3. Method</h2>
<div class="fig">{pipeline_fig()}</div>
<h3>Representations compared</h3>
<ul>
  <li><b>Raw table</b> (baseline): the 13 features, categorical ones one-hot encoded, numeric ones standardised.</li>
  <li><b>Standard OnT:</b> OnT trained on the ontology alone, with OnT's own DeepOnto data preparation (30 subsumption axioms), 20 epochs. OnT has no representation for an individual, so a patient is the concatenation of the OnT vectors of its five categorical values.</li>
  <li><b>K-Ont:</b> OnT trained on the ontology plus the training patients as knowledge-graph individuals. Each patient is one sentence of its 13 facts (shown in section 4), trained as an instance of "patient" and linked by the five relations to its value concepts, with in-batch negatives; ontology rows are 15% of training; 3 epochs. A patient's representation is K-Ont's encoding of its sentence.</li>
  <li><b>Plain encoder</b> (control): the untrained starting model (OnT-MiniLM-galen) reading the same patient sentence. It separates the value of K-Ont's training from the value of simply encoding the patient as text.</li>
</ul>
<h3>Leakage control</h3>
<p>The diagnosis relation is removed from the ontology and never written into any sentence or training row. K-Ont is retrained inside every cross-validation fold on that fold's training patients only, so test patients are only ever encoded, never trained on. All vectors are mapped to the tangent space (logmap at the origin) and reduced to 16 numbers by PCA fitted on the training fold.</p>
<h3>Classifiers and evaluation</h3>
<p>Logistic regression, random forest (300–500 trees), RBF support-vector machine, histogram gradient boosting, and a multilayer perceptron (2 × 64 units, dropout 0.2, early stopping on an inner validation split, averaged over 3 seeds). Stratified 5-fold cross-validation repeated twice gives 10 folds, identical for every model and feature set. Metrics: ROC-AUC (main), accuracy and F1. Differences are paired fold by fold and tested with the Wilcoxon signed-rank test; "significant" means p &lt; 0.05.</p>

<h2 class="pb">4. One patient, two ways</h2>
<p>How each method turns the same knowledge-graph patient into numbers, and which other patients it considers most similar. Diagnoses are shown for reading only.</p>
{example_block("patient_001")}
{example_block("patient_002")}

<h2 class="pb">5. What the embeddings contain</h2>
<h3>Where each model places patients in the ontology</h3>
<div class="fig">{chart_typing(typ)}<div class="cap">For every patient, the ontology class nearest to it by OnT's own hierarchy score. The correct answer is "patient". K-Ont places all patients, including the 61 test patients of fold 0 it never trained on, under "patient"; standard OnT, which was never shown an individual, places them near "disease status".</div></div>
<h3>Hierarchy in hyperbolic space</h3>
<div class="fig">{chart_depth(dep)}<div class="cap">Distance from the centre of the Poincaré ball for every class, grouped by its level in the ontology (level 0 = top classes such as "entity"), and the median patient. In a well-formed hyperbolic embedding general classes sit near the centre and specific ones further out. K-Ont largely keeps this ordering and places patients, the most specific things, furthest out; OnT's classes are spread with little order.</div></div>
<div class="avoid"><h3>Map of all 303 patients</h3>
{'<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in DIAG.items()) + '<span><i style="background:white;border:1.5px solid #5b6472"></i>fold 0 test patient (unseen by K-Ont)</span></div>'}
<div class="fig">{chart_maps(pm)}<div class="cap">Each representation reduced to two dimensions with t-SNE. OnT's five-concept representation gives many patients exactly the same point ({distinct.get("OnT (5 concepts)")} distinct points for 303 patients), because patients with the same categories become identical. K-Ont and the plain encoder keep every patient distinct.</div></div></div>
<h3>Checks on the representations</h3>
<div class="fig">{chart_relations(checks)}<div class="cap">Relation check on fold 0's unseen patients: rotate every possible value concept by the relation (OnT's learned role rotation) and see whether the patient's own value is nearest. K-Ont beats chance on all five relations; OnT, whose relations were never trained on patients, is below chance on chest pain.</div></div>
{checks_tab}

<h2 class="pb">6. Test 1: prediction with the complete table</h2>
{legend_html(FEATS)}
<div class="figs2"><div class="fig">{dotplot(auc, FEATS, "mean_pct", "Table + embedding")}</div>
<div class="fig">{legend_html(EMBS)}{dotplot(auc, EMBS, "mean_pct", "Embedding on its own")}</div></div>
<div class="cap">Mean ROC-AUC over 10 folds.</div>
<h3>ROC-AUC (%)</h3>{main_tab}
<h3>Accuracy (%)</h3>{acc_tab}
<h3>Paired differences in AUC (points; wins/losses over 10 folds; Wilcoxon p)</h3>
<p class="small">Green: significantly better; red: significantly worse.</p>
{paired_tab}
<p><b>Reading.</b> On its own, K-Ont's patient vector beats OnT's in every model ({k_vs_o_sig} of 5 significantly). Added to the complete table, no embedding improves on the table alone, and OnT's concept vectors often make it significantly worse. K-Ont added to the table is close to the table alone and slightly ahead of OnT added to the table. The plain encoder on the same sentence performs like K-Ont.</p>

<h2 class="pb">7. Test 2: few labelled patients</h2>
<p>The classifiers learn from 10, 20, 50 or 100% of the training patients' labels (stratified, 3 random subsets per fold). The embeddings, scalers and PCA use all training patients without labels, which is legitimate because no encoder ever sees a diagnosis.</p>
{legend_html(FEATS)}
<div class="fig">{chart_few(few)}<div class="cap">Mean ROC-AUC by share of labels. Gradient boosting cannot split on 24 patients (minimum leaf size 20), so at 10% it predicts one value for everyone with every feature set.</div></div>
<h3>Raw + K-Ont vs the table alone (AUC points; Wilcoxon p over 30 runs, 10 at 100%)</h3>{few_tab}
<p><b>Reading.</b> The hypothesis that pretrained embeddings help when labels are scarce does not hold here: with 10–50% of the labels, adding K-Ont lowers AUC, significantly in most models. The embeddings carry no information beyond the table, so the extra 16 numbers per patient only give the classifier more to overfit on when it sees 24 to 120 labelled patients.</p>

<h2>8. Test 3: columns missing from the table</h2>
<p>Chest pain type and thallium scan (and, in the second case, the number of major vessels) are removed from the table the classifiers see, while the knowledge-graph embeddings still contain them. OnT's concept vectors hold chest pain and thallium but cannot hold the number of vessels; K-Ont's patient sentence holds all three.</p>
<div class="avoid">{legend_html(FEATS)}
<div class="figs2"><div class="fig">{dotplot([r for r in mis if r["scenario"] == "hide cp+thal"], FEATS, "auc_pct", "Chest pain, thallium hidden")}</div>
<div class="fig">{dotplot([r for r in mis if r["scenario"] == "hide cp+thal+ca"], FEATS, "auc_pct", "Chest pain, thallium, vessels hidden")}</div></div></div>
{mis_tab}
<p><b>Reading.</b> This is where a knowledge-graph embedding pays off: with three columns hidden, adding K-Ont recovers {signed(miss3m)} AUC points on average and improves all five models significantly ({miss3sig} of 5). K-Ont is ahead of OnT, most clearly for the random forest, because OnT cannot represent the vessel count. The plain encoder on the same sentence recovers about as much, so the benefit comes from embedding each patient's facts as an individual.</p>

<h2 class="pb">9. Conclusions</h2>
<ul>
  <li><b>K-Ont outperforms standard OnT as a representation of knowledge-graph individuals.</b> It types unseen patients correctly, keeps patients distinct, beats chance on all five relations, and its vectors predict the outcome better than OnT's. This agrees with our battery-materials results.</li>
  <li><b>A knowledge-graph embedding helps a classifier when the graph holds information the classifier's input lacks</b> (test 3). It does not help when the input already contains everything (test 1), and it hurts with very few labels when it adds no new information (test 2).</li>
  <li><b>K-Ont's training adds little beyond encoding the patient as text on this dataset.</b> The heart ontology is shallow and states no medical knowledge, so there is little for training to inject. The added value of K-Ont over OnT here is that it can embed individuals at all.</li>
</ul>
<h3>Limitations</h3>
<ul>
  <li>One small dataset (303 patients) and a shallow ontology; 10 folds give limited statistical power for differences of 1–2 AUC points.</li>
  <li>One K-Ont configuration (3 epochs, 16 PCA components); no tuning was done for either OnT or K-Ont, to keep the comparison fair.</li>
  <li>The relation check uses one fold's model; typing and maps use fold 0's K-Ont.</li>
</ul>
<h3>Recommended next steps</h3>
<ul>
  <li>Test on a knowledge graph with real domain knowledge that the table lacks: risk factors, drug–condition links, comorbidities, or links between patients.</li>
  <li>Report the missing-information setting (test 3) as the main use case of knowledge-graph embeddings, and keep the plain-encoder control in every comparison.</li>
  <li>For the battery project, the same lesson applies: K-Ont adds most where the knowledge graph carries structure or knowledge that the generator's prompt does not already contain.</li>
</ul>

<h2>Appendix: reproducibility</h2>
<pre>python heart/build_tbox.py --kg heart_disease_knowledge_graph.owl --out data/heart/heart_tbox.owl
python -m tbox.prepare --owl data/heart/heart_tbox.owl --out-dir data/heart/ont_tbox     # OnT's own data prep
python heart/kont_heart.py --stage prep | ont | folds | eval                          # test 1
python heart/report_data.py --stage fold0 | tests | visuals                           # tests 2-3, figures
python heart/report_datasets.py ; python heart/report_pdf.py                          # this report</pre>
<p class="small">Folds: sklearn RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0). Base encoder: Hui97/OnT-MiniLM-L12-galen. Results: data/heart/results.json and data/heart/report/.</p>
</body></html>"""
    hp = os.path.join(R, "KOnt_vs_OnT_heart_disease_report.html")
    open(hp, "w", encoding="utf-8").write(doc)
    pdf = os.path.join(R, "KOnt_vs_OnT_heart_disease_report.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + hp.replace("\\", "/")], check=True, capture_output=True, timeout=180)
    print("wrote", pdf, os.path.getsize(pdf), "bytes")


if __name__ == "__main__":
    main()
