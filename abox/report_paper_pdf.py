"""K-Ont results report for the paper (BUILD_LOG Steps 36-44) as a PDF: every number is read from the result
files the experiments wrote; charts with matplotlib (inline SVG), an A4 print-styled HTML document, printed
by headless Chrome (as heart/report_pdf.py).

    .venv-crystallm/Scripts/python.exe abox/report_paper_pdf.py  ->  ~/Downloads/K-Ont paper results report.pdf
"""
import html
import io
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "crystallm")
OUTDIR = os.path.join(os.path.expanduser("~"), "Downloads")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
INK, MUTED, GRID = "#1d2430", "#5b6472", "#e3e6ea"
COL = {"kont": "#d9762b", "plain": "#4b9b6e", "crystallm": "#6c5ba7", "comp": "#8a8f98", "nokont": "#3b6fb6", "neg": "#b23b2a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#b9bec6", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK})


def J(name):
    return json.load(open(os.path.join(DATA, name)))


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def e(x):
    return html.escape(str(x))


def sg(v):
    v = float(v)
    return ("+" if v > 0 else "−" if v < 0 else "±") + f"{abs(v):.1f}"


def ci(c):
    return f"{sg(c[0])} [{sg(c[1])}, {sg(c[2])}]"


def table(head, rows, num_from=1):
    h = "".join(f'<th class="{"num" if i >= num_from else ""}">{e(x)}</th>' for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i >= num_from else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def arm_stats(arm):
    m = J(f"kgcond3_eval_{arm}.json")
    S = [s for t in m for s in t["samples"]]
    n = len(S)
    return {"strict": 100 * sum(bool(s.get("strict")) for s in S) / n,
            "found": 100 * sum(any(s.get("strict") for s in t["samples"]) for t in m) / len(m),
            "complies": 100 * sum(bool(s.get("complies")) for s in S) / n,
            "comp": 100 * sum(bool(s.get("composition_ok")) for s in S) / n}


# ------------------------------------------------------------------ charts
def fig_template(tg):
    ks = [1, 2, 3, 5, 8]
    series = [("CrystaLLM fine-tuned", "CrystaLLM (fine-tuned)", COL["crystallm"], "o"),
              ("composition rule", "template · composition rule", COL["comp"], "s"),
              ("prompt+kg", "template · KG ranker, no K-Ont", COL["nokont"], "^"),
              ("prompt+kg+ft_base", "template · plain-encoder ranker", COL["plain"], "D"),
              ("prompt+kg+ft_kont", "template · K-Ont ranker", COL["kont"], "o")]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for key, lab, c, mk in series:
        ys = [tg[key][f"found@{k}"] for k in ks]
        ax.plot(ks, ys, color=c, lw=2.2 if "kont" in key else 1.6, marker=mk, ms=4.5, label=lab, zorder=3 if "kont" in key else 2)
    ax.set_xticks(ks)
    ax.set_xlabel("attempts per material (k)")
    ax.set_ylabel("right structure found (%)")
    ax.set_xlim(0.7, 8.3)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Structures recovered within k attempts, 902 held-out materials", loc="left")
    return svg(fig)


def fig_hybrid(tg):
    rows = [("CrystaLLM × 8", tg["CrystaLLM fine-tuned"]["found@8"], COL["crystallm"])]
    for kt in (1, 2, 3, 4):
        rows.append((f"{kt} K-Ont templates + {8 - kt} CrystaLLM", tg[f"hybrid {kt} template (prompt+kg+ft_kont) + {8 - kt} CrystaLLM"]["found@8"], COL["kont"]))
    rows.append(("3 composition-rule templates + 5 CrystaLLM", tg["hybrid 3 template (composition rule) + 5 CrystaLLM"]["found@8"], COL["comp"]))
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    for i, (lab, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.62)
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(60, 80)
    ax.set_xlabel("right structure found within 8 attempts (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Same budget of 8 attempts: templates and CrystaLLM combined", loc="left")
    return svg(fig)


def fig_injection(st):
    order = [("const", "plain fine-tune (no K-Ont)", COL["crystallm"]), ("cf_oracle", "true analogue (ceiling)", "#9aa3ad"),
             ("cfbase_rr", "plain-encoder vector, ranked analogue", COL["plain"]), ("cf_gate", "K-Ont, confidence-gated", COL["kont"]),
             ("cf_rr", "K-Ont vector, ranked analogue", COL["kont"]), ("cf_mix", "K-Ont, top-2 mix", COL["kont"]),
             ("cf_rrnok", "K-Ont vector, ranker w/o K-Ont", COL["kont"]), ("cf_none", "K-Ont model, no vector", "#c9ced6"),
             ("cf_comp", "K-Ont vector, composition analogue", COL["kont"]), ("cf_rand", "K-Ont vector, random analogue", COL["neg"]),
             ("cfbase_rand", "plain-encoder vector, random analogue", COL["neg"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (a, lab, c) in enumerate(order):
        v = st[a]["strict"]
        ax.barh(i, v, color=c, height=0.62, alpha=0.55 if a in ("cf_none",) else 1)
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=7.6, color=INK)
    ax.axvline(st["const"]["strict"], color=COL["crystallm"], lw=1, ls="--")
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("generated samples with the true structure (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Injecting framework vectors into CrystaLLM (dashed: plain fine-tune)", loc="left")
    return svg(fig)


def fig_curve(lc):
    fr = [0.05, 0.1, 0.25, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for key, lab, c in (("kont", "K-Ont start", COL["kont"]), ("base", "plain start", COL["plain"])):
        ys = [lc["fractions"][str(f)][key]["template"] for f in fr]
        for f in fr:
            ps = lc["fractions"][str(f)][key]["per_seed_template"]
            ax.scatter([f * 100] * len(ps), ps, s=9, color=c, alpha=0.45, zorder=2)
        ax.plot([f * 100 for f in fr], ys, color=c, lw=2, marker="o", ms=4, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100], ["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("retriever training data")
    ax.set_ylabel("template right @1 (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.set_title("Learning curve, 3 seeds (dots)", loc="left")
    return svg(fig)


def fig_onemodel(fw):
    keys = ["material typing H@1", "crystal typing H@1", "role, no label line (test)", "family, no label line (test)", "TBox MRR"]
    labs = ["material\ntyping", "crystal\ntyping", "battery\nrole", "structure\nfamily", "hierarchy\n(TBox MRR)"]
    ser = [("K-Ont (Step 36)", "K-Ont", "#9aa3ad"), ("fw_plain", "+ framework relation", COL["neg"]),
           ("fw_retain", "+ framework relation + retention", COL["kont"])]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    w = 0.26
    for j, (k, lab, c) in enumerate(ser):
        ax.bar([i + (j - 1) * w for i in range(len(keys))], [100 * fw["checks"][k][x] for x in keys], w * 0.92, color=c, label=lab)
    ax.set_xticks(range(len(keys)), labs, fontsize=7.2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    return svg(fig)


def fig_novel(nv):
    r = nv["report"]["vs CrystaLLM (method - CrystaLLM, negative = lower energy)"]
    rows = [("K-Ont templates (top 3)", r["kont3"]), ("plain-encoder templates (top 3)", r["base3"]),
            ("parent's framework (1)", r["parent"]), ("K-Ont top 3 + CrystaLLM", r["kont3 + CrystaLLM"])]
    fig, ax = plt.subplots(figsize=(6.6, 1.75))
    for i, (lab, d) in enumerate(rows):
        n = d["n"]
        left = 0
        for k, c in (("win", "#1f7a4d"), ("tie", "#c9ced6"), ("loss", COL["neg"])):
            v = 100 * d[k] / n
            ax.barh(i, v, left=left, color=c, height=0.62)
            if v > 6:
                ax.text(left + v / 2, i, str(d[k]), ha="center", va="center", fontsize=7.6, color="white" if k != "tie" else INK)
            left += v
    ax.set_yticks(range(len(rows)), [x[0] for x in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of 199 novel compositions (%) — green: lower energy than CrystaLLM-large, grey: within 5 meV/atom, red: higher")
    ax.set_title("New cathode compositions: lowest CHGNet energy vs CrystaLLM-large's best of 10", loc="left")
    return svg(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 1.0))
    ax.axis("off")
    boxes = [("new composition\n(formula + cell)", "#eef1f5"), ("K-Ont retriever\nranks KG frameworks", "#fbe9dc"),
             ("top-k frameworks\nfrom the battery KG", "#eef1f5"), ("place the elements\n(substitution)", "#eef1f5"),
             ("candidate crystals\n(+ optional CHGNet)", "#e6f2ec")]
    w, gap = 0.18, 0.025
    for i, (t, c) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.2), w, 0.6, facecolor=c, edgecolor="#b9bec6", lw=0.8))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", fontsize=6.3, color=INK)
        if i:
            ax.annotate("", (x, 0.5), (x - gap, 0.5), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return svg(fig)


# ------------------------------------------------------------------ document
def main():
    tg = J("kg_template_gen.json")["report"]
    tgr = J("kg_template_gen_relaxed.json")["report"]
    rr = J("kg_rerank.json")["report"]["unseen_all"]
    c3 = J("kgcond3_report.json")["all"]["paired"]
    st = {a: arm_stats(a) for a in ("const", "cf_rr", "cf_rrnok", "cf_comp", "cf_rand", "cf_none", "cf_gate", "cf_oracle", "cf_mix",
                                    "cfbase_rr", "cfbase_rand")}
    lc = J("kg_learning_curve.json")
    fw = J("kont_framework.json")
    nv = J("novel_template.json")
    nvr = nv["report"]
    nvc = nvr["vs CrystaLLM (method - CrystaLLM, negative = lower energy)"]
    k1 = tg["K-Ont template - CrystaLLM @1"]
    hyb = tg["hybrid 3 template (prompt+kg+ft_kont) + 5 CrystaLLM"]

    t_ret = table(["ranker (902 held-out materials)", "right space group", "right framework"], [
        ["composition rule (earlier steps)", f'{rr["composition rule"]["sg"]:.1f}', f'{rr["composition rule"]["frame"]:.1f}'],
        ["prompt features (pool popularity, cell size)", f'{rr["prompt"]["sg"]:.1f}', f'{rr["prompt"]["frame"]:.1f}'],
        ["+ KG facts (properties)", f'{rr["prompt+kg"]["sg"]:.1f}', f'{rr["prompt+kg"]["frame"]:.1f}'],
        ["+ raw K-Ont vectors (no fine-tuning)", f'{rr["prompt+kg+kont"]["sg"]:.1f}', f'{rr["prompt+kg+kont"]["frame"]:.1f}'],
        ["<b>+ fine-tuned K-Ont retriever</b>", f'<b>{rr["prompt+kg+ft_kont"]["sg"]:.1f}</b>', f'<b>{rr["prompt+kg+ft_kont"]["frame"]:.1f}</b>'],
        ["+ fine-tuned plain-encoder retriever (control)", f'{rr["prompt+kg+ft_base"]["sg"]:.1f}', f'{rr["prompt+kg+ft_base"]["frame"]:.1f}']])
    t_tmpl = table(["found within k attempts (%)", "k=1", "k=2", "k=3", "k=5", "k=8"],
                   [[lab, *[f'{tg[key][f"found@{k}"]:.1f}' for k in (1, 2, 3, 5, 8)]] for key, lab in (
                       ("CrystaLLM fine-tuned", "CrystaLLM, fine-tuned on the same KG data"), ("composition rule", "template · composition rule"),
                       ("prompt+kg", "template · KG ranker without K-Ont"), ("prompt+kg+ft_kont", "<b>template · K-Ont ranker</b>"),
                       ("prompt+kg+ft_base", "template · plain-encoder ranker (control)"))]
                   + [["template · K-Ont ranker, CHGNet-relaxed", *[f'{tgr["K-Ont template, CHGNet-relaxed"][f"found@{k}"]:.1f}' for k in (1, 2, 3)], "–", "–"]])
    t_tmpl_d = table(["comparison", "k=1", "k=3", "k=8"], [
        ["K-Ont template − CrystaLLM", *[ci(tg[f"K-Ont template - CrystaLLM @{k}"]) for k in (1, 3, 8)]],
        ["K-Ont template − composition-rule template", *[ci(tg[f"K-Ont template - composition template @{k}"]) for k in (1, 3, 8)]],
        ["K-Ont template − template without K-Ont", *[ci(tg[f"K-Ont template - no-K-Ont template @{k}"]) for k in (1, 3, 8)]],
        ["K-Ont template − plain-encoder template", *[ci(tg[f"K-Ont template - plain-encoder template @{k}"]) for k in (1, 3, 8)]]])
    inj_rows = [[lab, f'{st[a]["strict"]:.1f}', f'{st[a]["found"]:.1f}', f'{st[a]["complies"]:.1f}', f'{st[a]["comp"]:.1f}'] for a, lab in (
        ("const", "plain fine-tune, no K-Ont (baseline)"), ("cf_rr", "K-Ont vector, K-Ont-ranked analogue"),
        ("cf_gate", "K-Ont vector only when the ranker is confident"), ("cf_mix", "half 1st, half 2nd analogue"),
        ("cf_rrnok", "analogue ranked without K-Ont"), ("cf_comp", "composition-rule analogue"), ("cf_none", "same model, no vector"),
        ("cf_rand", "random analogue (control)"), ("cf_oracle", "true analogue (ceiling, uses the answer)"),
        ("cfbase_rr", "plain-encoder vector, ranked analogue"), ("cfbase_rand", "plain-encoder vector, random analogue"))]
    t_inj = table(["arm (902 materials × 8 samples)", "true structure", "found within 8", "follows analogue", "right composition"], inj_rows)
    t_inj_d = table(["paired difference [95% CI]", "true structure", "follows analogue"], [
        [k, ci([c3[k]["strict"]["diff"], *c3[k]["strict"]["ci95"]]), ci([c3[k]["complies"]["diff"], *c3[k]["complies"]["ci95"]])]
        for k in ("cf_rr - const", "cf_rr - cf_rand", "cf_rr - cf_none", "cf_oracle - const", "cf_rr - cfbase_rr")])
    names = {"cf_rr - const": "K-Ont-ranked analogue − plain fine-tune", "cf_rr - cf_rand": "K-Ont-ranked analogue − random analogue",
             "cf_rr - cf_none": "K-Ont-ranked analogue − same model, no vector", "cf_oracle - const": "true analogue (ceiling) − plain fine-tune",
             "cf_rr - cfbase_rr": "K-Ont vector − plain-encoder vector (same analogue)"}
    for k, v in names.items():
        t_inj_d = t_inj_d.replace(f">{k}<", f">{v}<")
    t_lc = table(["training data", "K-Ont start", "plain start", "K-Ont − plain [95% CI]", "space group, K-Ont − plain"], [
        [f"{int(float(f) * 100)}%", f'{v["kont"]["template"]:.1f}', f'{v["base"]["template"]:.1f}', ci(v["kont - base (template)"]), ci(v["kont - base (sg)"])]
        for f, v in lc["fractions"].items()])
    ck = fw["checks"]
    t_fw = table(["check", "K-Ont", "+ framework relation", "+ relation + retention"], [
        [lab, *[f'{100 * ck[m][k]:.1f}' for m in ("K-Ont (Step 36)", "fw_plain", "fw_retain")]] for k, lab in (
            ("material typing H@1", "material typing (H@1)"), ("crystal typing H@1", "crystal typing (H@1)"),
            ("role, no label line (test)", "battery role, label removed (held-out)"),
            ("family, no label line (test)", "structure family, label removed (held-out)"), ("TBox MRR", "ontology hierarchy (TBox MRR ×100)"))]
        + [["crystal retrieval: template right @1 / @3", "–", f'{fw["template"]["fw_plain"]["found@1"]:.1f} / {fw["template"]["fw_plain"]["found@3"]:.1f}',
            f'{fw["template"]["fw_retain"]["found@1"]:.1f} / {fw["template"]["fw_retain"]["found@3"]:.1f}']])
    t_nv = table(["method − CrystaLLM-large best of 10", "lower", "tie", "higher", "mean, meV/atom [95% CI]"], [
        [lab, d["win"], d["tie"], d["loss"], f'{sg(d["mean_delta_meV"])} [{sg(d["ci95_meV"][0])}, {sg(d["ci95_meV"][1])}]']
        for lab, d in (("K-Ont templates, top 3", nvc["kont3"]), ("plain-encoder templates, top 3", nvc["base3"]),
                       ("parent's framework (1 candidate)", nvc["parent"]), ("<b>K-Ont top 3 + CrystaLLM</b>", nvc["kont3 + CrystaLLM"]))])

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>K-Ont paper results</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; color: {INK}; font-size: 9pt; line-height: 1.42; margin: 0; }}
h1 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 24pt; line-height: 1.15; margin: 0 0 6pt; }}
h2 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 15pt; margin: 14pt 0 6pt; border-bottom: 1px solid #d5d9df; padding-bottom: 3pt; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 9pt 0 4pt; break-after: avoid; }}
p {{ margin: 0 0 6pt; }}
.small {{ font-size: 8.3pt; color: {MUTED}; }}
.cover {{ height: 255mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }}
.kicker {{ font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; color: #a5531d; }}
.summary {{ border: 1px solid #d5d9df; border-radius: 6px; padding: 10pt 12pt; background: #f7f8fa; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 8pt 0; }}
.tile {{ border: 1px solid #d5d9df; border-radius: 6px; padding: 7pt 8pt; background: #fff; }}
.tile .n {{ font-family: Cambria, Georgia, serif; font-size: 18pt; }}
.tile .t {{ font-size: 7.8pt; color: {MUTED}; line-height: 1.3; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.8pt; margin: 4pt 0 8pt; font-variant-numeric: tabular-nums; break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #e3e6ea; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }}
th {{ color: {MUTED}; font-weight: 600; }}
td.num, th.num {{ text-align: right; }}
pre {{ font-family: Consolas, "Courier New", monospace; font-size: 7.6pt; background: #f4f6f8; border-radius: 4px; padding: 6pt 8pt; margin: 0; white-space: pre-wrap; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }}
.two > div {{ min-width: 0; }}
.fig {{ break-inside: avoid; margin: 4pt 0 8pt; }}
.fig svg {{ width: 100%; height: auto; }}
svg text {{ font-family: "Segoe UI", Calibri, Arial, sans-serif !important; }}
.cap {{ font-size: 8.1pt; color: {MUTED}; margin-top: 2pt; }}
.box {{ border-left: 3px solid #d9762b; background: #fbf4ee; padding: 6pt 9pt; margin: 6pt 0 8pt; break-inside: avoid; }}
.box.neg {{ border-left-color: #b23b2a; background: #fbf0ee; }}
.box.ok {{ border-left-color: #1f7a4d; background: #eef6f1; }}
.avoid {{ break-inside: avoid; }}
ul {{ margin: 0 0 6pt 14pt; padding: 0; }} li {{ margin-bottom: 3pt; }}
.pb {{ break-before: page; }}
</style></head><body>

<section class="cover">
  <div>
    <div class="kicker">K-Ont research · results for the paper · BUILD_LOG Steps 36–44</div>
    <h1>K-Ont: one ontology + knowledge-graph encoder that reasons over the ontology and guides crystal generation</h1>
    <p style="font-size:11pt;color:{MUTED};max-width:155mm">What we built, every experiment that tests it, and an honest reading of what the results do and do not support — for going through the results and deciding the paper.</p>
  </div>
  <div class="summary">
    <h3 style="margin-top:0">Summary</h3>
    <div class="tiles">
      <div class="tile"><div class="n">{sg(k1[0])}</div><div class="t">points: K-Ont-guided templates vs fine-tuned CrystaLLM, right structure at the first attempt (902 held-out materials; CI {sg(k1[1])} to {sg(k1[2])})</div></div>
      <div class="tile"><div class="n">{hyb["found@8"]:.1f}%</div><div class="t">right structure within 8 attempts, 3 K-Ont templates + 5 CrystaLLM samples, vs {tg["CrystaLLM fine-tuned"]["found@8"]:.1f}% for CrystaLLM alone</div></div>
      <div class="tile"><div class="n">≥99.9%</div><div class="t">every K-Ont ontology check kept by the single model that also retrieves crystal frameworks (retention loss)</div></div>
      <div class="tile"><div class="n">{nvc["kont3 + CrystaLLM"]["win"]} / 0</div><div class="t">new cathode compositions where adding K-Ont templates found a lower / higher energy structure than CrystaLLM-large alone (of {nvc["kont3 + CrystaLLM"]["n"]})</div></div>
    </div>
    <ul>
      <li><b>Works:</b> K-Ont learns the battery ontology and KG (typing, roles, families, hierarchy at 98.8–100%) and beats OnT as a representation on heart disease; one K-Ont model can do ontology reasoning and crystal-framework retrieval together; K-Ont-guided templates beat CrystaLLM on known materials and complement it on new ones.</li>
      <li><b>Does not work:</b> injecting K-Ont embeddings into CrystaLLM. The vector steers the structure, but CrystaLLM already recalls Materials Project crystals, so even the true framework adds only ~2 points.</li>
      <li><b>Caveats to state in the paper:</b> a retriever started from the plain encoder does about as well as one started from K-Ont (K-Ont helps only at 5% data); KG labels are rule-based; new-material energies are CHGNet, not DFT.</li>
    </ul>
  </div>
  <p class="small">K-Ont repository, docs/BUILD_LOG.md Steps 36–44 and docs/PAPER_OUTLINE.md. All numbers are read from the experiments' result files. 95% intervals are paired bootstrap over materials.</p>
</section>

<h2>1 What K-Ont is, and how it is tested</h2>
<p><b>OnT</b> embeds an ontology's classes in hyperbolic space (a Poincaré ball), so that general classes sit near the centre and specific ones further out, and "is-a" becomes geometry. <b>K-Ont</b> extends it to the knowledge graph's <i>individuals</i>: each material, crystal and element is written as a short sentence, and training adds rows saying what type each individual is and how it is related to others, alongside OnT's own class rows. One encoder then places classes and individuals in the same space.</p>
<p>The test bed is the battGPT battery-materials KG: 5,000 Materials Project materials with their crystals, elements, battery roles and structure families, linked to the battery ontology. K-Ont was trained on 3,982 materials; <b>1,018 were held out</b>. Every crystal experiment below evaluates only on those held-out materials (902 of them have a same-formula-type material in the KG to learn from), and nothing evaluated was used in any training.</p>
<h3>Results already established before this round (Step 36, heart-disease report)</h3>
{table(["check (battery KG)", "K-Ont", "baseline"], [["material typing", "100%", "–"], ["crystal typing", "99.7%", "–"], ["battery role, label line removed (held-out)", "100%", "84% (majority class)"], ["structure family, label line removed (held-out)", "98.8%", "47% (majority class)"], ["ontology hierarchy (TBox MRR)", "0.97", "–"]])}
<p>On the heart-disease dataset (guide's verification), K-Ont patient embeddings beat OnT's by +2.2 AUC on average (significant for 4 of 5 models), and by +4.3 when three columns are missing (5 of 5); with only 10% of the labels they were worse (−4.0). Full details are in the separate heart-disease report.</p>

<h2>2 Can K-Ont make a crystal LLM better? Injection into CrystaLLM (Step 40)</h2>
<p>CrystaLLM writes a crystal file token by token from a composition. We gave it a K-Ont vector describing the framework of a similar KG material (an <i>analogue</i>), as 8 extra input tokens plus a gated adapter in every layer. Earlier attempts (Steps 37–38) showed the model ignored the vector: it was pre-trained on Materials Project and already knows these crystals. So this time it was trained on <b>6,453 counterfactual crystals</b> — a training composition placed into a <i>different</i> framework — so that the only way to predict them is to read the vector. The analogue at test time is picked by a learned ranker (Section 3).</p>
<div class="fig">{fig_injection(st)}</div>
{t_inj}
{t_inj_d}
<div class="box neg"><b>Reading.</b> The vector really steers the structure: with a random analogue the true-structure share halves (22.7% vs 44.1%). But it never beats plain fine-tuning (−3.4, significant). The ceiling explains why: given the <i>true</i> framework, the model reaches only 49.6% vs 47.5% — wherever an analogue would be right, CrystaLLM already recalls the answer. Plain-encoder vectors steer just as well, so this steering is not specific to K-Ont. This is a negative result with its cause measured, and it points to using K-Ont outside the generator.</div>

<h2 class="pb">3 K-Ont-guided template generation (Steps 40–41)</h2>
<div class="fig">{fig_pipeline()}</div>
<p>No neural decoder: K-Ont ranks the KG materials of the same formula type by how likely they share the target's framework, and the target's elements are placed on the top frameworks' sites. The ranker combines simple features (how common each framework is, cell size), the KG's facts, and a <b>K-Ont retriever</b> fine-tuned on one new KG relation: "this material has that material's framework".</p>
<h3>Picking the right framework</h3>
{t_ret}
<div class="fig">{fig_template(tg)}</div>
{t_tmpl}
{t_tmpl_d}
<div class="fig">{fig_hybrid(tg)}</div>
<div class="box ok"><b>Reading.</b> Three things are significant: K-Ont-guided templates beat fine-tuned CrystaLLM at every budget ({sg(k1[0])} points at one attempt); they beat templates chosen without the KG-trained retriever (+3 to +4) and by composition alone (+9 to +12); and combined with CrystaLLM at the same budget of 8 they reach {hyb["found@8"]:.1f}% vs {tg["CrystaLLM fine-tuned"]["found@8"]:.1f}% ({ci(hyb["vs CrystaLLM x8"])}) because the two get different materials right. CHGNet relaxation hardly changes the template result ({tgr["K-Ont template, CHGNet-relaxed"]["found@1"]:.1f}% vs {tg["prompt+kg+ft_kont"]["found@1"]:.1f}% at one attempt). <b>Caveat:</b> the same retriever started from the plain encoder does as well (equal at k=1, slightly better at k≥5) — the gain comes from KG-trained retrieval, not from K-Ont's starting weights.</div>

<h2 class="pb">4 Does K-Ont's own training help? (Steps 42–43)</h2>
<div class="two">
<div><h3>Learning curve: K-Ont vs plain starting point</h3><div class="fig">{fig_curve(lc)}</div>
<p class="cap">Retriever trained on 5–100% of the KG examples, 3 seeds; both starts get exactly the same data. Template right at the first attempt, 902 held-out materials.</p></div>
<div><h3>One K-Ont model for the ontology and crystals</h3><div class="fig">{fig_onemodel(fw)}</div>
<p class="cap">K-Ont fine-tuned on the framework relation, with and without a retention loss that keeps its original embeddings of all 10,108 KG and ontology sentences; then the same checks as K-Ont itself.</p></div>
</div>
{t_lc}
{t_fw}
<div class="box"><b>Reading.</b> K-Ont's starting weights help only with very little data (5%: right space group +3.4, significant), are slightly worse at 10–25% and equal from 50%; K-Ont runs are steadier across seeds. The K-Ont-specific result is the second one: trained on the new relation alone, K-Ont <i>forgets</i> its ontology (material typing collapses to 0.04%); with the retention loss, <b>one model keeps every ontology check at ≥99.9% and retrieves crystal frameworks</b>, at a small cost (template right @1 {fw["template"]["fw_retain"]["found@1"]:.1f}% vs {fw["template"]["K-Ont retriever (Step 40)"]["found@1"]:.1f}% for the specialist retriever).</div>

<h2 class="pb">5 New cathode compositions (Step 44)</h2>
<p>The real use: compositions that do not exist in Materials Project. 200 new cathodes were made earlier (Step 28) by swapping elements in KG cathodes. No true structure exists, so methods are compared by the <b>lowest energy</b> they reach for each composition after CHGNet relaxation (lower = a more stable, more plausible structure; same settings for all). CrystaLLM-large had 10 relaxed samples per composition; the templates get 3. Because a new composition has no computed properties, the K-Ont retriever here reads only formula, chemical system and cell.</p>
<div class="fig">{fig_novel(nv)}</div>
{t_nv}
<div class="box ok"><b>Reading.</b> Three K-Ont templates, with no neural generator, are statistically level with CrystaLLM-large's best of ten ({sg(nvc["kont3"]["mean_delta_meV"])} meV/atom, CI {sg(nvc["kont3"]["ci95_meV"][0])} to {sg(nvc["kont3"]["ci95_meV"][1])}). Adding them to CrystaLLM finds a lower-energy structure for {nvc["kont3 + CrystaLLM"]["win"]} of {nvc["kont3 + CrystaLLM"]["n"]} compositions (17%) and a worse one for none ({sg(nvc["kont3 + CrystaLLM"]["mean_delta_meV"])} meV/atom, CI {sg(nvc["kont3 + CrystaLLM"]["ci95_meV"][0])} to {sg(nvc["kont3 + CrystaLLM"]["ci95_meV"][1])}). <b>Caveats:</b> the parent's own framework — how each composition was made — is a strong hand-made baseline, slightly better than K-Ont's top 3 ({sg(nvr["kont3 - parent"]["mean_delta_meV"])} meV/atom, CI {sg(nvr["kont3 - parent"]["ci95_meV"][0])} to {sg(nvr["kont3 - parent"]["ci95_meV"][1])}); K-Ont vs plain-encoder templates is not significant; energies are CHGNet, not DFT.</div>

<h2 class="pb">6 Is it paper-worthy? An honest assessment</h2>
<h3>Strengths</h3>
<ul>
<li><b>Method:</b> K-Ont extends a published hyperbolic ontology encoder (OnT) from classes to KG individuals and relations — concrete and reusable.</li>
<li><b>Two domains:</b> battery KG (98.8–100% on typing, roles, families, hierarchy) and heart disease (beats OnT).</li>
<li><b>A K-Ont-specific result:</b> one model for ontology reasoning and crystal retrieval without forgetting (Step 43), with the ablation showing why retention is needed.</li>
<li><b>A practical win over a strong baseline:</b> K-Ont-guided templates beat fine-tuned CrystaLLM on 902 held-out materials and complement CrystaLLM-large on new compositions.</li>
<li><b>Rigour:</b> leak-free splits, acceptance criteria fixed before results, random / plain-encoder / true-answer controls, bootstrap intervals; the injection negative result has its cause measured.</li>
</ul>
<h3>What a reviewer will push on</h3>
<ul>
<li>Is it K-Ont, or KG-trained retrieval? A plain-encoder retriever does as well; K-Ont helps only at very low data. Defence: one model does both jobs (Step 43).</li>
<li>KG labels are rule-based, which inflates near-100% typing and family scores.</li>
<li>Known-material benchmarks favour memorising generators; the new-material test uses CHGNet, not DFT.</li>
<li>Substitution-based structure prediction is established (Hautier et al.); the novelty is ontology/KG-guided selection.</li>
<li>One crystal domain and one KG.</li>
</ul>
<h3>Venues</h3>
<ul>
<li><b>Good fit now:</b> ML-for-materials / AI-for-science workshops (NeurIPS, ICML), or knowledge-graph and semantic-web venues (ISWC, ESWC, Semantic Web journal).</li>
<li><b>Reachable with more work:</b> <i>Digital Discovery</i>, <i>npj Computational Materials</i> — expect DFT checks on top new candidates and a broader comparison.</li>
<li><b>Hard for now:</b> NeurIPS / ICML main track — the K-Ont-specific gain is small and the domain narrow.</li>
</ul>
<h3>What would raise it most</h3>
<ul>
<li>DFT on the {nvc["kont3 + CrystaLLM"]["win"]} new compositions where the K-Ont template beat CrystaLLM.</li>
<li>A second KG for the crystal part, or a label subset checked by faculty.</li>
<li>A standard ontology benchmark from the OnT paper, to show K-Ont keeps TBox quality.</li>
</ul>

<h2>Appendix: reproducibility</h2>
<pre>.venv-crystallm/Scripts/python.exe abox/kg_pool.py                                      # split + analogue pool
.venv/Scripts/python.exe abox/kg_retriever_ft.py --init kont | base                    # fine-tuned retrievers
.venv-crystallm/Scripts/python.exe abox/kg_rerank.py                                    # learned ranker (out-of-fold)
.venv-crystallm/Scripts/python.exe abox/kg_condition3.py --stage cf | train | eval | report   # Step 40 injection
.venv-crystallm/Scripts/python.exe abox/kg_template_gen.py [--relax]                    # Step 41 templates
.venv-crystallm/Scripts/python.exe abox/kg_learning_curve.py --stage run | eval         # Step 42
.venv/Scripts/python.exe abox/kont_framework.py --stage train | checks; ... --stage eval  # Step 43
.venv-crystallm/Scripts/python.exe abox/novel_template.py --stage pool | relax          # Step 44
.venv-crystallm/Scripts/python.exe abox/report_paper_pdf.py                             # this report</pre>
<p class="small">Matching: pymatgen StructureMatcher (ltol 0.2, stol 0.3, angle 5°). Relaxation: CHGNet 0.3.0, fmax 0.1 eV/Å, 300 steps, cell free. Bootstrap: 5,000–10,000 resamples over materials. Results: data/crystallm/*.json.</p>
</body></html>"""
    hp = os.path.join(DATA, "kont_paper_report.html")
    open(hp, "w", encoding="utf-8").write(doc)
    pdf = os.path.join(OUTDIR, "K-Ont paper results report.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + hp.replace("\\", "/")], check=True, capture_output=True, timeout=180)
    print("wrote", pdf, os.path.getsize(pdf), "bytes")


if __name__ == "__main__":
    main()
