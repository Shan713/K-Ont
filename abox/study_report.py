"""Plain-language study report of the K-Ont work (BUILD_LOG Steps 1-55), for the team and the guide.

What OnT and K-Ont are, every experiment in order with what we did, what came out and what it means, and the
limitations -- in simple words. Numbers are read from the experiments' result files; charts reuse
report_paper_pdf.py's. Printed to PDF by headless Chrome.

    .venv-crystallm/Scripts/python.exe abox/study_report.py  ->  ~/Downloads/K-Ont study report.pdf
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
import report_paper_pdf as R  # noqa: E402

DATA = R.DATA
OUT = os.path.join(os.path.expanduser("~"), "Downloads", "K-Ont study report.pdf")


def step(title, did, found, means, tone="ok"):
    return f"""<div class="step avoid"><h3>{title}</h3>
<p><span class="tag">What we did</span> {did}</p>
<p><span class="tag">What we found</span> {found}</p>
<div class="means t-{tone}"><b>What it means:</b> {means}</div></div>"""


def main():
    tg = R.J("kg_template_gen.json")["report"]
    fw = R.J("kont_framework.json")
    lc = R.J("kg_learning_curve.json")
    nv = R.J("novel_template.json")
    st = {a: R.arm_stats(a) for a in ("const", "cf_rr", "cf_rrnok", "cf_comp", "cf_rand", "cf_none", "cf_gate", "cf_oracle", "cf_mix",
                                      "cfbase_rr", "cfbase_rand")}
    ss = {a: R.arm_stats(a, "_scratch") for a in ("const", "cf_rr", "cf_rrnok", "cf_rand", "cf_none", "cf_oracle")}
    hz = R.J("kg_substitution.json")["report"]
    ps = R.J("kg_pipeline_select_n300_s40.json")["report"]
    eh = R.J("teammate_ehull.json")
    sk = R.J("kgseed_eval.json")["report"]
    gr = json.load(open(os.path.join(HERE, "data", "ont_bench", "galen_results.json")))
    g = lambda t, m: gr[f"{t} | {m}"]["results"]["sentence_combined"]
    c52 = fw["checks"]["fw_retain_galen_anchor"]
    pr = ps["right_single_pick"]
    haut1 = [v["@1"] for k, v in hz["found"].items() if k.startswith("Hautier") and "order" in k][0]

    body = f"""
<section class="cover">
<div>
<div class="kicker">K-Ont project · study report · plain-language edition</div>
<h1>Teaching an AI to understand battery-materials knowledge — and using it to design crystals</h1>
<p class="lead">A step-by-step, easy-to-read account of what OnT and K-Ont are, every experiment we ran, what came out, how to read the results, and what they do not show.</p>
<p class="small">Team: Dr. Gowtham Ramesh, Shantharam P, Sarvesha Nandhakumar, Akshay KS, RD Tarun, S Ajeth · Source: the K-Ont repository (docs/BUILD_LOG.md, Steps 1–55)</p>
</div>
<div class="summary">
<h3>The whole story in 6 sentences</h3>
<ol>
<li>We built a <b>knowledge graph (KG)</b> of 5,000 battery-related materials, connected to a <b>battery ontology</b> (a formal "dictionary of concepts and how they relate").</li>
<li>We extended an existing AI method, <b>OnT</b>, which understands ontologies, into <b>K-Ont</b>, which also understands the individual materials and crystals in the KG.</li>
<li>K-Ont learned the battery ontology and KG almost perfectly (98.8–100% on its checks) and also worked on a second domain (heart disease).</li>
<li>Trying to make a crystal-generating AI (<b>CrystaLLM</b>) smarter by feeding K-Ont's knowledge <i>into</i> it helped only a little.</li>
<li>Using K-Ont to <b>pick a known crystal "template"</b> and building the new crystal from it worked much better: it beat CrystaLLM, beat the standard textbook method, and on brand-new cathode compositions an independent calculation agreed with our ranking in 32 of 34 cases.</li>
<li>Finally we made <b>one K-Ont model</b> that does everything at once (ontology reasoning, battery knowledge, the original OnT benchmark, and crystal-template retrieval) — and showed the ontology itself is what gives K-Ont its reasoning ability.</li>
</ol>
</div>
</section>

<h2>Part 1 — The basic ideas</h2>
<h3>What is an ontology?</h3>
<p>An ontology is a carefully built <b>vocabulary of concepts plus the rules connecting them</b>. In our battery ontology, for example, "olivine structure" <i>is a kind of</i> "polyanion structure family", and a material <i>has</i> a crystal structure. Think of it as a family tree of ideas, with extra rules like "every material has some crystal structure".</p>
<h3>What is a knowledge graph (KG)?</h3>
<p>A KG is a big network of <b>facts about specific things</b> that uses the ontology's vocabulary: "LiFePO<sub>4</sub> is a substance", "LiFePO<sub>4</sub> has crystal structure X", "X belongs to the olivine family", "LiFePO<sub>4</sub> belongs to the positive electrode". Our KG (built with the battGPT pipeline from the Materials Project database) has <b>5,000 materials and about 5.6 million facts</b>.</p>
<h3>What is OnT?</h3>
<p>OnT ("Language Models as Ontology Encoders", Oxford) is an AI model that reads the <i>names</i> of ontology concepts and turns each into a point in a special curved space (a <b>Poincaré ball</b>, a kind of hyperbolic space). In that space general concepts sit near the centre and specific ones further out, so "is a kind of" becomes something you can <i>measure</i> as a distance. OnT only handles <b>concepts (classes)</b>, not individual things.</p>
<h3>What is K-Ont (our contribution)?</h3>
<p>K-Ont extends OnT to the <b>individuals</b> of a knowledge graph — every material, crystal and element. Each individual is written as a short sentence (its formula, properties, structure…), and K-Ont is trained on three kinds of examples at once:</p>
<ul>
<li><b>Ontology rules</b> (TBox): "olivine structure is a kind of polyanion structure family" — OnT's original job.</li>
<li><b>Typing facts</b>: "LiFePO<sub>4</sub> is a substance", "crystal X is a crystal structure".</li>
<li><b>Relation facts</b>: "LiFePO<sub>4</sub> has structure X", "X has structure family olivine", "LiFePO<sub>4</sub> belongs to the positive electrode".</li>
</ul>
<p>After training, concepts <i>and</i> individuals live in the same curved space, so K-Ont can answer questions like "what type is this new material?" or "which battery role does it have?" by measuring distances.</p>
<h3>What is CrystaLLM, and what is a "template"?</h3>
<p><b>CrystaLLM</b> is a published AI (a GPT-style language model) that writes a crystal-structure file (CIF) atom by atom when given a chemical formula. A <b>template</b> is a known crystal from our KG that we reuse as a mould: we keep its arrangement of atoms and simply swap in the new material's elements (for example, take LiTiSn(PO<sub>4</sub>)<sub>3</sub> and replace Sn by V to get LiTiV(PO<sub>4</sub>)<sub>3</sub>). The hard part is choosing the <i>right</i> mould — that is where K-Ont comes in.</p>
<h3>How we kept the tests honest</h3>
<ul>
<li><b>Held-out materials:</b> K-Ont was trained on 3,982 materials; 1,018 were kept aside. All crystal tests use only those held-out ones (902 of them have a usable template in the KG).</li>
<li><b>Rules fixed before results:</b> for the main tests we wrote down the pass/fail rule before running.</li>
<li><b>Controls:</b> random choices, a plain (untrained) encoder, and "cheating" ceilings that use the true answer, so each gain can be attributed.</li>
<li><b>Error bars:</b> 95% confidence intervals (bootstrap). If an interval does not cross zero, the difference is real, not luck.</li>
</ul>

<h2 class="pb">Part 2 — The experiments, step by step</h2>
<h3 class="phase">Phase A · Does K-Ont learn the knowledge?</h3>
{step("A1. K-Ont on the 5,000-material battery KG (Step 36)",
      "Trained K-Ont on the KG and tested it on the 1,018 held-out materials it never saw.",
      "Material typing 100%, crystal typing 99.7%, battery role 100% (vs 84% for 'always guess cathode'), structure family 98.8% (vs 47% for 'always guess the most common family'), ontology hierarchy score 0.97 out of 1.",
      "K-Ont really learned the ontology and the KG facts, and generalises to new materials. Caveat: the KG's role/family labels are themselves produced by rules, so ~100% partly means K-Ont reproduced those rules.")}
{step("A2. A second domain: heart disease (guide's verification)",
      "Turned 303 patients into KG individuals with a heart-disease ontology; compared OnT vs K-Ont patient embeddings inside 4 classic ML models and a neural network (10-fold cross-validation).",
      "K-Ont beat OnT by +2.2 AUC on average (significant for 4 of 5 models) and by +4.3 when three data columns were missing (5 of 5); with only 10% of the labels it was worse (−4.0).",
      "K-Ont is a better way to represent individuals than OnT, especially when data are incomplete — it is not only a battery trick.")}

<h3 class="phase">Phase B · Can K-Ont make a crystal-generating AI better from the inside?</h3>
{step("B1. Feed K-Ont's vector into CrystaLLM (Steps 37–40)",
      "Gave CrystaLLM a K-Ont vector describing a similar known crystal (an 'analogue') as extra input tokens (a 'soft prompt') and trained it to follow it, including on 6,453 deliberately mismatched examples so it had to read the vector.",
      f"The vector really steers the output (true structure {st['cf_rr']['strict']:.1f}% with the K-Ont-picked analogue vs {st['cf_rand']['strict']:.1f}% with a random one), but plain fine-tuning without K-Ont was better ({st['const']['strict']:.1f}%). Even the true analogue only reached {st['cf_oracle']['strict']:.1f}%.",
      "CrystaLLM was trained on the same database and already 'remembers' these crystals, so extra hints can add at most ~2 points.", "neg")}
{step("B2. Our own GPT trained from scratch, without the test answers (Step 45)",
      "Pre-trained a CrystaLLM-sized model (25M parameters) on our laptop on 300,000 crystals with every test composition removed (169 minutes), then repeated B1.",
      f"Now K-Ont helps: {ss['cf_rr']['strict']:.1f}% vs {ss['const']['strict']:.1f}% for plain fine-tuning (+1.5, significant). But even the true analogue gives only {ss['cf_oracle']['strict']:.1f}%.",
      "Once memorisation is removed, K-Ont knowledge helps causally — but a vector is a narrow channel: it says <i>which</i> framework, while the model must still write every atom itself.", "mid")}
{step("B3. Feed the template as text instead (Step 53)",
      "Instead of a vector, wrote the K-Ont-chosen template's header (space group, cell size) directly into CrystaLLM's prompt (a 'hard prompt').",
      f"CrystaLLM got the structure right {sk['found']['seeded: top-1 template x1']:.1f}% of the time — worse than simply using the template itself ({tg['prompt+kg+ft_kont']['found@1']:.1f}%).",
      "Even with the right cell, placing the atoms is where the LLM goes wrong, and the template already has them. Conclusion of Phase B: pushing K-Ont knowledge <i>into</i> the LLM adds little.", "neg")}

<h3 class="phase">Phase C · Using K-Ont to choose templates (what works)</h3>
{step("C1. K-Ont-guided template generator (Steps 40–41)",
      "For each held-out material, K-Ont ranks the KG materials with the same formula pattern; the top ones are used as templates (atoms swapped in). A K-Ont retriever was trained on one new KG relation: 'this material has that material's framework'.",
      f"Right structure at the first try: K-Ont templates {tg['prompt+kg+ft_kont']['found@1']:.1f}% vs CrystaLLM {tg['CrystaLLM fine-tuned']['found@1']:.1f}% (+12.2, significant). Within 8 tries: 71.4% vs 68.3%. 3 templates + 5 CrystaLLM samples: 77.3% vs 68.3% for CrystaLLM alone.",
      "Choosing the right known framework is the key step, and the knowledge graph makes that choice much better than chance or simple rules. Templates and CrystaLLM get different materials right, so together they cover more.")}
{step("C2. Against the standard textbook method (Step 47)",
      "Compared K-Ont's template choice with the established data-mined substitution method (Hautier et al., 2011) on the same templates.",
      f"First try: K-Ont {tg['prompt+kg+ft_kont']['found@1']:.1f}% vs Hautier {haut1:.1f}% (+8.4, significant); with 5+ tries the standard method catches up.",
      "K-Ont picks the right mould first much more often — this answers the reviewer question 'isn't this just element substitution?'.")}
{step("C3. One final answer per material (Step 48)",
      "A practical system must commit to one answer. We pooled 3 K-Ont templates + 5 CrystaLLM samples, let a physics model (CHGNet) relax each, and picked the lowest-energy one (300 materials).",
      f"Right {pr['3 templates + 5 CrystaLLM, lowest energy (pipeline)']:.1f}% vs {pr['CrystaLLM x8, lowest energy']:.1f}% when CrystaLLM picks its own best of 8 the same way (+5.7, significant).",
      "The best design is: K-Ont proposes templates, CrystaLLM proposes its own guesses, physics decides.")}

<h3 class="phase">Phase D · Brand-new cathode materials</h3>
{step("D1. 200 new cathode compositions (Step 44)",
      "Compositions not in Materials Project (made by swapping elements in known cathodes). No true answer exists, so we compared the lowest energy each method's structure reaches (lower = more stable).",
      "3 K-Ont templates are statistically level with CrystaLLM-large's best of 10; adding them to CrystaLLM gives a lower-energy structure for 34 compositions and never a worse one.",
      "On genuinely new materials, K-Ont templates are a useful complement to the LLM.")}
{step("D2. Independent check by a teammate (Step 50)",
      "Sent the 34 winning structures (and CrystaLLM's for the same compositions) to a teammate, who computed the energy above the convex hull (a standard stability measure).",
      f"K-Ont template lower in {eh['kont_lower']} of {eh['n']}; median {eh['median_ehull']['kont']:.3f} vs {eh['median_ehull']['crystallm']:.3f} eV/atom; {eh['thresholds']['<= 50 meV']['kont']} vs {eh['thresholds']['<= 50 meV']['crystallm']} plausibly stable (within 50 meV/atom).",
      "An independent calculation agrees with our ranking. Still to confirm: the exact method used, one broken case (LiCo<sub>2</sub>BiO<sub>4</sub>), and the compositions where CrystaLLM won.")}

<h3 class="phase">Phase E · Making K-Ont itself better</h3>
{step("E1. Does K-Ont's starting knowledge help retrieval? (Step 42)",
      "Trained the template retriever from K-Ont and from a plain encoder on 5–100% of the data, 3 seeds each.",
      "K-Ont helps only with very little data (5%: +3.4 points on space group); from 50% data both are equal.",
      "For template choice alone, what matters is training on the KG relation, not K-Ont's starting point. K-Ont's specific value has to come from doing several jobs in one model (E3).", "mid")}
{step("E2. GALEN, OnT's own benchmark (Step 49)",
      "Tested whether K-Ont still does OnT's original job on a large medical ontology (GALEN), using OnT's own evaluation.",
      f"Prediction task: K-Ont {g('prediction','K-Ont v2 (Step 36)')['MRR']:.3f} vs OnT {g('prediction','OnT-galen (published)')['MRR']:.3f} (ranking quality kept). Harder inference task: about half lost ({g('inference','K-Ont v2 (Step 36)')['MRR']:.3f} vs {g('inference','OnT-galen (published)')['MRR']:.3f}).",
      "Learning batteries cost part of the old knowledge on the hardest task — fixed next.", "mid")}
{step("E3. One K-Ont model for everything (Steps 43, 46, 51, 52, 55)",
      "Tried several ways to add the template-retrieval skill without forgetting: a 'retention' term that keeps K-Ont's original answers stable while it learns the new relation; training it natively; hard examples; and finally protecting the GALEN concepts and the short relation phrases too (version 52c).",
      f"Version 52c: crystal typing {c52['crystal typing H@1']:.0%}, role {c52['role, no label line (test)']:.0%}, family {c52['family, no label line (test)']:.1%}, hierarchy 0.97, template retrieval {fw['template']['fw_retain_galen_anchor']['found@1']:.1f}% at the first try, GALEN prediction back to OnT's level ({g('prediction','K-Ont + GALEN + anchor retention (Step 52c)')['MRR']:.3f}); same on 3 seeds. Without retention, K-Ont forgot material typing completely (0%).",
      "One model can hold the original ontology, the battery ontology + KG and the crystal skill at once — the cleanest K-Ont-specific result. Small costs: material typing 92–95%, part of GALEN inference.")}
{step("E4. Does the ontology matter? (Step 54)",
      "Retrained K-Ont with the ontology rules removed (keeping all KG facts).",
      "Material and crystal typing fell from ~100% to 64%, the hierarchy score from 0.97 to 0.31; battery roles, families and template retrieval stayed.",
      "The 'Ont' in K-Ont is what gives it type reasoning. The KG facts give roles, families and retrieval.")}
{step("E5. A bigger KG (Step 56)",
      "Built 5,901 more Materials Project materials on the laptop in 3 memory-safe chunks (all validated), of which 5,267 became usable templates, and re-ran the template test on the same 902 materials with the same K-Ont model (no retraining).",
      "Materials with a correct template available: 85.6% → 90.9%. Found within 3 tries: 66.1% → 69.5% (+3.4, significant); within 8 tries: 72.6% → 76.1% (+3.4, significant). The first try did not change (54.8% → 54.9%).",
      "More knowledge in the KG directly means more crystals can be built correctly. The first guess doesn't improve because more candidates also means more look-alikes to choose between.")}

<h2 class="pb">Part 3 — How to read the results, in one table</h2>
<table>
<tr><th>Question</th><th>Answer</th><th>Evidence</th></tr>
<tr><td>Does K-Ont learn ontology + KG knowledge?</td><td class="good">Yes</td><td>98.8–100% on held-out materials (A1)</td></tr>
<tr><td>Is it better than OnT for individuals?</td><td class="good">Yes</td><td>Heart disease +2.2 AUC (A2)</td></tr>
<tr><td>Does the ontology itself matter?</td><td class="good">Yes, for type reasoning</td><td>100% → 64% typing without it (E4)</td></tr>
<tr><td>Can one K-Ont do everything?</td><td class="good">Yes</td><td>52c, stable over 3 seeds (E3)</td></tr>
<tr><td>Does injecting K-Ont into an LLM help?</td><td class="mid">Only a little</td><td>+1.5 at best; ceiling under 3 points (B1–B3)</td></tr>
<tr><td>Do K-Ont-chosen templates beat CrystaLLM?</td><td class="good">Yes</td><td>+12.2 first try; +5.7 single answer (C1, C3)</td></tr>
<tr><td>…and the standard substitution method?</td><td class="good">Yes, at the first try</td><td>+8.4 (C2)</td></tr>
<tr><td>Do they help for new materials?</td><td class="good">Yes, as a complement</td><td>34 better / 0 worse; 32/34 confirmed (D1, D2)</td></tr>
<tr><td>Is K-Ont better than a plain encoder for choosing templates?</td><td class="bad">No (equal)</td><td>60.5% vs 60.5% (C1, E1)</td></tr>
</table>

<h2>Part 4 — Limitations, honestly</h2>
<ul>
<li><b>Rule-based labels:</b> the KG's battery roles and structure families come from rules, so very high scores partly mean K-Ont reproduced those rules.</li>
<li><b>K-Ont vs a plain encoder:</b> for choosing templates alone, an ordinary encoder trained on the same KG relation does as well. K-Ont's distinct value is doing it in the same model as the ontology reasoning.</li>
<li><b>Known materials:</b> most tests use Materials Project materials, which favours generators that memorised them; the new-cathode test addresses this only with energies.</li>
<li><b>Energies, not experiments:</b> stability comes from CHGNet and the teammate's calculation (method to be confirmed), not from synthesis in a lab.</li>
<li><b>Templates cannot invent new frameworks:</b> they only reuse frameworks already in the KG; the bigger KG (E5) raised coverage from 85.6% to 90.9%, but about 9% of materials still have no correct template.</li>
<li><b>One crystal domain:</b> all crystal experiments use one battery KG; the heart-disease study is a second domain only for the representation part.</li>
<li><b>Small costs in the one-model version:</b> material typing 92–95% and part of GALEN's hardest task.</li>
</ul>

<h2>Glossary</h2>
<dl>
<dt>AUC</dt><dd>Area under the ROC curve: 0.5 = random, 1.0 = perfect classifier.</dd>
<dt>Confidence interval (CI)</dt><dd>The range the true difference probably lies in; if it does not include 0, the difference is real.</dd>
<dt>CIF</dt><dd>Crystallographic Information File — the standard text format for a crystal structure.</dd>
<dt>CHGNet</dt><dd>A machine-learned physics model that estimates a crystal's energy and relaxes its atoms to a stable position.</dd>
<dt>Energy above hull</dt><dd>How far a structure is above the most stable possible combination of its elements; ≤ ~0.05 eV/atom is usually "plausibly stable".</dd>
<dt>Framework / space group</dt><dd>The geometric arrangement of a crystal; the space group is its symmetry class (230 kinds).</dd>
<dt>Hyperbolic space / Poincaré ball</dt><dd>A curved space where there is much more room far from the centre — ideal for tree-like hierarchies.</dd>
<dt>MRR</dt><dd>Mean reciprocal rank: 1.0 means the right answer is always ranked first.</dd>
<dt>TBox / ABox</dt><dd>The ontology's rules about concepts (TBox) and facts about individuals (ABox).</dd>
</dl>
"""
    css = f"""
@page {{ size: A4; margin: 17mm 16mm 17mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; color: #1d2430; font-size: 10pt; line-height: 1.5; margin: 0; }}
h1 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 25pt; line-height: 1.15; margin: 0 0 8pt; }}
h2 {{ font-family: Cambria, Georgia, serif; font-weight: 400; font-size: 17pt; margin: 16pt 0 8pt; border-bottom: 1px solid #d5d9df; padding-bottom: 3pt; break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }}
h3.phase {{ font-size: 12.5pt; color: #a5531d; margin-top: 14pt; }}
p {{ margin: 0 0 6pt; }} ul, ol {{ margin: 0 0 6pt 16pt; padding: 0; }} li {{ margin-bottom: 3pt; }}
.cover {{ height: 255mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }}
.kicker {{ font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; color: #a5531d; }}
.lead {{ font-size: 12pt; color: #5b6472; max-width: 160mm; }}
.small {{ font-size: 8.6pt; color: #5b6472; }}
.summary {{ border: 1px solid #d5d9df; border-radius: 6px; padding: 10pt 14pt; background: #f7f8fa; }}
.step {{ border: 1px solid #e3e6ea; border-radius: 6px; padding: 7pt 11pt; margin: 6pt 0 8pt; }}
.step h3 {{ margin-top: 2pt; }}
.tag {{ display: inline-block; font-size: 7.6pt; text-transform: uppercase; letter-spacing: .06em; color: #fff; background: #5b6472; border-radius: 3px; padding: 0 5pt; margin-right: 4pt; }}
.means {{ border-left: 3px solid #1f7a4d; background: #eef6f1; padding: 5pt 9pt; margin-top: 4pt; }}
.means.t-neg {{ border-left-color: #b23b2a; background: #fbf0ee; }}
.means.t-mid {{ border-left-color: #d9762b; background: #fbf4ee; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0; break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #e3e6ea; padding: 4pt 5pt; text-align: left; vertical-align: top; }}
th {{ color: #5b6472; }}
.good {{ color: #1f7a4d; font-weight: 600; }} .mid {{ color: #a5531d; font-weight: 600; }} .bad {{ color: #b23b2a; font-weight: 600; }}
dl dt {{ font-weight: 600; margin-top: 4pt; break-after: avoid; }} dl dd {{ margin: 0 0 2pt 12pt; }}
.avoid {{ break-inside: avoid; }} .pb {{ break-before: page; }}
"""
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>K-Ont study report</title><style>{css}</style></head><body>{body}</body></html>'
    hp = os.path.join(DATA, "kont_study_report.html")
    open(hp, "w", encoding="utf-8").write(html)
    subprocess.run([R.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}",
                    "file:///" + hp.replace("\\", "/")], check=True, capture_output=True, timeout=180)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
