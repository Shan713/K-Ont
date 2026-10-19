# Paper outline (draft) — K-Ont: one hyperbolic ontology + knowledge-graph encoder that reasons over the ontology and guides crystal generation

Status: draft from BUILD_LOG Steps 36–55 (2026-10-19). All tables filled.

## Claim (what the evidence supports)

1. **K-Ont** extends OnT (hyperbolic ontology embeddings) from TBox classes to KG individuals (ABox): materials, crystals, elements and their relations, in one Poincaré-ball encoder.
2. On a 5,000-material battery KG it learns the ontology and the facts (typing, roles, families, hierarchy: 98.8–100%), and as a representation it beats OnT in a second domain (heart disease).
3. Trained on one more KG relation, *"has the framework of"*, **the same K-Ont model** also retrieves crystal frameworks for unseen compositions, keeps every battery-ontology check AND OnT's original GALEN benchmark (retention loss covering KG sentences, GALEN concepts and K-Ont's relation phrases; Step 52c, stable over 3 seeds).
3b. **The ontology matters**: without its TBox axioms K-Ont types materials/crystals correctly 64% instead of ~100% and loses the hierarchy (0.97 → 0.31) (Step 54).
4. A **K-Ont-guided template generator** (retrieve framework → substitute elements) beats a fine-tuned crystal LLM (CrystaLLM) and the standard substitution method (Hautier et al.; +8.4 at the first attempt) on 902 held-out materials; combined with CrystaLLM and a CHGNet check it commits to one answer that is right 69.3% vs 63.7% (+5.7) at the same budget; on new cathodes an independent calculation puts the K-Ont template lower above hull in 32/34.
5. Negative result, reported as a finding: **pushing K-Ont knowledge into a crystal LLM** -- as a vector (soft prompt, Steps 40/45) or as the retrieved template's text (hard prompt, Step 53) -- adds under 3 points even with perfect knowledge (on CrystaLLM and on a GPT trained from scratch without the test compositions); K-Ont retrieval + direct construction + physics selection is the effective design.

What we do **not** claim: that K-Ont's pre-trained weights beat a plain encoder as a retriever starting point (equal at full data, better only at 5% data); that the KG labels are expert-curated (they are rule-based).

## 1 Introduction
- Ontologies/KGs hold materials knowledge; crystal generators (CrystaLLM, MatterGen) ignore it.
- Question: can an ontology-aware embedding of a KG make crystal generation better, and how should it be used?
- Contributions: (i) K-Ont, ABox extension of OnT; (ii) evaluation on a battery KG + a medical dataset; (iii) one-model ontology + framework retrieval; (iv) template generation beating CrystaLLM; (v) a careful negative result on embedding injection with its cause.

## 2 Related work
OnT / HiT (hyperbolic ontology embeddings); KG embeddings (TransE, box/hyperbolic); crystal generation (CrystaLLM, CDVAE, MatterGen, DiffCSP); substitution/prototype-based structure prediction (Hautier et al.); retrieval-augmented generation for materials.

## 3 Method
3.1 Battery KG (battGPT pipeline, 5,000 materials; pipeline figure from the deck; faculty-verified pipeline).
3.2 K-Ont: TBox rows (OnT) + ABox type rows + existential relation rows, in-batch negatives, Poincaré ball; verbalisation of individuals.
3.3 Framework relation + retention loss (Step 43).
3.4 Learned ranker (prompt features, KG facts, retriever score) and template generator (Step 41).
3.5 Injection into CrystaLLM (prefix + gated adapters, counterfactual framework training) — for the negative result (Step 40).

## 4 Experiments
**Splits.** K-Ont trained on 3,982 materials; 1,018 held out. Crystal experiments: pool and all training = K-Ont-seen materials; evaluation = all 902 K-Ont-unseen KG materials with a same-formula-type candidate. Nothing evaluated was in any training (K-Ont, retriever, ranker [out-of-fold], generator).

**Table 1 — K-Ont on the battery KG (Step 36, held-out where applicable).** material typing 100%, crystal typing 99.7%, role (label line removed) 100% vs 84% majority, family 98.8% vs 47%, TBox MRR 0.97.

**Table 2 — Second domain: heart disease (Steps from the heart report).** OnT vs K-Ont embeddings in 4 ML models + MLP, 10-fold: K-Ont +2.2 AUC (4/5 significant); missing columns +4.3 (5/5); few labels −4.0 at 10% (report honestly).

**Table 3 — One model for both (Step 43).**

| | K-Ont | + framework relation | + framework relation + retention |
|---|---|---|---|
| material / crystal typing | 1.000 / 0.997 | 0.0004 / 0.769 | 0.999 / 1.000 |
| role / family (held-out, no label) | 1.000 / 0.988 | 0.944 / 0.954 | 1.000 / 1.000 |
| TBox MRR | 0.969 | 0.969 | 1.000 |
| template found@1 / @3 | – | 57.8 / 68.4 | 55.0 / 66.5 |

**Table 4 — Template generation vs CrystaLLM (Step 41), found@k on 902 held-out materials.**

| | @1 | @3 | @8 |
|---|---|---|---|
| CrystaLLM fine-tuned | 48.3 | 61.2 | 68.3 |
| template, composition rule | 51.3 | 55.8 | 60.8 |
| template, KG ranker w/o K-Ont | 56.3 | 63.2 | 68.4 |
| **template, K-Ont ranker** | **60.5** | **66.9** | **71.4** |
| template, plain-encoder ranker (control) | 60.5 | 67.6 | 72.7 |
| K-Ont template, CHGNet-relaxed | 60.8 | 67.3 | – |
| 3 templates + 5 CrystaLLM (budget 8) | | | **77.3** (+9.0 [6.9, 11.1]) |

**Table 4b — One committed answer per material (Step 48, random 300 of the 902).** CHGNet lowest-energy pick: CrystaLLM x8 63.7, K-Ont template top-1 61.7, **3 K-Ont templates + 5 CrystaLLM 69.3 (+5.7 [+1.3, +9.7] vs CrystaLLM x8)**; any-of-8 ceiling 79.3.

**Table 5 — Novel cathode compositions (Step 44, not in Materials Project).** Lowest CHGNet energy per composition vs CrystaLLM-large's best of 10 relaxed samples (5 meV/atom tie band), 199 compositions.

| method − CrystaLLM-large | lower | tie | higher | mean [95% CI], meV/atom |
|---|---|---|---|---|
| K-Ont templates, top 3 | 34 | 126 | 39 | +1.3 [−4.1, +6.4] |
| plain-encoder templates, top 3 | 34 | 113 | 52 | +2.9 [−2.6, +8.0] |
| parent's framework (1) | 25 | 145 | 25 | −2.8 [−7.7, +1.5] |
| **K-Ont top 3 + CrystaLLM** | **34** | 165 | **0** | **−7.5 [−11.8, −4.0]** |

**Table 5b — Independent energy-above-hull check (Step 50; method to be confirmed).** The 34 compositions where CHGNet favoured the K-Ont template: K-Ont template lower e_hull in 32/34; median 0.025 vs 0.088 eV/atom; within 50 meV/atom 23 vs 13; on/below hull 3 vs 1. Selected by CHGNet, so it confirms the ranking; the unbiased set (CrystaLLM-won + ties) is pending.

**Figure — Learning curve (Step 42).** K-Ont vs plain start, 5–100% data, 3 seeds: K-Ont better at 5% (SG +3.4 [0.5, 6.3]), worse at 10–25% (−2.5), equal from 50%; K-Ont steadier across seeds.

**Table 6 — Injection into CrystaLLM (Step 40), the negative result.** plain fine-tune 47.5; K-Ont-steered 44.1; random analogue 22.7; true analogue (ceiling) 49.6; plain-encoder vectors 44.9. The vector steers (+21.4 vs random) but the ceiling is +2.1: CrystaLLM recalls Materials Project structures, so conditioning can't add much; retrieval + templates is the effective route.

**Table 7 — Injection without memorisation (Step 45).** CrystaLLM-architecture GPT pre-trained from scratch on 300k CrystaLLM training CIFs with every evaluation composition removed; same arms. Plain 20.6, K-Ont-ranked analogue 22.1 (**+1.5 [+0.5, +2.5], passes**), true analogue 23.3 (ceiling +2.7), no vector 17.4, random 11.1. K-Ont injection helps causally once memorisation is removed, but the vector channel is narrow; templates reach 60.5%.

**Table 8 — One K-Ont for everything (Steps 49, 52c, 54, 55).** GALEN (OnT paper's benchmark) prediction MRR / AUC: OnT-galen 0.339 / 0.978, K-Ont v2 0.335 / 0.952, **52c 0.334–0.339 / 0.975–0.976 (3 seeds)**; inference 0.022 / 0.973 → 0.014 / 0.95. 52c battery checks: crystal typing 1.000, role 0.99–1.00, family 0.988, TBox MRR 0.969, material typing 0.92–0.95; crystal retrieval @1 54.2–54.8. Ablation without TBox axioms: typing 0.64 / 0.64, TBox MRR 0.31.

**Table 9 — Against the standard substitution method (Step 47).** found@1/2/3/5/8: K-Ont 60.5 / 64.9 / 66.9 / 69.4 / 71.4; Hautier 52.1 / 60.3 / 64.5 / 68.1 / 70.1 (K-Ont − Hautier +8.4 [5.5, 11.2] @1).

**Table 10 — Hard prompt (Step 53).** CrystaLLM seeded with the K-Ont template's CIF header 45.9 vs the template itself 60.5 (−14.6) and CrystaLLM unseeded 48.3.

## 5 Discussion / limitations
- KG labels are rule-based; the KG pipeline is verified by faculty but not expert-curated.
- Retrieval gains come from KG-trained retrieval (a plain encoder trained the same way chooses templates as well); K-Ont's distinct value is doing it in the same model as the ontology reasoning (52c) and the ontology is what gives that reasoning (54).
- GALEN inference is only partly recovered (MRR 0.014 vs OnT's 0.022).
- Material typing 0.92-0.95 in the one-model recipe.
- Evaluation on known MP materials favours memorising generators; the novel-composition test (Table 5) addresses this only energetically (CHGNet), not with DFT.
- Single KG domain for crystals (battery materials).

## 6 Reproducibility
All scripts in `abox/` (kg_pool, kg_retriever_ft, kg_rerank, kg_template_gen, kg_condition3, kg_learning_curve, kont_framework, novel_template); BUILD_LOG Steps 36–44; seeds and splits fixed; pre-set acceptance criteria stated before results.

## Still to do before submission
1. Confirm the teammate's e_hull method (DFT?) and recheck LiCo2BiO4; extend the check to the compositions where CrystaLLM won (unbiased).
2. Bigger KG (more templates) on the workstation (laptop RAM 15.7 GB is too small to extract a 15k KG).
3. Target venue choice (e.g. a materials-informatics journal such as *npj Comput. Mater.* / *Digital Discovery*, or an ML-for-science workshop) — the guide's call.
4. Figures: KG pipeline, K-Ont geometry (Poincaré disc), template generator diagram, learning curve, per-method bars.
