# Paper outline (draft) — K-Ont: one hyperbolic ontology + knowledge-graph encoder that reasons over the ontology and guides crystal generation

Status: draft from BUILD_LOG Steps 36–44 (2026-10-17). Numbers marked **[pending]** come from runs still in progress.

## Claim (what the evidence supports)

1. **K-Ont** extends OnT (hyperbolic ontology embeddings) from TBox classes to KG individuals (ABox): materials, crystals, elements and their relations, in one Poincaré-ball encoder.
2. On a 5,000-material battery KG it learns the ontology and the facts (typing, roles, families, hierarchy: 98.8–100%), and as a representation it beats OnT in a second domain (heart disease).
3. Trained on one more KG relation, *"has the framework of"*, **the same K-Ont model** also retrieves crystal frameworks for unseen compositions without losing any ontology ability (with a retention loss).
4. A **K-Ont-guided template generator** (retrieve framework → substitute elements) beats a fine-tuned crystal LLM (CrystaLLM) on 902 held-out materials, and the two combined beat CrystaLLM by +9 points at the same budget.
5. Negative result, reported as a finding: **injecting K-Ont embeddings into CrystaLLM** steers its output but cannot beat plain fine-tuning on Materials Project structures — the model already recalls them (ceiling +2.1 even with the true framework).

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

**Table 5 — Novel cathode compositions (Step 44, not in Materials Project).** Lowest CHGNet energy per composition, K-Ont templates vs CrystaLLM-large's best of 10 relaxed samples; win/tie/loss (5 meV/atom band) **[pending]**.

**Figure — Learning curve (Step 42).** K-Ont vs plain start, 5–100% data, 3 seeds: K-Ont better at 5% (SG +3.4 [0.5, 6.3]), worse at 10–25% (−2.5), equal from 50%; K-Ont steadier across seeds.

**Table 6 — Injection into CrystaLLM (Step 40), the negative result.** plain fine-tune 47.5; K-Ont-steered 44.1; random analogue 22.7; true analogue (ceiling) 49.6; plain-encoder vectors 44.9. The vector steers (+21.4 vs random) but the ceiling is +2.1: CrystaLLM recalls Materials Project structures, so conditioning can't add much; retrieval + templates is the effective route.

## 5 Discussion / limitations
- KG labels are rule-based; the KG pipeline is verified by faculty but not expert-curated.
- Retrieval gains come from KG-trained retrieval; K-Ont's pre-training helps only at very low data.
- Evaluation on known MP materials favours memorising generators; the novel-composition test (Table 5) addresses this only energetically (CHGNet), not with DFT.
- Single KG domain for crystals (battery materials).

## 6 Reproducibility
All scripts in `abox/` (kg_pool, kg_retriever_ft, kg_rerank, kg_template_gen, kg_condition3, kg_learning_curve, kont_framework, novel_template); BUILD_LOG Steps 36–44; seeds and splits fixed; pre-set acceptance criteria stated before results.

## Still to do before submission
1. Fill Table 5 pending numbers (novel-composition run in progress).
2. DFT check of a few top novel candidates (optional, strengthens Table 5).
3. Target venue choice (e.g. a materials-informatics journal such as *npj Comput. Mater.* / *Digital Discovery*, or an ML-for-science workshop) — the guide's call.
4. Figures: KG pipeline, K-Ont geometry (Poincaré disc), template generator diagram, learning curve, per-method bars.
