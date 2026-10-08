# Handoff: KG-driven crystal generation (read this first)

Written 2026-10-08 on the laptop for the Claude session on the workstation. It carries the context of a
long laptop session: the goal, what has been measured, the scripts, the traps already hit, and the next
tasks (A and B-data-prep) with acceptance criteria. Full history: `docs/BUILD_LOG.md` (Steps 22-32).

## 1. Goal and the novelty claim

The project's novelty is **a KG/ontology-conditioned language model for crystal generation**, applied to
novel battery cathodes. The base generator is CrystaLLM (Antunes et al. 2024; the authors' `crystallm`
package, sparse-cloned into `K-Ont/CrystaLLM`; checkpoints `crystallm_v1_small` / `_large` in
`data/crystallm/`). The KG is battGPT's battery-materials KG (5,000 Li/Na materials from Materials
Project, plus battGPT's ontology: structure families, battery roles, bonds, coordination, oxidation states).

The claim we are building evidence for:

> A KG-driven CrystaLLM generates crystal structures that satisfy domain requirements specified through
> the ontology (structure family / framework, battery role, chemical plausibility, stability) more often
> than the unconditioned model, without losing validity.

It is assembled from three levers, each a different way the KG drives generation:
- **A. Retrieval / seeding** -- the ontology retrieves analogues; generation is seeded with the analogue's
  framework. (Task A below.)
- **B. KG as reward** -- KG plausibility + ontology compliance + CHGNet stability as a reward; CrystaLLM is
  preference-fine-tuned (DPO) so its *raw* samples move towards them. (Task B: data prep now, training later.)
- **C. Ontology-constrained decoding** -- later.
- **D. KG-conditioned diffusion** -- a diffusion generator built for conditioning (MatterGen-style
  adapters), conditioned on KG/ontology labels and the OnT embedding. (Task D below; the longest job.)

**Who runs what (decided 2026-10-08):** the **workstation** runs Task D (setup now, training once the GPU
is free) and later A-novel; the **laptop** runs A-test (22 test cathodes) and B data prep.

The judging metric is **compliance with KG/ontology conditions** (plus validity, plausibility, stability),
not only exact recovery of a known structure: CrystaLLM already recovers most known structures alone.

## 2. What has been measured (all numbers verified)

Test set: CrystaLLM's own held-out test split (`data/crystallm/test_cifs`, from `cifs_v1_test.tar.gz`),
filtered to Li/Na, 2-5 elements, <= 40 atoms: 971 compositions, 55 "cathode-like" (transition metal + O/S/F).

| finding | number | source |
|---|---|---|
| CrystaLLM large, composition only, 55 cathode-like | 44/55 found, 64% of CIFs match | `phaseC_nearmiss_large.json` |
| + true space group (ceiling of any SG hint) | 50/55, 81% | same |
| + our random-forest SG guesses | 28/55 (hurts; CrystaLLM's own SG choice is right 70%) | same |
| KG parent's SG as hint (test cathodes with a plausible analogue) | right for 10/22, never where CrystaLLM failed | `test_analogues.json` |
| pipeline calibration, 95 non-cathode | 87/95 (composition), 93/95 (true SG) | `phaseC_nearmiss_calib_large.json` |
| CHGNet lowest-energy pick is the right structure | 39/52 (composition), 47/54 (true SG) | `phaseC_chgnet_large.json` |
| KG plausibility top-1 vs random pick (cathode-like) | +8.2 [+0.9,+16.6], +10.3 [+1.5,+19.4], +1.7 [-6.1,+9.5] (3 sample sets, bootstrap 95% CI) | Step 32 |
| KG filter precision (match rate among KG-plausible vs all) | +6 to +20 points, cathode domain only | `kg_steer_eval.json` |
| KG priors on intermetallics (non-cathode set) | mislead: top-1 41% vs random 46% | same |
| resampling until 5 KG-plausible | no gain (67.9% vs 67.7%) at 2.7x samples -- dropped | `kg_steer_sample_large.json` |
| KG filter then CHGNet | = CHGNet (75%) with 10-22% fewer relaxations | Step 31 |
| stability screen (CHGNet on MP phase diagram) on 60 known cathodes | 10/10 unstable flagged, 21/24 stable within 0.05; rule e_hull <= 0.05 | `stability_calib.json` |

Bottom line: text hints about the structure don't beat CrystaLLM's own knowledge; the KG helps with
domain-specific selection and screening. The levers A/B are how we try to make the KG drive generation.

## 3. Repo map

| file | what it does |
|---|---|
| `abox/phaseC_generate.py` | CrystaLLM generation (batched, seeded per material x prompt), post-processing, `load_truth()`, `sg_symbol()`, prompts `composition`, `oracle`, `rf_top1/5`, `parent_sg`; `--candidates`, `--start`, novel mode (no reference) |
| `abox/phaseC_nearmiss.py` | per-CIF: right composition, right SG, strict/loose match, volume ratio |
| `abox/phaseC_chgnet.py` | CHGNet relaxation vs the known structure (dE, match after relax) |
| `abox/kg_priors.py` -> `data/crystallm/kg_priors.json` | KG chemistry priors: bond lengths (875 element pairs), coordination numbers, per-element volumes |
| `abox/kg_steer.py` | `kg_score(structure)` (bond z + coordination NLL + 3 x volume + 5 x clash; `plausible` flag) and the selection evaluation |
| `abox/kg_steer_sample.py` | resampling until KG-plausible (done; no gain) |
| `abox/novel_propose.py` | (laptop, MP key) 200 novel substitution candidates -> `novel_candidates.json` |
| `abox/novel_queue.py`, `novel_chgnet.py` | the novel pilot (running/ran on the workstation) |
| `abox/novel_steer_analysis.py` | KG plausibility + `keeps_parent_framework` (StructureMatcher.fit_anonymous vs parent) + shortlist |
| `abox/stability_hull.py` | (fetch on laptop with MP key; compute anywhere) e_above_hull from CHGNet on the MP phase diagram |
| `data/crystallm/novel_parents.json` | parents of the novel candidates: CIF, SG, ontology structure family |
| `data/crystallm/test_analogues.json` | 22 test cathodes with a one-swap KG analogue (TM<->TM or Li<->Na): analogue CIF, SG, swap |
| `data/crystallm/kg_training_source.json.gz` | all 5,000 KG materials: CIF, SG, crystal system, structure_family, battery_role, e_above_hull, band gap, and `split` (train 3,905 / val 497 / test 493 / excluded 105) |

Environments: `.venv-crystallm` (torch+CUDA, pymatgen, chgnet, smact, sklearn) for everything here.
The **Materials Project key exists only on the laptop** (`battGPT/.env`, never committed). Anything that
needs MP (novelty checks, MP phase-diagram fetch) is done on the laptop; the workstation gets files.

## 4. Traps already hit (don't repeat them)

1. **Test-set reference CIFs are asymmetric units** with a placeholder `'x, y, z'` operator, and indented
   by pymatgen. Always load them with `phaseC_generate.load_truth()` (collapses whitespace, then
   CrystaLLM's `postprocess`). Reading them directly made every match count wrong (BUILD_LOG Step 26).
2. **pymatgen 2026 writes monoclinic space groups in full form** (`P12_1/c1`); CrystaLLM's vocabulary uses
   short form. Always go through `phaseC_generate.sg_symbol(number)`.
3. **`SymmOp.as_xyz_string` was renamed** in current pymatgen; `phaseC_generate` adds the alias on import.
   Import it before using CrystaLLM's helpers.
4. **Use conventional standard cells** for prompts and references (`SpacegroupAnalyzer(s, symprec=0.1)
   .get_conventional_standard_structure()`); MP's primitive/odd cells made the sanity set fail.
5. **CHGNet energies already include MP2020 corrections.** Never apply `MaterialsProject2020Compatibility`
   on top (it put 58/60 known cathodes on the hull).
6. **KG priors are for ionic Li/Na compounds.** Don't trust the KG score on intermetallics.
7. **Sample sizes are small (~50 compositions).** Report bootstrap 95% CIs over compositions for every gain.
8. Large model, batch 10, fits a workstation GPU but not the laptop's 8 GB on 32-atom cells.
9. Seeds are fixed per material x prompt (`--seed`); reuse them so runs are reproducible.
10. **Commits: no Claude co-author line** (the user's standing instruction). Commit only result/code files;
    add `.gitignore` exceptions (`!/data/crystallm/<name>`) for new result files. Never overwrite earlier
    result files -- new names.

## 5. Task A: KG-seeded generation (do this first)

Question: does seeding CrystaLLM with a KG analogue's framework make outputs comply with the ontology
requirement "keep the analogue's framework" (and be plausible/stable), compared with composition-only, and
does generation add anything over plain substitution?

**Sets**
- A-novel: the 200 novel candidates (`novel_candidates.json`; parent in `novel_parents.json`).
- A-test: the 22 test cathodes in `test_analogues.json` (truth known: also report exact match).

**Prompt levels** (10 samples each, large model, seeded):
- L0 `composition` -- `data_<cell>` only (exists).
- L1 `parent_sg` -- + the analogue's space group (exists for A-novel; add for A-test).
- L2 `parent_lattice` -- L1 + the analogue's six lattice parameters, scaled to the target composition:
  lengths x (V_target / V_analogue)^(1/3), angles unchanged, with V_target = V_analogue x
  V_KG(target cell) / V_KG(analogue cell), V_KG = sum(n_element x volume_per_atom[element]) from
  `kg_priors.json`. Write them exactly as CrystaLLM's CIFs do, after the space-group line:
  `_cell_length_a 6.3898` ... `_cell_angle_gamma 90.0000` (4 decimals); the prompt ends there and
  CrystaLLM writes the rest (atom sites).
- L3 `substitution` (control, no generation) -- the analogue's conventional structure with the element
  swapped, lattice scaled as in L2, relaxed with CHGNet. If L3 matches or beats L2 on every metric,
  generation adds nothing beyond substitution, and the write-up must say so.

The analogue's cell must be the **conventional standard cell** (trap 4); the target cell composition is the
analogue's conventional cell with the swap applied (for A-novel this equals `novel_candidates[i].cell`).

**Metrics per level** (per CIF and per composition; reuse `kg_steer.kg_score`, `novel_steer_analysis.analyse`):
valid %, right composition %, KG-plausible %, keeps analogue framework % (fit_anonymous), CHGNet relaxed
energy (lowest per composition), and for A-test strict match vs truth (`load_truth`). Bootstrap CIs.
Export the lowest-energy structure per composition and level so the laptop can run the MP stability screen.

**Acceptance**: a table level x metric for A-novel and A-test, with CIs; files
`data/crystallm/seeded_results_*.json`, `seeded_analysis.json`, a CIF tarball of best structures; BUILD_LOG
Step 33. Keep the L3 control.

## 6. Task B: KG-reward preference fine-tuning -- data prep only for now

Goal (later): fine-tune CrystaLLM (small first) with DPO so its raw samples are more often valid,
KG-plausible, framework-consistent and low-energy, without any selection afterwards.

**B1. Supervised corpus in CrystaLLM's format** from `kg_training_source.json.gz` (`train` and `val` only):
conventional standard cells; use CrystaLLM's own preprocessing so the format matches its training data
(the authors' repo has it under `bin/` -- extend the sparse checkout: `git -C CrystaLLM sparse-checkout add
bin`; read `bin/preprocess.py` and `bin/tokenize_cifs.py` before using them). Check every CIF tokenizes with
`CIFTokenizer` and fits the model's block size (1024 tokens); report how many were dropped and why.

**B2. Preference pairs.** For up to ~1,000 `train` compositions (<= 40 atoms; stratify so roles/families
are represented) generate 8 samples each with CrystaLLM **small** (composition prompt, seeded). For each:
valid, right composition, `kg_score`, match to the KG's own structure (truth: the KG CIF), CHGNet relaxed
energy minus the KG structure's relaxed energy (dE). Reward (fixed in advance, write it down before looking
at results): invalid or wrong composition = worst; otherwise r = -kg_total - 10 x max(0, dE). Pairs per
composition: highest-r vs lowest-r sample if r differs by > 0.5. Same for `val` compositions (~150) as a
held-out preference set. Save `data/crystallm/dpo_pairs_{train,val}.jsonl` (prompt, chosen CIF, rejected
CIF, both rewards and components).

**B3. Write (don't run) the training/eval protocol** in BUILD_LOG: DPO on CrystaLLM small (reference = the
original model), beta 0.1; evaluate on `test` compositions with raw samples only: valid %, KG-plausible %,
framework/structure match vs the KG structure, CHGNet dE distribution. Controls: the original model, and
supervised fine-tuning on the chosen samples only (separates "more battery data" from "preference").

**Acceptance for now**: B1 corpus + stats, B2 pair files + stats (how many pairs, reward distribution), B3
written. Commit and push; report the numbers.

## 7. Task D: KG-conditioned diffusion (workstation; start now)

Goal: a diffusion model that generates crystals **conditioned on KG/ontology attributes** -- the most
direct form of the project's claim. Steps, each to be reported before the next:

**D0. Choose and verify the model (no installs yet).** Candidate: MatterGen (Microsoft; property-conditioned
diffusion with fine-tuning adapters). Check, from the official repository: licence, released checkpoints,
how to fine-tune on a new conditioning property (categorical and continuous), GPU memory and time for
fine-tuning, the data format it expects, Python/CUDA requirements. If MatterGen is unsuitable, compare
alternatives briefly (e.g. DiffCSP/DiffCSP++, CrystalFormer) and recommend one. Write the findings into
BUILD_LOG Step 34 and stop for the user's go-ahead before installing (installs and checkpoint downloads
need the user's OK: name, source and size).

**D1. Environment** in its own venv (e.g. `.venv-diffusion`), never touching `.venv-crystallm`.

**D2. Data** from `data/crystallm/kg_training_source.json.gz` (`train` / `val` / `test` splits as given;
never train on `test` or `excluded`): conventional standard cells, <= 40 atoms. Conditioning labels per
material:
- `battery_role` (categorical: PositiveElectrode / NegativeElectrode / none) -- 1,795 labelled;
- `structure_family` (categorical, 9 ontology families + none) -- only 445 labelled: report class counts;
- `chemical_system` (as the model supports it);
- optionally the OnT embedding of the material's composition-only sentence as a continuous vector
  (the K-Ont model under `data/runs/`; discuss with the user before adding -- it is the novel part but
  also the riskiest).

**D3. Fine-tune** the pretrained model with adapters on these labels (start with `battery_role` alone, a
short run to check it learns; then add family).

**D4. Evaluate** conditioned vs unconditioned sampling, the same compositions/chemical systems: validity,
**compliance** (requested role/family satisfied -- family via `StructureMatcher.fit_anonymous` against KG
members of that family, role via battGPT's rules or a classifier trained on the KG labels), KG plausibility
(`kg_steer.kg_score`), CHGNet stability, novelty vs the training set; bootstrap CIs. Compare with CrystaLLM
(composition prompt, and the Task A seeded levels) on the same targets.

## 8. Pending from today

The novel pilot (`novel_queue.py`) and its follow-up `novel_steer_analysis.py` (s3) run on the workstation.
When `novel_steer_analysis.json`, `novel_shortlist.md` and a `novel_best.tar.gz` are pushed, the laptop
runs the MP stability screen on the shortlist (needs the MP key).
