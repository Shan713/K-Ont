# Template editor + limitations plan (Step 58)

Two tracks run side by side. Track 1 fixes the limitations the paper states; Track 2 tests whether a
generator that *edits* a K-Ont-retrieved template beats both the template and CrystaLLM.

## Track 2 — the template editor

### Why
Step 40 put K-Ont knowledge *into* CrystaLLM as a vector or a header and gained at most ~2 points: the model
still writes every coordinate from scratch. Step 41 used the template *as* the answer: 60.5% right at the first
attempt, but only when the template's framework is exactly right. For 85.6% of targets (90.9% with the bigger
KG) some KG framework is right, yet the first template is right for 60.5% — many misses are near-misses
(right framework, wrong site assignment, distorted cell). An editor reads the template and writes a corrected
structure: retrieval supplies the framework, the language model supplies the corrections.

### Input and output
CrystaLLM-small has a 1,024-token context; a CIF is ~376 tokens (median, p90 ~556), so template + target in
full does not fit. The template enters as a **compact draft** — only CIF lines CrystaLLM already knows:

```
_symmetry_space_group_name_H-M <sg of the substituted template>
_cell_length_a ... _cell_angle_gamma ...        (6 lines)
<atom-site rows: El label multiplicity x y z 1> (one per symmetry-distinct site)
data_<target formula>                            <- the target CIF starts here (CrystaLLM layout)
...
```
Loss only on the target part. At test time the prompt is draft + `data_<formula>` + the atom-properties
block (as CrystaLLM's own prompt), and the model writes the rest.

### Training pairs (no evaluation composition anywhere)
From CrystaLLM's training split (`cifs_v1_train.pkl.gz`, read with `safe_pickle`), every CIF whose reduced
formula is a K-Ont-unseen KG material or a novel composition removed (`scratch_gpt.banned_formulas()`, as
Step 45). Group the rest by anonymous formula; for a target CIF take templates of the same anonymous formula
and a different formula:
- **near** template: the most chemically similar formula (`kg_condition2.similarity`) — what retrieval gives;
- **random** template of the same formula type — teaches the editor to *change* a wrong framework.
Each template is substituted exactly as `kg_template_gen` (elements by amount then electronegativity,
volume scaled by atomic radii) and written as a draft. Pairs whose draft + target exceed 1,024 tokens are
dropped. Target ~150k pairs (+ 3k validation).

### Models (two bases, as Step 45)
- **editor-MP**: fine-tuned from CrystaLLM-small (pre-trained on Materials Project; may recall test structures);
- **editor-scratch**: fine-tuned from the Step 45 from-scratch GPT (never saw an evaluation composition) — the
  controlled result.

### Evaluation (902 K-Ont-unseen targets, same matcher 0.2 / 0.3 / 5)
Drafts from the same K-Ont ranker top-3 templates as Steps 41/48. Arms at equal budget:
| arm | candidates |
|---|---|
| template (Step 41) | top-k templates |
| CrystaLLM (Step 40) | k samples |
| editor | k samples, cycling over the top-3 drafts |
| editor + templates | as the Step 48 mix |
Metrics: found@1/3/8; single answer by CHGNet energy on the 300-target subset; paired bootstrap.
Pass criterion fixed now: editor found@1 > template found@1 (60.5) with the 95% interval above 0, on
editor-scratch.

## Track 1 — limitations

| Limitation (paper Sec. VI) | Action | Status |
|---|---|---|
| Choice among candidates loses 10 points | learned picker: +1.2 (n.s.); misses are near-ties under CHGNet resolution -> re-relax near-ties 500 steps | refine queued after the malware run |
| 3 templates + 5 CrystaLLM was not tuned | 1-2 templates + 6-7 CrystaLLM look better (72.0 vs 69.3) on the 300 used to find it -> confirm on the other 602 targets | relaxations queued after refine |
| Template coverage (~9% no correct candidate; no templates for >5-element systems) | mixed-site templates: known frameworks with 2-4 metals ordered over equivalent sites | design after the editor pairs |
| E_hull check was on 34 CHGNet-selected compositions | unbiased random sample of templates and CrystaLLM structures for the teammate's calculation | zip on request |
| Second domain only heart (303 patients) | PE-malware ontology, 1,000 files, same protocol | running (GPU) |
| Picker / mix numbers on 300 targets | extend to all 902 with the relaxations above | queued |
| Rule-based role / family labels | cannot fix without expert labels; stays stated | — |

Machine rule: one GPU job at a time, no CPU pools during GPU training.
