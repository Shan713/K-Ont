# K-Ont vs FYP_K-OnT: what differs, what was measured, and how to scale up

Comparison of this repo (K-Ont) with a teammate's parallel implementation of the same idea,
[akshayks13/FYP_K-OnT](https://github.com/akshayks13/FYP_K-OnT) (5 commits, last 23 Sep 2026), plus
a plan for training on more materials. Everything marked **measured** was run on this machine
(RTX 4060 laptop) on 24 Sep 2026; nothing here is taken from either repo's own claims without a check.

---

## 1. Same goal, same starting point

Both extend OnT (Yang et al., ISWC 2025) from ontology classes to a populated battery-materials KG,
following the same design spec (`ONT_ABOX_EXTENSION.md`; his copy is an expanded version of the
original). Both keep OnT's model and losses unchanged in principle, verbalize each material as a
labelled `Field: value` sentence, write extra OnT training rows, keep exact numbers as a separate
z-scored vector for later concatenation, and evaluate on held-out materials.

## 2. Where they differ

| | **K-Ont (ours)** | **FYP_K-OnT (teammate)** |
|---|---|---|
| Source KG | 250 materials, 325k triples, ontology v0.3.x (built by battGPT `50e635c`) | 5,000 materials, 8.37M triples, 527 MB TTL, ontology v0.2.2 (5 Aug 2026); not in the repo |
| Material selection | targeted MP searches by structure-family ratio patterns + classifier | first N materials in file order |
| KG reading | rdflib, whole graph in memory (9 s at 325k triples) | streaming block parser, never loads the graph (71 s at 5,000 materials) |
| TBox | 23 classes: structure-family tree + electrode roles, 3 relations | 21 classes: battery-role tree (10 axioms) incl. "Layered Oxide Cathode" etc.; 21 object properties |
| What gets typed | materials, crystals, elements | materials only |
| Crystal sentence | symmetry (+ lattice, volume, site counts in `full`); no material identifier (Step 16) | `"Crystal structure of <formula>"` |
| Geometry variant | `full` and `no_geometry` | one variant (no lattice) |
| hasStructure training | OnT relation rows + our in-batch negatives patch (Step 17) | **is-a rows** `material ⊑ "has structure some <crystal>"` with explicit crystal negatives (same chemical system first), plus relation rows |
| TBox oversampling | repeat to ~1:1 (×18), self-negatives removed | repeat to ~1:1 with **fresh random negatives per copy**, self-negatives excluded |
| Missing numbers | dropped if >50% missing, else value + mask; impossible elastic values dropped | filled with `0.0` (his own §8.2 says not to); no physical-range bounds |
| Base encoder | `Hui97/OnT-MiniLM-L12-galen` (pretrained OnT) | plain `all-MiniLM-L12-v2` |
| Held-out split | 20%, stratified by structure family, verified absent from all training rows | random sample |
| Evaluation baseline | always scores the untrained model too | untrained model for typing only |
| Trained at scale? | 250 materials, GPU, several variants | 25 materials only; 5,000 never run (Mac MPS ran out of memory past ~200) |
| Extra | leak probes, per-step timing, patches for OnT | numeric-as-text probe (§8.1), HGT graph-neural-network pipeline + HGT→CrystaLLM study |

**Worth adopting from his repo:** the streaming extractor (the only practical way to read a
527 MB KG), fresh negatives per oversampled copy, the explicit hard-negative is-a rows for
hasStructure (a data-side alternative to our loss patch), and his numeric probe (§8.1: for the same
sentence, band gap 0 vs 8 eV gives cosine 0.9999 as text, 0.35 with the z-scored vector).

## 3. Head-to-head on the same data (measured)

His extractor, pointed at our KG (`--ttl .../battgpt_kg_ont/battery_kg.ttl --max-materials 250`),
returned the same 250 materials with **zero mismatches** against our `entities.json` on formula,
band gap, formation energy, energy above hull, bulk modulus, site count and electrode role. So the
two pipelines can be compared on identical data. His `train_abox.py` was then run unmodified on it:
51 held out, 3 epochs, batch 32, CUDA. It trained in 220 s (4,700 is-a + 277 relation rows) and reported:

| His check (held-out) | His result | Random |
|---|---|---|
| typed as "substance" | 100% (before: 0%) | — |
| hasStructure MRR, pool of 50 crystals | **1.000** | 0.090 |
| electrode role accuracy (n = 19) | **100%** | 0.25 |

**Both relation numbers are answered by text the question already contains** (`data/fyp_compare/leak_probe.*`):

| | untrained MiniLM | his trained model |
|---|---|---|
| hasStructure MRR, as evaluated | **0.895** | 1.000 |
| — with the material's `Structure:` line removed | 0.765 | 1.000 |
| electrode role accuracy, as evaluated | 0.000 | 1.000 |
| — with the material's `Battery role:` line removed | 0.000 | **0.895** |

- hasStructure: the untrained encoder already scores 0.895, because his crystal sentence is
  `"Crystal structure of NaCeO2"` and the material sentence contains that line and the formula twice
  more. This is the leak we found and fixed in our own data (Step 16).
- Electrode role: without the `Battery role:` line his model scores 0.895, which is exactly the
  majority-class baseline (17 of the 19 held-out materials are cathodes: always answering "cathode"
  gives 17/19). The 100% comes from reading the role back out of the sentence that states it.

Typing (100% held-out) is real in both pipelines. His crowding check (mean pairwise distance
4.66 → 14.90, min 0.40 → 3.02) also shows no collapse.

**Update, Step 21 (measured):** the hasStructure training methods were compared on identical,
leak-free data (BUILD_LOG Step 21). Held-out MRR via rotation / text concept:

| | `full` | `no_geometry` (max 0.724) |
|---|---|---|
| ours (in-batch negatives) | 1.000 / 1.000 | 0.651 / 0.651 |
| his (is-a rows with crystal wrong answers) | 0.863 / 0.948 | 0.637 / 0.651 |
| both combined | 1.000 / 1.000 | 0.651 / 0.651 |

No difference without geometry. With geometry, ours is equal or better, and combining adds nothing.
His method is a valid patch-free alternative. All three still only match information both sentences
state, so polymorph data remains the next step.

## 4. What the comparison says, in short

- **Ours is the more trustworthy pipeline today:** leak-free relation evaluation, untrained baselines,
  stratified and verified held-out split, principled missing-value handling, physically bounded values,
  and a working large-GPU training setup.
- **His is the more scalable pipeline today:** it reads a 5,000-material, 8.4M-triple KG in about a
  minute, and his KG exists. His hasStructure is-a rows are now tested (Step 21): equal to ours
  without geometry, slightly lower with it. His fresh-negatives oversampling is still untested.
- Neither has shown that relation learning beats "matching information stated on both sides"
  (see our Step 19). That needs harder data: polymorphs.

## 5. Plan: training on more materials

**Measured costs used below:** our KG ingestion 250 materials in 370 s (≈1.5 s/material, including MP
fetch and CrystalNN bonds); graph build 16 s for 325k triples; our `full` training ≈1.06 s/step,
`no_geometry` ≈0.62 s/step (batch 32); his pipeline ≈0.50 s/step; about 26 training rows per training
material in our design and 24 in his. His KG build was 2,669 s for 5,000 materials (Mac).

| Materials (80% train) | Our rows | `full`, 3 epochs | `no_geometry`, 3 epochs | His pipeline, 3 epochs |
|---|---|---|---|---|
| 250 (today) | 5.2k | 9 min | 5 min | 4 min |
| 1,000 | ~21k | ~35 min | ~20 min | ~15 min |
| 5,000 | ~104k | ~2.9 h | ~1.7 h | ~1.2 h |

Estimates scale linearly from the measured step times. GPU memory does not grow with material count,
since the batch size is fixed.

**Step 1: get more data. Two routes, both viable.**
- **A. Grow our own KG** with `battGPT/scripts/populate_kg_ont.py --cap N` (ontology v0.3.x, so our
  TBox, structure families and verbalizer work unchanged). Blocker: its candidate search found only
  248 candidates (13 ratio-pattern searches), so it must be widened, e.g. all Li/Na-containing
  materials and several polymorphs per formula. About 2 h ingestion for 5,000 at the measured rate.
- **B. Use the teammate's existing 5,000-material KG** (527 MB). Fast to obtain, but it's ontology
  v0.2.2 (no structure-family classes like ours) and our rdflib extractor would need about 8–12 GB RAM
  for 8.4M triples on a 15.7 GB machine. Use his streaming extractor instead (verified identical output
  above), extended to also collect space group, lattice and structure family.

Recommendation: **route A, first at 1,000 materials, then 5,000**, deliberately including polymorphs
(same formula, several crystals) so hasStructure can no longer be solved by composition matching.

**Step 2: the head-to-head on the same leak-free data — done (Step 21; result above), design kept for reference:**
1. our method: relation rows + in-batch negatives (Step 17);
2. his method: is-a rows `material ⊑ "has structure some <crystal>"` with explicit hard crystal negatives;
3. both combined.

Same split, same leak-free sentences, same checks: held-out hasStructure among test crystals and
among same-chemistry crystals, typing, TBox clustering (his metric), and the symmetry-label probe
from Step 19. Then repeat the winner at 1,000 materials.

**Step 3: things that break at scale, fix before 5,000:**
- **TBox oversampling explodes.** Balancing 1:1 means ×18 today, but ×~300 (ours) or ×4,000 per
  axiom (his) at 5,000 materials. Replace repetition with a loss weight on TBox rows, keeping his
  fresh-negatives idea if repetition stays.
- **Validation set:** `best_lambda` still comes from 2 TBox queries. Build a real validation split
  from held-out materials.
- **Numeric scaler:** refit on training materials only (`split.json`).
- **Electrode-role evaluation:** remove the `Battery role:` line from the material sentence (or
  evaluate without it), otherwise role accuracy is circular for both pipelines.
