# Build log — extending OnT to a populated knowledge graph

This is a running, plain-language log of every step in this project: what we did, why, and what it
means. Read top to bottom in order — each entry assumes you've read the ones before it. The
design questions and their answers are logged too, not just the code, because the "why" is the
part that doesn't show up in a diff.

Background reading: [`ONT_ABOX_EXTENSION.md`](../ONT_ABOX_EXTENSION.md) is the original design
spec. Some of it is now out of date (it was written against an older ontology and a since-deleted
9.6M-line KG) — this log calls out every place where we deviated from it and why.

---

## Step 0 — What we're actually building

**The goal in one sentence:** OnT is a model that learns to place *ontology class names* (like
"substance", "crystal") on a map (technically: a hyperbolic embedding space) so that
general/specific relationships between classes turn into distances on the map. We want it to also
place *real materials* (like "LiFePO4") on that same map, using their actual chemistry — not just
the class they belong to.

**Why that's not automatic:** OnT was built to read an ontology file (OWL/TBox) and walk its class
hierarchy. It never opens the *populated* knowledge graph (the ABox — real named things like
`mp-10178`, with real numbers attached). So by default, OnT has no idea materials exist.

**What we're adding:** a pipeline that turns each material (and its crystal, elements, etc.) into
an English sentence, adds that sentence to OnT's training data in the exact same format OnT
already uses for classes, and — separately — keeps the real numeric values (band gap, formation
energy, ...) as a plain list of floats. Text goes through OnT's model. Numbers get glued on after
training, because a language model reads "0.00" and "8.00" as similar-looking text, not as
different magnitudes (we verified this with a probe — see §8.1 of the spec doc).

---

## Step 1 — Checked what actually exists on this machine before touching anything

Before writing any code we checked:

- **The OnT and CrystaLLM source code** — not present locally. Cloned fresh from the real GitHub
  repos you pointed us to (`HuiYang1997/OnT`, `lantunes/CrystaLLM`) and read the actual training
  code, not just the paper/README, because the design doc turned out to describe the row format
  slightly wrong (see Step 2).
- **The populated KG** — the 9.6M-line, ~5,000-material graph the design doc describes no longer
  exists on this machine. What exists is `output/battgpt_kg_cathodes/battery_kg.ttl` in the parent
  `battGPT` repo: only **10 materials**, built against the *current* ontology (v0.3.2). We
  confirmed this is a different shape from what the doc assumes (different class names for
  crystals, different property-node layout, no bulk/shear modulus, battery-cell voltage/capacity
  living on a separate node) by loading it with `rdflib` and inspecting a real material's full
  neighbourhood by hand, not by trusting the doc's example block.
- **The Materials Project API key** in `.env` — tested it live (`mp_api.client.MPRester`), both a
  single-ID fetch and a composition search. Both work.

**Why check instead of assuming the doc is right:** the doc was accurate when written, but the
ontology and the KG have both moved on since. Building on stale assumptions would mean writing code
against data that no longer exists.

---

## Step 2 — Read the real OnT training code (the doc had the row format wrong)

OnT trains on rows read from JSONL files. There are two *kinds* of rows, and the doc's write-up
blurred them together:

1. **`train.jsonl`** — "is-a" rows: `{"child": "...", "parent": "...", "negative": ["...", ...]}`.
   Example: child = "LiFePO4 sentence", parent = "substance", negative = ["crystal", "element"].
   The *loader* (`ont/data/load.py`) only keeps the **first** negative in that list per row — if
   you write four negatives on one line, three are silently thrown away. So we generate **one row
   per negative** instead of stuffing a list into one row.

2. **`train_exist.jsonl`** — relation rows: `{"Concept": "...", "role": "...", "con": "..."}`.
   Example: Concept = "LiFePO4 sentence", role = "has structure", con = "its crystal's sentence".
   **These rows carry no negatives of their own.** The training loss
   (`ont/losses/logical_loss.py: LogicalConstraintLoss.exist_loss`) borrows its negatives from
   whatever `train.jsonl` batch happens to be running at the same time. This matters a lot for our
   plan to use "hard negatives" (e.g. a crystal from a similar but wrong material) for the
   `hasStructure` relation — that has to happen through the type-row negatives, not by attaching a
   `negative` field to an exist row, because there's no such field to attach it to.

We verified both of these by reading `ont/data/load.py` and `ont/losses/logical_loss.py` directly,
not by re-reading the doc more carefully — the doc's `train_exist.jsonl` example (§4, Phase 4) has
a `"negative"` key in it that the real loader doesn't recognise at all.

---

## Step 3 — Decided how to grow the KG, and why the "obvious" way was wrong

To train and evaluate anything, 10 materials isn't enough (no held-out set, no per-class
negatives). The natural fix is: pull more materials from the Materials Project API.

**The problem we found:** two of the five enrichment steps in `scripts/build_kg.py` —
`BattINFOMapper` (assigns cathode/anode/electrolyte/separator) and `StructureFamilyMapper`
(assigns which of the 9 crystal-structure-prototype classes a material belongs to, e.g. "layered
oxide", "spinel", "olivine") — decide those labels **by looking up the material's exact chemical
formula in a hand-written Python dictionary** covering only the ~22 materials your team has
individually verified. Any material pulled fresh from the API, with a formula not in that
dictionary, gets **no label** for either of those two things.

That matters because `StructureFamily` is exactly the label the ontology's own documentation
names as "the intended conditioning vocabulary for hyperbolic (OnT) embedding" — it's the good,
fine-grained "is-a" hierarchy we want OnT to learn. Without it, every new material would only be
typed as the generic top-level "substance" class, which (as flagged earlier) teaches the model
nothing, since *every* material is a substance.

**Decision (confirmed with you):** extend `StructureFamilyMapper` with a second, rule-based tier
that fires only when the hand-curated lookup misses. `BattINFOMapper` already *had* this two-tier
pattern (curated lookup, then a composition-based fallback already used in the existing 10-material
KG — e.g. NaMnO2's role comes from that fallback, not the curated table) — we followed the same
shape rather than inventing a new pattern.

### How the new structure-family rules work (plain language)

Each of the 9 structure "prototypes" is, physically, a specific *ratio of atoms* arranged with a
specific *symmetry* (space group — a number crystallographers assign to describe exactly how a
crystal's pattern repeats). Both pieces are already visible in this codebase's own curated
examples (e.g. the code already says LiFePO4 is "Pnma (#62) olivine-type phosphate"). We
generalised that pattern instead of guessing from element lists:

| Family | Defining ratio | Defining space group(s) | How sure are we? |
|---|---|---|---|
| Spinel | O : cation ≈ 4:3 (AB₂O₄) | #227 (Fd-3m) | High — ratio *and* symmetry both required |
| Olivine | metal : phosphorus ≈ 1:1 | #62 (Pnma) | High |
| Layered oxide | alkali:metal:O ≈ 1:1:2 | #166 or #12 | High |
| Rock salt | alkali:metal:O ≈ 1:1:2 | #225 (cubic) | High |
| Perovskite | O : cation ≈ 3:2 (ABO₃) | #221/#167/#140/#62 | Medium |
| NASICON | metal : phosphorus ≈ 2:3 | corroborating only | Medium |
| Garnet | contains alkali + lanthanide + (Zr/Nb/Ta) + O | corroborating only | Medium |
| LGPS-type | contains alkali + (Ge/Si/Sn) + P + S | corroborating only | Medium |
| Argyrodite | contains alkali + P + S + halogen | corroborating only | Medium |

The interesting case is **layered oxide vs. rock salt**: both have the *exact same* atom ratio
(1 alkali : 1 metal : 2 oxygen). The only thing that tells them apart is symmetry — layered oxide
has the metals sorted into orderly layers (rhombohedral or monoclinic symmetry), rock salt has
them jumbled together at random (cubic symmetry). This is precisely why the file's own original
docstring warns against classifying "from composition alone" — for this one pair, composition
genuinely isn't enough, and we only split them apart using the space group number pymatgen already
computes for every material (`SpacegroupAnalyzer`), never a guess.

**Every heuristic-tier label is marked as such.** The RDF `rdfs:comment` on each classified crystal
says either `"Curated benchmark: ..."` (a human individually checked it, matches Materials
Project) or `"Heuristic (space-group + stoichiometry match): ..."` (a rule matched, nobody looked
at it individually). If nothing matches, the material gets **no** structure-family label at all —
we never force a guess onto ambiguous data. Code:
[`pipeline/processing/structure_family_mapper.py`](../../pipeline/processing/structure_family_mapper.py)
(in the main `battGPT` repo, not this one — that's where the ontology-population pipeline lives).

*(Next entry: testing the classifier against real materials, then the actual population run.)*
