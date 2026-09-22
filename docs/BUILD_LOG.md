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

## Step 4 — Tested the classifier, found and fixed a real bug, then found a real MP data quirk

**Unit tests first.** Before touching real data we ran 14 synthetic test cases (a `MaterialRecord`
with a made-up composition + space group, checked against the expected label) through the new
code. Three failed: the layered-oxide / rock-salt branch. The bug was arithmetic — for one alkali
atom + one metal atom + two oxygens, the ratio of oxygen to total cations is `2 / (1+1) = 1.0`, and
the code had `2.0` written instead. Fixed, re-ran, all 14 pass, plus a regression check that the
original 12 curated formulas still classify exactly as before (unaffected by the new code).

**Then live data, and a second, more interesting problem.** We picked two real battery-relevant
compounds the classifier had never seen (`LiCoPO4` — textbook olivine; `LiTi2O4` — textbook spinel)
and fetched them from the live Materials Project API. Both initially came back **unclassified**.

The reason isn't a classifier bug — it's about *which* Materials Project entry we used. A single
formula like `LiCoPO4` has **20 different entries** in Materials Project, one per DFT-computed
polymorph (different atomic arrangement, same formula). We were picking the one with the lowest
computed energy — which turned out to be a low-symmetry, purely computational structure
(`theoretical=True`, space group `Cc`), not the real, experimentally-known olivine structure.

The fix: Materials Project tags each entry with `theoretical` — `False` means the structure matches
a real measurement in an experimental database (ICSD), not just a DFT relaxation. Battery cathode
materials are very often a few meV/atom *above* MP's computed 0-Kelvin energy minimum (that's
normal — the true lowest-energy arrangement at absolute zero isn't always what forms and stays
stable at room temperature), so picking "lowest energy" alone can walk right past the real material
toward a synthetic DFT artifact. We now pick, in order: (1) on the hull (`is_stable=True`), else
(2) the lowest-energy entry that's experimentally verified (`theoretical=False`), else (3) lowest
energy overall, else (4) whatever comes back. With that fix, both compounds classify correctly:
`LiCoPO4` → olivine (Pnma, #62, the real ICSD-matched entry), `LiTi2O4` → spinel (Fd-3m, #227, its
one stable entry). This refines the original design doc's simpler rule (§10.1: "prefer `isStable`,
else lowest energy above hull, else first") — same idea, one more tier, empirically motivated.

## Step 5 — Found a much better way to search, then a duplication bug, then ran the population

**First attempt at search, and why we changed it.** The first version searched Materials Project
by *element set* — "give me every compound containing lithium, cobalt, and oxygen." That pulls
back the **entire Li-Co-O phase diagram**: LiCoO₂ (the one we want), but also Li₂CoO₃, LiCo₂O₄,
Co₃O₄, and every other stoichiometry that phase diagram happens to contain. Only about 18% of what
came back matched any of our 9 structure-family ratios — not because the classifier was wrong, but
because we were asking a question one level too broad.

**The fix:** Materials Project can also be searched by *shape* — a "1 atom : 1 atom : 2 atoms"
ratio pattern, independent of which actual elements fill those slots (it calls this an "anonymous
formula", e.g. `ABC2`). We found the exact pattern string for each of our 9 families by asking MP
what pattern our own curated examples already use (`LiCoO2` → `ABC2`, `LiMn2O4` → `AB2C4`,
`LiFePO4` → `ABCD4`, `Na3V2(PO4)3` → `A2B3C3D12`, ...) instead of guessing the notation. Combining
"this exact ratio" with "must contain lithium/sodium and oxygen" gets almost only compounds that
are *candidates* for that family in one search — cutting the total number of searches from ~58 to
13, and roughly doubling the useful fraction of hits. The garnet search stayed low-yield even so
(194 hits, ~0 real garnets) because a stuffed-garnet ratio without also requiring the specific
lanthanide+zirconium chemistry pulls in unrelated compounds that just happen to share the ratio —
the classifier correctly rejects those (no lanthanide, no garnet label), which is working as
intended, not a search failure.

**A real duplication bug, caught before it reached the KG.** Ten of the newly found formulas
turned out to already be in our hand-curated set — but the search sometimes picked a *different*
Materials Project entry for the same formula (e.g. curated `LiMn2O4` is `mp-22584`; the search's
own polymorph-selection independently landed on `mp-1272804`, a different structure with the same
formula). Left unfixed, the KG would have had two separate "LiMn2O4" materials under two different
URIs, silently duplicating a material we'd already verified by hand. Fixed by checking every new
candidate's formula against the curated set before adding it, and dropping it if already covered —
we always keep the hand-verified id, never a second one.

**The final population run.** `scripts/populate_kg_ont.py` (in the `battGPT` repo) ties this
together: run the searches, classify each candidate, always keep the curated 22, add every newly
*classified* material, then fill up to the cap with the rest (closest-to-stable first), and run
each through the same five enrichment steps `scripts/build_kg.py` already uses (Pymatgen structure
+ bonding, SMACT chemical sanity check, BattINFO role, structure family, electrochemistry) before
building, validating, and exporting the graph. Nothing about the existing pipeline changed — this
script only decides *which* material_ids to feed it.

**Results.** The run ingested all 250 selected materials successfully (0 failures), built
325,138 RDF triples (that includes the ~61,600-triple EMMO background closure every export
carries so the file is readable standalone — the material-specific content is ~263,500 triples),
and **passed the ontology's own validator with 0 errors and 0 warnings**. Output:
`output/battgpt_kg_ont/` in the `battGPT` repo (not committed there — same as the existing
`battgpt_kg`/`battgpt_kg_cathodes` outputs, it's regenerable from `scripts/populate_kg_ont.py`
and would add >100MB to that repo for no reason).

| | Before (curated only) | After this run |
|---|---|---|
| Total materials | 22 | **250** |
| With a `StructureFamily` label | 12 | **55** |
| With a `belongsToElectrode` battery role | ~10 | **105** |

Structure-family breakdown (55 total — LayeredOxide 16, NASICON 14, Spinel 9, Olivine 8,
Perovskite 3, Argyrodite 3, Garnet 1, LGPS-type 1): the ratio-pattern searches were most
productive for the oxide and phosphate families (plenty of real candidates share those ratios),
thinnest for garnet and LGPS-type (those really are rare, narrow chemistries — 1 example of each
beyond the curated set is an honest reflection of how few real compounds fit that recipe, not a
search failure). We verified one heuristically-classified material by hand,
[`mp-5670`](https://materialsproject.org/materials/mp-5670) (LiTi₂O₄), by reading its actual
exported RDF: correctly typed `chsub:Substance`, correctly given `NegativeElectrodeRole` (from the
*existing*, untouched battery-role heuristic — Ti+O anode rule), correctly given
`SpinelStructureIndividual` with the comment *"O:cation ratio 1.33 matches the spinel AB2O4
framework (~1.33) and space group #227 (Fd-3m)"* — exactly the reasoning we designed, visible
directly in the graph for anyone to check.

The other 195 materials carry no structure-family label — real chemistry, real crystal/site/bond
data, real DFT properties, just typed at the generic `chsub:Substance` level because no rule
matched their ratio+symmetry combination. That's the honest outcome, not a shortfall to fix: we
chose not to guess (Step 3's whole point), and 195 "just substances" is exactly what "don't guess"
looks like at this scale.

*(Next entry: building the actual OnT ABox extraction pipeline — extractor, verbalizer, numeric
export, row generation — now that we have a KG worth extracting from.)*
