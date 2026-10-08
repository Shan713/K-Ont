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

## Step 7 — Built the extractor: KG → `entities.json`

[`abox/extract.py`](../abox/extract.py) reads `battery_kg.ttl` once with `rdflib` and writes a
plain, structured `entities.json`: one section per entity type (materials, crystals, unit cells,
sites, species, elements, space groups, crystal systems, battery cells), each keyed by the short
id the KG's own URIs already use (e.g. `material/mp-5670`), with every cross-reference (a
material's crystal, a site's species, a species' element, ...) resolved to another key in the same
file — so the next stage never has to touch RDF again, just follow plain dictionary lookups.

**Two deliberate deviations from `ONT_ABOX_EXTENSION.md`'s Phase 1**, both explained inline in the
script's own docstring, not just here:
- The spec put this script inside the vendored `OnT/` checkout. We didn't — `OnT/` is a
  gitignored, third-party clone (Step 1); our own code has no reason to live inside it, where it
  would never even be committed. It lives in `K-Ont/abox/` instead.
- The spec worried about streaming a 9.6M-line TTL "block-wise" because that's too big to load
  directly. Our real KG is 325K triples / 21MB — `rdflib.Graph().parse()` loads the whole thing in
  about 9 seconds, no streaming needed. (If the KG grows enough that this stops being true, *that*
  future point is when to revisit it — not now, speculatively.)

**Read the output by eye**, per the spec's own advice, rather than trusting the row counts alone —
walked one material (`mp-5670`, LiTi₂O₄) all the way down: material → its crystal → its unit cell →
one of its sites → that site's species → that species' element → the crystal's space group. Every
link resolved correctly, values match what we already hand-verified in the raw RDF back in Step 6.

**That check caught two real things:**

1. **A labelling bug, same shape as one already known.** Coordination-geometry individuals are
   named `"TetrahedralGeometryIndividual"` in the ontology itself — same `...Individual` suffix
   convention as the `StructureFamily`/`BatteryRole` individuals we already had to strip a suffix
   from. First pass forgot to do it here too. Fixed the same way, all three now read cleanly
   ("Tetrahedral", "SpinelStructure", "PositiveElectrode") instead of with the suffix attached.

2. **A real, pre-existing data-quality issue in Materials Project itself — not our bug, but worth
   knowing about before it reaches the numeric channel.** `mp-5670`'s `bulk_modulus` came back as
   **−4729.5 GPa**. Real bulk moduli are positive and, for even the stiffest known solids (diamond),
   don't exceed ~450 GPa. We checked the *raw* Materials Project API response directly (not our
   pipeline's transform of it) — it really does say `{'voigt': -9607, 'reuss': 148, 'vrh': -4729.5}`.
   This is a known MP phenomenon: for some materials the elastic-tensor fit is numerically
   ill-conditioned (often high-symmetry or soft-phonon materials), so the Voigt average comes out
   badly wrong while the Reuss average looks fine, and the VRH value we store is their mean —
   wrong in, wrong out. We scanned all 250 materials for implausible values across every property:
   band gap and energy-above-hull (the two properties battery relevance actually turns on) are
   **completely clean across all 250** — this only affects the elastic-property columns
   (`bulk_modulus`, `shear_modulus`, `universal_anisotropy`, `poisson_ratio`), and only on
   **6 of 250 materials** (`mp-5670`, `mp-25385`, `mp-48`, `mp-861667`, `mp-985591`, `mp-985592`).
   Noted for the numeric-scaling stage (not fixed here — the extractor's job is to report what's
   really in the KG, not to editorialize it): those four columns need an outlier
   guard (e.g. drop-if-out-of-physical-range) before computing a column's mean/std, or six bad
   values will drag the whole column's scale off for every other material too.

## Step 8 — Built the verbalizer: `entities.json` → V(a) sentences, two variants

[`abox/verbalize.py`](../abox/verbalize.py) turns each material, crystal, and element into a
labelled-field English sentence, in the two variants we agreed on:

- **`full`** — every field, including 3D geometry (lattice, volume, density, a compact per-element
  site count like `Sites: 28 (Li:4, P:4, O:16, Fe:4)` instead of listing all 28, mostly-repeated,
  symbols one by one).
- **`no_geometry`** — the same sentence with the geometry block removed entirely (not just the
  numeric side-channel, which doesn't exist yet — the sentence itself never mentions it).

**What stays in *both* variants, and why:** formula, chemical system, structure family, battery
role, metallic/stable flags, band gap, formation energy, energy above hull, Fermi energy, total
magnetization, bulk/shear modulus, Poisson ratio, open-circuit voltage, specific capacity, space
group, crystal system. Space group and crystal system are symmetry *labels*, not coordinates —
CrystaLLM's own prompt format already accepts a target space group directly
(`bin/make_prompt_file.py --spacegroup`), so stating it in the sentence isn't handing over anything
CrystaLLM couldn't already be told. Bond distances are left out of *both* variants outright — some
crystals carry ~80 bonds (Step 7), and the spec itself says to omit a bond list when it would be
long rather than force it in.

**The elastic-modulus outlier guard from Step 7 is applied here too, and we watched it work**:
`mp-5670`'s bulk/shear modulus (the −4729.5 GPa one) are silently absent from its sentence, exactly
as designed — no "Bulk modulus: -4729.5 GPa" line ever gets written, in either variant.

**Crystals borrow their property-tier facts from the owning material** (band gap, formation
energy only — a *smaller* set than the material's own sentence, just enough to tell crystals apart
for the `hasStructure` ranking task without turning V(crystal) into a full restatement of
V(material), which the original caveat about "the material-to-crystal link being too easy" already
flagged as a risk to control).

**Another doc-staleness catch**: `ONT_ABOX_EXTENSION.md`'s own element template (§7) includes
"Atomic number: 3" — checked the ontology directly and `hasAtomicNumber` exists as a predicate
(inherited from the EMMO chemical-substance import) but the population pipeline never actually
writes it onto any element individual in the real KG. We verbalize elements from the 6 fields that
really are there (group, period, electronegativity, valence electrons, atomic mass, covalent
radius) rather than inventing the seventh.

**Checked against OnT's real token budget, not assumed.** Downloaded the actual
`sentence-transformers/all-MiniLM-L12-v2` tokenizer and measured every one of the 1,075
verbalizations produced (250 materials × 2 variants, 250 crystals × 2 variants, 75 elements).
Zero go over 256 tokens; the longest is a material `full` sentence at 245/256 (a NASICON formula
with a long chemical system name eating extra tokens for the parentheses). One gotcha worth
flagging for whoever runs training later: the tokenizer's *own* bundled config truncates at 128 by
default — OnT only gets the full 256 because `ont/hit.py` explicitly re-wraps it with
`max_seq_length=256`. Construct the tokenizer some other way (e.g. testing it standalone) and it
will silently cut sentences off a third of the way through instead.

## Step 9 — Built the numeric module: the float side-channel `x`

[`abox/numeric.py`](../abox/numeric.py) builds the real-number vector `x` that gets glued onto
OnT's text encoding *after* training (`z = concat(v, x)` — not yet; that's the export step, once
OnT is actually trained). This step only produces `x` itself: raw values, z-scored values, and the
scaler (mean/std per column) that produced them, in the same `full` / `no_geometry` variants as the
verbalizer, sharing the same PROPERTY-vs-GEOMETRY split.

**The missing-data rule is measured, not asserted.** The spec (§8.2) says: a column missing on
over half the population gets *dropped* entirely (not zero-filled — a linear projection can't
tell "really zero" from "we don't know," so a mostly-missing column filled with 0 would look like
a real, confident low value to everything downstream); a column missing on under half keeps a
companion 0/1 "was this real" flag instead. Rather than hand-declare which columns fail that test
from what we already knew from Step 7's presence scan, the code *measures* each candidate column's
real missing fraction against the actual 250 materials and decides from that — so this keeps
working correctly if the underlying data's coverage ever changes, instead of silently going stale.

Measured and dropped: `bulk_modulus` (77% missing), `shear_modulus` (78%), `poisson_ratio` (78%),
`open_circuit_voltage` (96%), `specific_capacity` (96%). Measured and kept-with-a-mask:
`is_theoretical` (6% missing — mostly present, gets a companion `is_theoretical_known` column).
Everything else (band gap, formation energy, energy above hull, Fermi energy, magnetization,
metallic/stable/gap-direct flags, and — `full` variant only — the full lattice/volume/density/site
count) is 0% missing across all 250 and kept outright. Final column count: **19 for `full`, 10 for
`no_geometry`**.

**The two dropped-for-being-96%-missing columns are exactly the two values this project cares
about most** (open-circuit voltage, specific capacity — real electrode data, not computed DFT
properties). Applying the spec's own rule honestly means they can't sit in the main, dense `x`
vector next to columns that are 100% real for every material — but throwing them away entirely
felt like the wrong call given why we're building this at all. They're written to a **separate**
file, [`battery_cell_features.json`](../abox/numeric.py) — the same 10 materials that have real
electrode data (Step 1: only the original cathode test set does; the live Materials Project
ingestion path this population run used never fetches insertion-electrode data), openly sparse,
not pretending to be anything else. Whichever future step builds CrystaLLM's export can decide
whether/how to use it — this module's job was to report what's really there, not make that call.

**Verified by hand, not just by the log lines.** Picked `mp-5670` (the material whose bulk/shear
modulus we already know are garbage — Step 7) and printed its entire 19-value vector: those two
columns are simply absent, exactly as designed. Picked one of the 16 materials genuinely missing
`is_theoretical` (`mp-1143`) and confirmed its `is_theoretical` reads `0.0` *and*
`is_theoretical_known` reads `0.0` — the mask correctly says "don't trust this one," rather than
silently agreeing with the 234 materials that really are `0.0` (not theoretical). Recomputed one
column's mean and standard deviation independently (plain Python, not reusing the module's own
math) and it matched the scaler's stored values exactly. Checked for `NaN`/`inf` across all 500
material x variant vectors (250 materials x 2 variants): zero.

**One placeholder, flagged for later, not hidden:** the scaler is fit on all 250 materials, because
we don't have a train/test split yet — that needs more materials than we're confident classifying
today, and is a Phase 5 concern (evaluation), not this step's. The `Scaler` class takes an explicit
list of which material ids to fit on, so switching to a real train-only split later is a one-line
change, not a redesign — but it *is* currently fit on everything, and that's worth remembering
before trusting any number this scaler produces as if it came from a proper train/test split.

## Step 10 — Built the row generator: verbalizations → the actual JSONL rows OnT reads

[`abox/rows.py`](../abox/rows.py) turns `entities.json` + the verbalizations into
`train_abox_type_*.jsonl` (hierarchy rows) and `train_abox_exist_*.jsonl` (relation rows), in the
real schema from Step 2 — `{child, parent, negative}` with one row per negative, `{Concept, role,
con}` with no negative field at all.

**A modeling question we had to answer correctly before writing any code, not after**: is
`hasStructureFamily` (crystal → e.g. "olivine") a *type* fact or a *relation*? Checked
`battgpt.ttl` directly rather than going with the materials-science instinct to call a material
"an olivine" the way you'd call it "a substance." Both `hasStructureFamily` and
`belongsToElectrode` are declared `owl:ObjectProperty` — relations between two individuals (a
crystal and a canonical family individual; a material and a canonical role individual) — not
`rdf:type` assertions. So there is genuinely only **one** type/hierarchy fact per entity in this
whole KG (`material rdf:type Substance`, `crystal rdf:type CrystalStructure`, `element rdf:type
ChemicalElement`); structure family and battery role are both **exist rows**, same shape as
`hasStructure`, not type rows. Getting this backwards would have meant asking OnT's hierarchy loss
to learn something the ontology never actually asserts.

**Hard negatives for `hasStructure`, implemented the way the real code allows, not the way that
would be simplest to imagine.** `train_exist.jsonl` rows have no negative field (Step 2) — the
exist loss borrows its negatives from whichever `train.jsonl` *batch* happens to be running
concurrently at that training step, a loose statistical pairing we can't directly control from the
data side. The lever that *does* exist: raise how often a genuinely confusable crystal shows up as
a *negative value* somewhere in `train.jsonl`, so it has more chances to be borrowed. Every
material now gets one extra type row whose parent is (truthfully) `"substance"` and whose negative
is a hard-negative crystal — same chemical system if one exists elsewhere in the KG, else same
space group, else any other crystal — instead of an abstract label like `"crystal structure"`. No
new loss pathway invented, just a harder value in a slot that already existed. Checked on the
tracked example: `mp-5670` (LiTi₂O₄)'s hard negative came back as `mp-38280` (LiTiO₂) — same Li-O-Ti
chemical system, genuinely confusable, exactly the intended case.

**Read the actual output by eye and found a real bug**: the two canonical individuals whose
readable names carry their own internal casing (`"NASICON"`, `"LGPS-type"`) came out as `"Nasicon
structure family"` and `"Lgps-type structure family"` — a stray `.capitalize()` call was
lower-casing everything after the first letter. Fixed by dropping the standalone-capitalized-phrase
attempt entirely and using the same `"Label: value"` shape as every other sentence in the
verbalizer (`"Structure family: NASICON"`), which sidesteps the casing question rather than trying
to get it right through capitalization logic.

**A verification false alarm, also worth logging** (the point of reading output isn't to only
report the real bugs): a first check for self-referential hard negatives — a material's own
crystal accidentally picked as its "negative" — using naive string-splitting on the sentence text
flagged 7 materials as broken. Turned out to be a bug in the *check*, not the generator: several
formulas contain their own parentheses (`Na3V2(PO4)3`), which broke a `.split("(")` call looking
for the id in the wrong place. Re-checked directly against `entities.json` (material_project_id
fields, no string-parsing of rendered sentences) instead: **0/250 real self-references.**

**Row counts** (both variants, since `full` and `no_geometry` only differ in sentence content, not
row structure): **3,125 type rows**, **402 exist rows** (250 `hasStructure` + 55
`hasStructureFamily` + 97 `belongsToElectrode`) — comfortably in the doc's own "thousands of
training rows, not one row per triple" target (§9). Type rows outnumber exist rows roughly 8:1,
mostly "what kind of thing is this" rows that are individually easy to learn (a crystal is
obviously not a chemical element) — that skew is exactly the kind of number the design doc's Phase
5 count-logging step exists to examine before training, not something to silently correct here by
guessing a better ratio.

**Scope, stated plainly in the output filenames**: these are `train_abox_type_*` /
`train_abox_exist_*`, not `train.jsonl` — this is the ABox half only. Merging in TBox
class-hierarchy rows and TBox oversampling (design doc Phases 3–4) needs DeepOnto, the ontology's
OWL API, and a small extracted schema file, none of which exist in this repo yet.

## Step 11 — Built the tiny TBox schema OWL

[`tbox/build_schema.py`](../tbox/build_schema.py) generates `data/battgpt_tbox.owl`: the small,
self-contained class/property file OnT's own `prepare_ontology_data()` (in the vendored `OnT/`
checkout — this step runs OnT's real code, we don't reimplement DeepOnto verbalization ourselves)
reads via DeepOnto. 23 classes, 3 object properties, 19 `subClassOf` axioms, **zero
`owl:imports`**, 82 triples total.

**Why zero imports isn't just a size preference — checked what actually happens if you don't
follow it.** Read the real code (`OnT/ont/data/prepare.py`,
`ELNormalizedData.create_dataset`): `for ont_in_closure in ont.owl_onto.getImportsClosure():
axioms.extend(...)`. It walks *every* axiom in the full imports closure, not just the file it's
handed. One `owl:imports` pointing at even one of `battgpt.ttl`'s own imports (EMMO alone is tens
of thousands of axioms) means DeepOnto verbalizes and OnT tries to train on all of it. There's no
partial-import option here — the only way to keep this tiny is to import nothing, exactly the
design doc's own instruction (pitfall #1), now understood from the code, not just followed on
faith. `build_schema.py` asserts this at the end of every run (`assert not imports`) rather than
just hoping it stays true.

**Reused the real IRIs the ABox already types things with**, checked directly with rdflib against
`battgpt.ttl` (queried the actual `rdfs:subClassOf`/`rdfs:domain`/`rdfs:range`/`rdfs:label` triples
rather than recalling them from memory) — so a class in this tiny file and an `rdf:type` in the
KG provably refer to the same concept. Two classes needed a label added here that they don't carry
in `battgpt.ttl` itself (`ChemicalSubstance`/`ChemicalElement`'s real EMMO labels live in the EMMO
import we're deliberately not pulling in) — given `"substance"` and `"chemical element"`, matching
exactly what `verbalize.py` already calls them, so the TBox and ABox sides agree on the words.

**A second thing only the real code, not the design doc, reveals**: `create_dataset` only ever
processes `SubClassOf` and `EquivalentClasses` axioms. `battgpt.ttl`'s `hasStructure` /
`hasStructureFamily` / `belongsToElectrode` are plain `rdfs:domain`/`rdfs:range` declarations
(`ObjectPropertyDomain`/`ObjectPropertyRange` axioms) — a completely different axiom type the code
never looks at. Left as-is, none of our three relations would produce a single training row,
silently. Fixed by synthesizing an actual existential-restriction `SubClassOf` axiom for each one
(`ChemicalSubstance ⊑ ∃hasStructure.CrystalStructure`, etc.) — this is what turns a relation into
something DeepOnto's verbalizer and OnT's nf3 extraction can actually see.

**One deliberate simplification, caught by reading the extraction code closely enough to see the
constraint, not by trial and error**: `belongsToElectrode`'s *real* range in `battgpt.ttl` is
`owl:unionOf(NegativeElectrodeRole, PositiveElectrodeRole)` — a complex class expression.
`create_dataset`'s nf3 handling explicitly requires the filler to be a plain named class
(`if parent["class"]["type"] != "IRI": return  # Skip if filler is complex`) — so using the real
union would parse fine, verbalize fine, and then get **silently dropped** at the last step, the
exact kind of quiet data loss this whole project has been trying to catch before it happens rather
than after. Used `battgpt:BatteryRole` (the atomic ancestor both roles belong to) as the filler
instead — broader than the true range (it also nominally covers Electrolyte/Separator, which
`belongsToElectrode` can never really point at), but the only way to get any training signal out
of the relation at all under this real constraint. Written down here and in the script's own
docstring, not left implicit.

**Verified**: the file round-trips cleanly back through `rdflib.Graph().parse()` (82 triples both
ways), and every class/property/axiom count matches what was hand-derived from the real ontology
query beforehand — 23 = 2 EMMO anchors + 3 bare battgpt classes + 4 StructureFamily tiers + 9
leaves + 5 BatteryRole tiers; 19 `subClassOf` = 16 hierarchy + 3 synthesized existential.

## Step 12 — Got DeepOnto actually running, and read its real output

**The install, honestly.** `deeponto` needed real, heavy dependencies to import at all — not
optional extras, hard requirements: `torch`, `geoopt` (OnT's own package, for the Poincaré ball —
turns out importing *anything* under `ont.*` triggers `OnT/ont/__init__.py`'s eager import chain,
`ont.model` → `ont.hit` → `geoopt` + `sentence_transformers`, even though all we wanted was the
tiny `ont.data.prepare` submodule), `sentence_transformers`, and a spaCy English model
(`en_core_web_sm`) that `deeponto`'s own verbalizer loads internally. The spaCy model's
auto-download printed "✔ Download and installation successful" and then immediately failed to
load in the same run — checked `pip show en-core-web-sm` directly afterward and it genuinely
wasn't installed (a real failure, not an in-process caching illusion); fixed by installing the
exact wheel URL from the log directly. K-Ont's `.venv` went from ~65MB (rdflib + tokenizers) to
about 2GB. Every one of these is now in `requirements.txt`, one line each, so the exact
install sequence is reproducible rather than something only this session's history remembers.

**Then it worked, and the JVM genuinely starts** (`8g maximum memory allocated to JVM. JVM
started successfully.`) — Java 23 on this machine, found automatically, no `JAVA_HOME` needed.
[`tbox/prepare.py`](../tbox/prepare.py) calls `OnT/ont/data/prepare.py`'s own
`prepare_ontology_data()` directly, unmodified — every number it produced matched what we'd
hand-derived from the schema file before running anything: **16 nf1 (hierarchy) + 3 nf3
(existential) axioms, 23 concepts, 3 roles** — exactly the counts from Step 11.

**First real output was wrong, and reading it caught why.** `concept_names.json` came back as raw
`"ArgyroditeStructure"`, `"hasStructureFamily"` — un-split CamelCase, not English. Root cause:
`build_schema.py` had copied `battgpt.ttl`'s own real label convention (verified earlier: the
ontology genuinely does label its classes with their exact CamelCase name, e.g. `rdfs:label
"StructureFamily"@en`) — faithful to the source, but DeepOnto's verbalizer uses whatever label it's
given *verbatim*. OnT's own `prepare.py` ships a `camel_case_to_spaced()` helper for exactly this
situation, but only invokes it as an all-or-nothing fallback when the *entire* vocabulary is empty
— providing labels for even a couple of classes (which we needed to, for the two EMMO anchor
classes whose real labels live in an import we're not pulling in) would have disabled that
fallback for everything else too. Fixed by computing the spaced label explicitly ourselves for
every class/property, copying the small helper function in rather than importing it (importing
from `ont.*` would mean this schema-building script needs the same torch/geoopt install as
training, for no reason). Re-ran: `"argyrodite structure"`, `"LGPS type structure"`, `"NASICON
structure"` — acronyms correctly preserved, same pattern already caught once in `abox/rows.py`'s
`.capitalize()` bug, now fixed at the source before it could repeat.

**Reading the real `train.jsonl` — not just the counts — surfaced two things the design doc never
mentions, because they only show up once real data exists to look at:**

1. **Every single row (19/19) carries all 10 negatives** `prepare.py` samples, and the loader
   (Step 2) keeps only the first. Confirms, on genuine TBox output this time rather than just our
   own ABox rows, that this waste is real and comes from OnT's own reference pipeline, not
   something specific to our data.
2. **Sampling negatives doesn't exclude the true answer, and our TBox is small enough that this
   matters.** `prepare.py` samples 10 negatives from *all* 23 concepts, without removing the one
   that's actually correct. With only 23 concepts total, that's a real collision risk: **7 of 19
   rows** (37%) contain the true parent somewhere in their own negative list, and **3 of 19**
   (16%) have it at *position 0* specifically — the one negative the loader actually uses. Those 3
   rows currently train on `d(child, parent) < d(child, parent)`, a margin-only constant with no
   useful gradient. This is a small-ontology effect specifically — the collision odds shrink fast
   as the concept count grows, so a large TBox wouldn't feel this the way our 23-concept one does.

Neither of these is fixed here — both point at `OnT/ont/data/load.py` (already flagged as
possibly needing a touch, back in the original design doc's own file list) and are logged as a
concrete TODO before training, not silently worked around or silently left for someone else to
rediscover.

`data/battgpt_ont/` (the real generated output — `train.jsonl`, `train_exist.jsonl`,
`concept_names.json`, `role_names.json`, `val.json`, `role_inverse.json`, all a few KB) is
committed alongside the schema and the fix, as concrete evidence of what this step actually
produces, not just a description of it.

*(Pipeline status: extract → verbalize → numeric → rows (ABox) and build_schema → prepare (TBox)
are all built, run, and verified against real output. What's left before training: merging ABox +
TBox rows with TBox oversampling (design doc Phase 4b), deciding whether to patch the negative-
sampling/loader issues just found, and then an actual training run.)*

## Step 13 — Built the merge step, and reading real merged rows caught a worse bug than Step 12's

[`merge_datasets.py`](../merge_datasets.py) combines the ABox rows (Step 10) with the real TBox
rows (Step 12) into a complete, self-contained directory —
`data/battgpt_merged_{full,no_geometry}/` — with every file OnT's loader
(`load_local_dataset`) expects in one place: `train.jsonl`, `train_exist.jsonl`,
`train_conj.jsonl`, `val.json`, `concept_names.json`, `role_names.json`, `role_inverse.json`.

**Checked, not assumed, that the two sides actually speak the same vocabulary.** ABox type rows
use plain English like `"crystal structure"` as a parent/negative value — that string is only
meaningful if it's the *exact* string the real TBox (`concept_names.json`) verbalizes
`CrystalStructure` as. Ran the check before writing this script, not after: all 6 class labels and
3 role labels the ABox side uses are an exact match against the real, DeepOnto-generated
vocabulary — not by luck, but because both sides were written against the same "spaced, lowercase
English" convention from the start. `merge_datasets.py` re-runs this check on every call, so a
future change to either side that breaks the match fails loudly instead of silently producing two
unrelated embeddings for what should be one concept.

**Fixed the loader-truncation waste from Step 12** by fanning the real TBox rows out one-negative-
per-row here, the same treatment every ABox row already got in Step 10 — 19 raw rows (10 negatives
each) become up to 190 single-negative rows.

**Reading the fanned-out rows — not just re-deriving numbers from Step 12 — caught a second,
worse self-negative bug Step 12 missed entirely.** Step 12 only checked whether a row's sampled
negative equaled its true *parent*. A quick eyeball of three random fanned rows turned up
`{"child": "olivine structure", "parent": "polyanion structure family", "negative": ["olivine
structure"]}` — the negative equals the **child**, not the parent. That's a materially worse row
than the parent-collision case: `d(child, parent) < d(child, parent)` (the parent case) is merely
unsatisfiable-and-flat, contributing zero gradient once past its constant margin; `d(child, parent)
< d(child, child) = 0` (the child case) asks the model to make a distance smaller than zero — an
impossible target that keeps pushing on every step it's sampled, not just a wasted one. Checked how
common this actually is: **9 of 19 raw TBox rows** contain their own child somewhere in the sampled
negatives — more common than the parent case's 7/19 — because `prepare.py`'s negative sampling
draws from the full concept list without excluding *either* endpoint of the axiom, not just the
parent. `fan_out_and_filter_type_rows()` now drops a row if the negative matches **either**
endpoint; re-verified directly against the final merged file (not just the intermediate counts):
**zero** parent-self-negatives and **zero** child-self-negatives survive, in both variants.

**TBox oversampling**, computed from the real post-fix counts, per the design doc's ~1:1 target:
type rows need an **18x** repeat (174 fanned TBox rows → 3,132, against 3,125 ABox rows); exist
rows need **134x** (3 TBox facts → 402, against 402 ABox rows). Both logged with an explicit
warning that they exceed the design doc's own "do not pick 50-100x in isolation" guidance — worth
saying plainly: this reflects how small our current TBox is (23 concepts, 3 relations) relative to
how many times each gets asserted across 250 materials, not a decision that a bigger repeat count
is the right fix. Growing the schema (more object properties in particular — we only ever
synthesized 3) would bring these ratios down more honestly than repeating the same 3 facts 134
times each.

**Final merged counts**: `train.jsonl` — 6,257 rows (3,125 ABox + 3,132 oversampled TBox);
`train_exist.jsonl` — 804 rows (402 + 402). Both variants share identical TBox content and row
counts, differing only in the ABox sentences' geometry content, exactly as designed.

Neither self-negative fix touches `OnT/ont/data/prepare.py` or `load.py` — both are transformations
this script applies on the way from the TBox's raw output into the merged file, so the vendored
OnT code stays completely unmodified while the actual training data it will read is clean.

*(Pipeline status: every stage the original spec called for — extract, verbalize, numeric, ABox
rows, TBox schema + prep, merge — is now built, run, and verified against real output.
`data/battgpt_merged_{full,no_geometry}/` is a complete directory `pipeline.fit()` could point at.
What's left: an actual training run, Phase 5/5b evaluation, export, and CrystaLLM wiring — none
started yet.)*

## Step 14 — Tried an actual training run, found a real device bug, and measured (not guessed) how
slow CPU really is

**First real attempt found a genuine bug, not a training problem.** Pointed `ont/pipeline.py`'s
`fit()` at the merged data and it crashed on the very first step: `RuntimeError: Passed CPU tensor
to MPS op`. Root cause, found by reading the actual code rather than guessing: `fit()` picks the
model's device with `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — on this Mac
(no CUDA), that's `"cpu"`. But a few dozen lines later it also sets `use_cpu=False` on
`SentenceTransformerTrainingArguments`, and *that* flag makes HuggingFace's own Trainer
independently re-detect the device — which DOES check Apple's `mps` backend, and finds it. Two
different parts of the same function silently disagreed about which device to use, and the crash
was that disagreement meeting in the middle. Fixed both to derive from one shared decision (`cuda`
> `mps` > `cpu`), and added an explicit `device=` override to `fit()` for future debugging —
exactly the "TOUCH `OnT/ont/pipeline.py` # device" the original spec's own file list anticipated.

**Then MPS itself turned out to be broken for this codebase.** With the device disagreement fixed,
training correctly started on `mps` — and immediately hit `RuntimeError: MPS backend out of memory
(MPS allocated: 20.09 GiB...)` on the very first step, for a 33M-parameter model with batch size
64. That's a wildly disproportionate allocation, most likely from how the hyperbolic
(Poincaré-ball, `geoopt`) operations or the exist/conj loss's extra forward passes interact with
Apple's MPS memory allocator — a real rough edge in this reference implementation's MPS support,
not something worth chasing down today. Fell back to CPU, which is what the rest of this step
actually measured.

**`OnT/` is gitignored, so a code fix inside it needs a different home to survive a re-clone.**
Committed the fix locally inside `OnT/`'s own git history (it's a real clone, with its own `.git`,
separate from `K-Ont`'s), then exported that commit as a plain diff into
[`patches/ont_pipeline_device_fix.patch`](../patches/ont_pipeline_device_fix.patch) — a file `K-Ont`
*does* track. [`patches/README.md`](../patches/README.md) says to apply it right after cloning
`OnT/` fresh, before running anything.

**Measured real CPU training time — and the first estimate given for this (in chat, not logged
here as fact) was wrong, off by roughly 50–100x.** Reasoning from dataset size and model size alone
suggested sub-second steps; a real run showed **~85–115 seconds per training step** (four real
steps timed: 114.7s, 88.8s, 91.0s, 85.1s). With 98 steps for one epoch on this dataset, that's
**roughly 2.5 hours for a single epoch on CPU alone** — not the "few minutes" first guessed. The
gap is because a "step" here isn't one forward pass: the hierarchy loss, the existential-role loss,
and the conjunction loss each run their own encode passes through MiniLM-L12, plus the hyperbolic
manifold math on top, all on unaccelerated CPU matrix multiplication. Worth stating plainly: this
correction is exactly why this log tries to measure rather than estimate — the estimate given
first, before running anything, was confidently wrong.

**Decision: move training to a machine with a working GPU** (an RTX 4060 laptop) rather than
either waiting out multi-hour CPU epochs or debugging MPS's memory blowup further right now.
[`docs/GPU_SETUP.md`](GPU_SETUP.md) is the checklist for that move — prerequisites, cloning +
patching `OnT/`, installing the CUDA build of `torch` before the rest of `requirements.txt` (so pip
doesn't silently grab a CPU-only build), and — importantly — **copying the already-generated
`K-Ont/data/` directory over rather than regenerating it** (13MB total, small enough to just
transfer directly; none of extract/verbalize/numeric/rows/TBox-prep/merge depend on which machine
runs the *training*, so there's no reason to redo any of it).

*(Next entry: once on the GPU machine — verify CUDA is actually used, then a real training run,
Phase 5/5b evaluation, export, CrystaLLM wiring.)*

## Step 15 — Moved to the RTX 4060 laptop: ~35x faster than CPU, the "MPS bug" was really activation memory, and the first real training runs (2026-09-22/23)

Working through [`GPU_SETUP.md`](GPU_SETUP.md) in order, checking each step before the next.

**GPU and driver (setup step 1) — fine.** `nvidia-smi` shows the `RTX 4060 Laptop GPU` with 8 GB,
driver 610.62, CUDA 13.3. That's newer than the `cu121` example the checklist gives, so the torch
wheel has to come from whatever pytorch.org currently lists for this driver, not from that example.

**Prerequisites (setup step 2) — one mismatch.** Java (Temurin 25) and Git are fine. The only Python
on this machine is **3.13**, not the 3.11 the checklist asks for, and there's no `py` launcher to
pick another version. We didn't build the venv on 3.13 and hope `deeponto`/JPype/spaCy wheels
exist for it — that's the same kind of guess the checklist was written to avoid. Installed 3.11.9
per-user with `winget install --id Python.Python.3.11` (the official python.org installer), checked
it actually runs (`3.11.9`, 64-bit), and built the venv from that exact interpreter's path rather than
whatever `python` on PATH happens to resolve to (still 3.13).

**The data wasn't actually here yet, and checking the files caught it — the folder's existence
didn't.** `data/` existed, with `battgpt_merged_full/` and `battgpt_merged_no_geometry/` inside it,
so at a glance the transfer looked done. It wasn't: each merged directory held only
`merge_summary.json`, and `git ls-files data` showed that **every** file in `data/` was one git
already tracks. The whole folder was 111 KB, not the ~13 MB the checklist describes. So the folder
was only what `git clone` brings, and the gitignored training files (`train.jsonl`,
`train_exist.jsonl`, ...) had never been copied from the Mac. Deliberately **not** regenerated here
(per setup step 5 — none of that pipeline depends on the training machine); waited for the transfer.

**When the data did arrive (a zip), checked it against what was already here before copying
anything in.** 13.4 MB, 49 entries — matching the ~13 MB the checklist expects. Some files in the
zip were also already tracked by git (e.g. `merge_summary.json`), with *different* byte sizes (377
vs 391). Extracted to a scratch folder and compared every tracked file first: all 14 were identical
in content — the only difference was Windows line endings git adds on checkout here
(`core.autocrlf=true`). So only the 28 files git *doesn't* track were copied in (skipping the Mac's
`.DS_Store`), leaving tracked files alone and `git status` clean. Then re-verified the training
files themselves, not just the summary: `train.jsonl` 6,257 rows and `train_exist.jsonl` 804 in
both variants, **exactly** Step 13's counts, and a fresh re-scan of every type row found zero self-
negatives (Step 13's fix survived the trip).

**Cloned `OnT/` and applied the device patch (setup step 3) — applied cleanly.** The fresh clone is
at upstream `82ef384` ("Release ontology-transformer 0.1.7"). Before trusting the clean `git apply`,
checked that the patch was made against the same file, not just one close enough to apply: its
`index 1e24146..` line matches the blob hash of the freshly cloned `ont/pipeline.py`
(`git rev-parse HEAD:ont/pipeline.py` → `1e24146`), so the patch's base is identical to upstream
today. Read the applied diff by eye (the `device=` argument, the cuda > mps > cpu pick, and
`use_cpu` now following that same pick). Also checked that `Optional` — which the new argument's
type hint needs — is already imported (`pipeline.py` line 7), and that the patched file still
parses. `K-Ont`'s own `git status` stays clean, as it should, because `OnT/` is gitignored.

**CUDA torch, then the gate (setup step 4) — passed, and checked it stayed passed.** Checked which
CUDA builds of torch actually exist for Python 3.11 on Windows rather than copying the checklist's
`cu121` example (which tops out at torch 2.5.1): `cu130` and `cu132` both carry torch 2.14.0, and
this driver supports up to CUDA 13.3. Installed `torch 2.14.0+cu130`. `torch.cuda.is_available()`
→ `True`, device name `NVIDIA GeForce RTX 4060 Laptop GPU`, and — so "available" wasn't taken on
faith — a real 4096×4096 matmul ran on `cuda:0`. Then `pip install -r requirements.txt`, and
re-checked afterward: still `2.14.0+cu130`, still `True` (the exact failure the checklist's install
order exists to prevent — pip swapping in a CPU build — didn't happen). The spaCy model installed
from its wheel URL and actually loads (`spacy.load` check, not the log line). One thing noted for
later: this pulled **sentence-transformers 6.1.0 / transformers 5.17.0**, newer than OnT's own last
compatibility fix ("Sentence Transformers 5+"). That mattered twice below.

### First GPU run: out of memory — and the real reason the Mac's MPS run failed too

Before running anything, read `fit()` again to see how it treats `output_dir/data`: if `train.jsonl`,
`concept_names.json`, `role_names.json` and `val.json` all exist it reuses them, otherwise it runs
DeepOnto prep on the TBox alone — which would "train successfully" on 19 TBox axioms instead of our
merged data. The checklist's inline snippet also never configures logging, so `fit()`'s own
`"on device: ..."` line is silently dropped. Wrote [`train_ont.py`](../train_ont.py) instead: it
copies the merged files in (refusing to mix with a different dataset already there), turns logging
on, calls OnT's own `fit()` unmodified otherwise, and wraps the trainer's `training_step` with a
wall-clock timer taken after `cuda.synchronize()` (GPU work is asynchronous — without the sync you'd
time the kernel *launch*, not the work), writing every step's time and peak GPU memory to
`step_times.json`.

The log confirmed `Reusing existing training data` and `on device: cuda` — then step 1 died:
`CUDA out of memory ... 14.38 GiB is allocated by PyTorch` on an 8 GiB card. A 33M-parameter model
at batch 64. That's the same shape as the Mac's MPS failure (20 GiB on step 1, Step 14), on a
completely different backend — so Step 14's guess that it was an MPS rough edge, or the hyperbolic
math, was wrong. Something in this codebase asks for that much memory on *any* device.

**Found it by measuring, not guessing.** The traceback put the crash in the very first encode pass of
the hierarchy loss (`hit_loss.py:46`) — before the exist/conj losses even run, so not "extra forward
passes" either. A probe with the model built exactly as `fit()` builds it: max sequence length 256
(correct), SDPA attention (fine), and a real batch pads to **64 × 222 tokens** — our ABox sentences
are long (median material sentence ~200 tokens; OnT's usual class names are ~5). One 64-sentence
forward+backward pass alone peaks at **4.36 GiB** of activations in fp32. And one training step keeps
several of these alive at once until backward: `LogicalConstraintLoss.forward` encodes the negatives,
then the exist rows' Concept and con, then `hit_loss` encodes child, parent and *the negatives
again*. Several 4+ GiB passes → the ~14 GiB we saw. Plain transformer activation memory on long
sentences — nothing exotic.

**Fix: gradient checkpointing** (recompute activations during backward instead of storing them).
Chosen over the two obvious alternatives on purpose: a smaller batch changes the training itself
(the exist loss borrows its negatives from the concurrent batch — Step 2), and bf16 is a real risk
next to Poincaré-ball math near the boundary. Measured on one real batch before patching anything:

| | peak memory | fwd+bwd time |
|---|---|---|
| no checkpointing | 4.50 GiB | 0.441 s |
| gradient checkpointing | **0.89 GiB** | 0.560 s (+27%) |

— with the embeddings and the gradient norm (22.7173…) **identical to every printed digit**, so the
math is unchanged. Added as an opt-in `gradient_checkpointing=` argument on `fit()`. First attempt
used the Trainer's own `gradient_checkpointing=True` flag and crashed before step 1 — transformers
5.17's Trainer passes an `every_n_layers` argument sentence-transformers 6.1's wrapper doesn't accept.
Enabled it directly on the underlying HF encoder instead (exactly the call the probe measured), and
had `train_ont.py` read the state back off the live model at step 1 rather than trust the flag:
`HF gradient checkpointing active: True`, `model device cuda:0`.

**Then the run got through all 98 steps and crashed in the end-of-epoch eval**:
`module 'numpy' has no attribute 'trapz'` — removed in NumPy 2.4 (we have 2.4.6), renamed
`np.trapezoid` with the same arguments. One call site (`ont/evaluation/ranking.py:40`); fixed, and
checked `np.trapezoid` gives the right area on a known curve (0.75) before trusting it. Worth noting
only because it cost a full epoch: a crash at the very *end* of a run is the expensive kind, and the
rerun was the only way to test the eval → save path end to end.

### The measured number

Smoke run (`data/runs/smoke_full_minilm_b64_e1/`, checklist settings: `full` variant, 1 epoch,
batch 64, plain `all-MiniLM-L12-v2`): **median 2.54 s/step** (first 3.20, min 1.85), 98 steps in
4 min 8 s of training, **peak GPU memory 2.04 GiB**, exit 0, model saved. Against the Mac CPU's
measured 85–115 s/step, that's **~35–45x faster** — one epoch in ~4 minutes instead of ~2.5 hours.
Loss 14.56 at step 1 → 1.94 at step 90. Re-running with the same seed reproduced the loss exactly
(14.5607 at step 1, 1.9437 at step 90), so the timing and loss aren't one-off luck. Step times crept
up from ~2.2 s to ~2.5 s over the epoch, consistent with a laptop GPU warming up; not investigated
further.

### The real training runs (design spec Phase 5 settings)

The checklist's smoke settings aren't the plan. `ONT_ABOX_EXTENSION.md` Phase 5 says: base
`Hui97/OnT-MiniLM-L12-galen` if available, batch 16–32, 1–3 epochs, `existence_loss_kind=hit`
(already `fit()`'s default). Checked the galen model rather than assuming: it exists on the HF hub,
uses mean pooling (matching what `HierarchyTransformer.from_pretrained` builds), and its weights load
with **zero** missing/unexpected keys — worth checking directly, since that loader deliberately
hides the load report. Its embeddings differ clearly from plain MiniLM's, so the pretrained OnT
weights are really in play. (`fit()` starts the *role* model fresh either way; only the encoder is
pretrained.)

**One more thing that only shows up once you read the eval data**: `val.json` holds exactly **2
queries** (one nf1, one nf3, all TBox — `prepare.py` samples 10% of 19 axioms). With more than one
epoch, `fit()` reloads whichever epoch scored the best val MRR at the end — on 2 queries, that's a
coin flip deciding which model you get. Added `select_best_epoch=` to `fit()` (upstream default
unchanged) and ran with it off: the final model is the last epoch, deterministically. `best_lambda`
(the centripetal weight used at inference) still comes from those same 2 queries and came out 0.0 —
treat it as unvalidated.

Two runs, one per variant — **batch 32, 3 epochs, galen base, last epoch kept** (`train_ont.py`):

| | `full` | `no_geometry` |
|---|---|---|
| steps | 588 | 588 |
| median s/step | **1.257** | **0.693** |
| training time | 12 min 0 s | 6 min 55 s |
| peak GPU memory | 1.29 GiB | 0.99 GiB |
| loss, every 100 steps | 2.37 → 0.77 → 0.58 → 0.53 → 0.52 | 2.34 → 0.78 → 0.58 → 0.53 → 0.52 |

Galen starts at loss 7.99 (step 1) against plain MiniLM's 14.56 — the pretrained OnT start really
does help. `no_geometry` steps are ~45% faster simply because its sentences are shorter.

### Did training do anything sensible? (Phase 5 checks — training-set fit, NOT Phase 5b)

[`abox/phase5_checks.py`](../abox/phase5_checks.py) runs the spec's Phase 5 sanity checks on a
trained model *and* on the galen model it started from, using OnT's own `score_hierarchy`. **Every
number below is on materials that were also in training** — there's no held-out split yet (Step 9's
note) — so this is "did the training move things the right way", not generalization. Output:
`data/runs/*/phase5_checks.json`. `full` variant, before → after (`no_geometry` is within a couple of
points on everything):

- **Class names stayed distinct** (spec: `encode("substance") ≠ encode("crystal")`). Substance ↔
  crystal structure distance 8.4 → 15.2; the closest pair of any two of the 23 classes 4.9 → 8.6.
- **Materials now type as `substance`: 0% → 100%** (nearest class; before training, 218/250 sat
  nearest `crystal structure`). Elements 97% → 100%.
- **Crystals went *down*: 100% → 78.4%** — and reading *which* ones explained it. All 54 misses are
  crystals with a structure family, and **all 54 land on their own family's class** (a spinel crystal
  nearest `spinel structure`, etc.), zero on a wrong family. That's the `hasStructureFamily` exist
  rows pulling crystals toward their family — informative placement, but strictly the ontology says
  a crystal is `rdf:type CrystalStructure` and only *related to* a family individual (Step 10), so by
  the ontology's own letter these are typing misses. Logged as-is; not scored as a pass or a failure.
- **No crowding collapse** (spec pitfall #3). Materials sit at ~0.41 of the ball radius (not piled
  at the boundary); same-family pairs tightened (median distance 7.6 → 5.0), which is what the
  hierarchy loss should do, but no two materials collapsed — minimum nearest-neighbour distance 0.92.
- **Confusable pairs moved *apart***, the spec's `V(LiMgP) ≠ V(LiZnP)` check. Neither of those two
  formulas is in this KG, so the same-chemical-system pairs stood in (86 pairs; the Step 10 example
  LiTi₂O₄ / LiTiO₂: 4.4 → 8.6). Median 5.5 → 6.5, none collapsed.

**And a real problem with the `hasStructure` data, found by running the check on the *untrained*
model first.** Before any training at all, plain distance (no role, no fine-tuning) already matches
each material to its own crystal among all 250 with **H@1 = 0.988** (random: MRR 0.024). Read the
sentences to see why: **all 250 crystal sentences start with the same `LiTi2O4 (mp-5670)` identifier
as their material's sentence**, and repeat the band gap and formation energy too (Step 8's design —
"just enough to tell crystals apart"). So `hasStructure` can be solved by string overlap before
learning anything — exactly the "material-to-crystal link being too easy" risk Step 8 flagged,
now measured. After training it's 0.44 MRR via the learned role (0.34 without it), *below* the
untrained string-match — the typing losses pull the material and crystal clouds apart toward
`substance` and `crystal structure`. Neither number means anything about relation learning while the
leak is there. **This needs fixing before Phase 5b's hasStructure eval is worth running** (drop the
id/formula from V(crystal), or reword it), but that's a verbalizer change plus regenerating rows — a
design decision, deliberately not made here.

### What changed in the repo (not yet committed), and where the patch lives now

- [`patches/ont_gpu_training_fixes.patch`](../patches/ont_gpu_training_fixes.patch) — gradient
  checkpointing, `select_best_epoch`, `np.trapezoid`. It's a diff *on top of* the Step 14 device
  patch (committed the device fix inside `OnT/`'s own history first, as on the Mac, then diffed).
  Verified by applying both patches, in order, to a fresh worktree at upstream `82ef384`: the result
  matches the working files exactly. `patches/README.md` and `GPU_SETUP.md` updated to apply both.
- `train_ont.py`, `abox/phase5_checks.py`; `GPU_SETUP.md` step 6 now uses `train_ont.py` and records
  the measured number.
- `data/runs/<run>/step_times.json` + `phase5_checks.json` only — the models themselves (~0.9 GB per
  run, in `data/runs/*/final/`) are gitignored like every other generated artifact here.

*(Pipeline status: training works on the GPU, ~35x faster than CPU, and both variants are trained
and saved. Before Phase 5b / export: (1) a held-out material split — every check above is training
fit; (2) fix the `hasStructure` identifier leak in V(crystal); (3) the spec's type-loss ablation
(hierarchy vs. a plain class margin), not started; (4) a real `val.json` — 2 TBox queries can't
choose `best_lambda` or an epoch.)*

## Step 16 — Fixed the hasStructure leak in V(crystal) — and the honest number underneath it is bad (2026-09-23)

**Measured which lines were the leak before removing anything.** Step 15 found every crystal sentence
opening with its material's `LiTi2O4 (mp-5670)`. Reading `abox/verbalize.py` turned up four candidate
channels, not one: (1) that identifier line; (2) band gap + formation energy copied verbatim from the
material into the crystal; (3) the material sentence's own `Structure: Crystal structure of LiTi2O4`
line, pointing at its crystal; (4) in `full`, the material and its crystal carry the *same* `Lattice:` /
`Volume:` / `Sites:` lines. Removed each in turn and re-measured the untrained galen model's plain-
distance ranking of each material's crystal among all 250 (random MRR 0.024):

| change (cumulative) | `full` MRR / H@1 | `no_geometry` MRR / H@1 |
|---|---|---|
| current | 0.994 / 0.988 | 0.968 / 0.948 |
| − crystal identifier line | **0.221 / 0.120** | **0.059 / 0.020** |
| − copied band gap / formation energy | 0.270 / 0.160 | 0.100 / 0.064 |
| − material's `Structure:` pointer | 0.338 / 0.208 | 0.136 / 0.088 |
| − geometry from the material (full only) | 0.293 / 0.168 | — |

The identifier line *is* the leak — removing it alone takes H@1 from 99% to 12% / 2%. The other lines
barely move the untrained model (removing them even nudges MRR *up*, since less text dilutes the
shared structural lines). Decisions, made on what each line *is*, not only on what it measured:
- **Removed the identifier line** — pure bookkeeping, not chemistry.
- **Removed the copied band gap / formation energy** from V(crystal). They're the material's
  properties, not the crystal's (crystals carry none of their own, Step 7), and a 3-decimal formation
  energy is a near-unique fingerprint: the untrained model doesn't exploit it, but a fine-tuned one is
  exactly what could learn to string-match it.
- **Left the material sentence alone** (byte-identical to before, checked). Its `Structure:` line names
  only its own formula, which no longer appears on the crystal side, so it's no longer a leak; and
  keeping V(material) unchanged keeps the exported embeddings meaning the same thing.
- **Left `full`'s shared lattice line alone.** Materials carrying geometry is what the `full` variant
  *is*; it moved the untrained number only 0.34 → 0.29. `no_geometry`, which shares no such line, is
  the control for whether training learns to exploit it.

**Consequence, accepted on purpose:** a `no_geometry` crystal is now only its family, space group and
crystal system — **67 distinct sentences among 250 crystals**. That's the honest content of a crystal
with no geometry. The ceiling for ranking by symmetry alone is MRR **0.447** (computed: perfect on
symmetry, random within same-symmetry groups), not ~1.0.

**Which forced a second fix, in the hard-negative picker.** Step 10's picker falls back to "same space
group" — with symmetry-only sentences, that can pick a crystal whose sentence is *identical* to the
material's own crystal's, putting exactly the text the material is pulled toward in as its negative:
the same impossible target Step 13 removed from the TBox rows. `_find_hard_negative_crystal` now skips
any candidate whose sentence equals the material's own crystal's, at every tier. Checked on the
regenerated rows: **0** such rows in either variant.

**Regenerated** verbalize → rows → merge (entities.json unchanged — no KG or DeepOnto re-run; the
previous outputs are backed up in `data/backup_pre_leakfix_20260923/`, gitignored). Verified against
the backup: material sentences byte-identical; row counts identical (6,257 type / 804 exist per
variant); **0/250** crystal sentences naming their material's mp-id or copying band gap / formation
energy. A leak scan flagged 2 `full` crystals — read them: elemental Si and C, whose "formula" is just
the element symbol in their own `Sites: 2 (Si:2)` line, the same composition line every `full` crystal
carries. Structural content, not the identifier; not changed.

**Made the checks fair for duplicate sentences first.** `phase5_checks.py` counted only strictly-
better scores, so an exact tie (two identical `no_geometry` crystals → identical embeddings) always
went the true crystal's way. Ties now count as a random tie-break. Also added the references the spec
asks to beat — the symmetry-only ceiling, and a hard-negative ranking (true crystal vs only the
crystals of other materials in the same chemical system: 113 materials have one, and random ranking
inside those small groups already scores MRR **0.679**) — plus an `--abox-dir` option so older runs are
checked against the sentences they actually trained on, not today's. The tie fix reproduced the older
runs' numbers exactly (they had no ties).

**Retrained both variants** (Step 15's settings: galen, batch 32, 3 epochs, last epoch; `full` 10.2
min at 1.06 s/step, `no_geometry` 6.2 min at 0.62 s/step). Training-set fit, all materials:

| hasStructure MRR | untrained, plain | trained, via role | trained, hard-neg (random 0.679) |
|---|---|---|---|
| `full`, **leaky** (Step 15) | 0.994 | 0.435 | 0.860 |
| `full`, **leak fixed** | 0.270 | **0.062** | 0.787 |
| `no_geometry`, **leaky** | 0.968 | 0.357 | 0.870 |
| `no_geometry`, **leak fixed** | 0.052 | **0.171** | 0.940 |

Everything else held: materials typed `substance` 100%, crystals 78% (the same own-family pattern as
Step 15), no collapse, confusable pairs apart, loss curve essentially identical (2.37 → 0.52).

**The leak had been hiding a real failure: training doesn't learn hasStructure.** In `full`, the trained
role scores MRR 0.062 — barely above random, and *below* what the untrained model gets from plain
distance (0.270). Found why in the code, not by guessing: `LogicalConstraintLoss.exist_loss` takes its
negatives by randomly permuting `neg_samples` — the encoded `negative` column of the *concurrent
type-row batch* (Step 2). Measured what those are on our data: **96.0%** are short class labels
(`space group` ×773, `crystal system` ×701, `battery role` ×629, ...); only the 250 hard-negative rows
(4.0%) carry a crystal sentence. So a hasStructure row is almost never asked to prefer its own crystal
over *another crystal* — the one thing the relation needs — and while the leak was in, it never had to
be, because the text did it for free. (`no_geometry`'s 0.171 beats `full`'s but is still well under its
0.447 symmetry ceiling.)

## Step 17 — A held-out split, and in-batch negatives: hasStructure is learned, and it generalizes (2026-09-23)

Both follow from Step 16: there was no way to tell generalization from memorization, and the exist
loss had no crystal-vs-crystal contrast.

**Held-out split** — [`abox/split.py`](../abox/split.py) → [`data/split.json`](../data/split.json)
(committed: it defines what "held out" means). 20% of materials, fixed seed, stratified by structure
family (every family with ≥2 members contributes; garnet and LGPS-type have one member each and stay in
train — holding out the only example would test a family the ABox never showed the model, a different
question): **199 train / 51 held-out**. A material and its crystal always land on the same side.
`abox/rows.py --split` filters `entities` down to the train side *before* any row is built, so one
filter covers every place a held-out entity could appear — as a child, an exist Concept/con, and as
another material's hard-negative crystal (the picker only ever looks inside `entities`). Checked on the
output rather than trusted: **0** held-out material sentences and **0** held-out-only crystal
sentences in any training row. (In `no_geometry`, 25 held-out crystal *texts* also belong to some train
crystal — symmetry-only sentences, Step 16's consequence, not an identity leak.) Without `--split`,
`rows.py` output is byte-identical to before, and so is `merge_datasets.py`'s default output (both
gained optional directory arguments; checked with `cmp`). Split data: 5,174 type / 631 exist rows,
TBox oversampling recomputed from the smaller ABox counts (15x / 105x).

**In-batch negatives** — [`patches/ont_exist_inbatch_negatives.patch`](../patches/ont_exist_inbatch_negatives.patch),
opt-in `exist_in_batch_negatives=` on `fit()` (default off = upstream). Adds one term to the exist loss:
each exist row *i* whose filler is an instance sentence is contrasted against another row *j* of the
**same batch and same role** — `material_i` should sit nearer `∃hasStructure.crystal_i` than
`∃hasStructure.crystal_j`. The rules for *j*, each there for a reason:
- same role — so ∃r.D_j and ∃r.D_i share the rotation and only the filler differs;
- never a TBox class-name filler, on either side — `material ⊑ ∃hasStructure.crystal structure` is
  *true*, so it must never be a negative; TBox rows are left entirely to the upstream loss;
- never identical filler text — two identical `no_geometry` crystals, or two "Battery role: cathode"
  rows, would be the impossible target again;
- only the clustering (contrastive) part of the hierarchy loss, so the centripetal part isn't counted
  twice.

Unit-tested the selection on a hand-built batch covering every case (duplicate texts, a TBox row, two
roles) over 2,000 draws: **0** violations. The three patches, applied in order to a fresh upstream
`82ef384`, reproduce `OnT/` exactly. `train_ont.py` gained `--data-prefix` and `--in-batch-negs`; an
in-batch run also logs how many exist rows actually received a negative.

**The split baseline** — `full`, split data, *no* in-batch term (galen, batch 32, 3 epochs; 486 steps,
8.6 min at 1.06 s/step, peak 1.24 GiB; loss 2.33 → 0.76 → 0.56 → 0.53). `phase5_checks.py --split`
reports each side separately — the first numbers in this log on materials the model never saw:

| `full`, split baseline | train (199) | **held-out (51)** |
|---|---|---|
| materials typed `substance` (before → after) | 0% → 100% | **0% → 100%** |
| crystals typed `crystal structure` (before → after) | 98% → 79% | 98% → 76% |
| hasStructure among that side's crystals, via role (random) | 0.063 (0.030) | **0.139** (0.089) |
| same, untrained plain distance | 0.269 | 0.537 |
| hard-negative, via role (untrained plain) | 0.769 (0.813) | 0.715 (0.910) |

Typing generalizes cleanly to unseen materials. hasStructure does not: on held-out materials the
trained role (0.139) is barely above random (0.089) and far *below* the untrained model's plain
distance (0.537). That's the number the in-batch run has to beat.

**The other three runs were paused, then run.** They were first held back on request with a placeholder
that makes `train_ont.py` refuse to start (it won't overwrite a run folder holding different data) —
all three refused with 0 steps and the GPU went idle — then run later the same day with exactly the
commands below.

**One crash on the way, caused by a driver update — not the code.** The first `full` in-batch attempt
died at step 357/486 with `CUDA error: unknown error`, right after cuBLAS logged
`CUBLAS_STATUS_EXECUTION_FAILED` on an ordinary matmul, inside upstream's plain encode pass
(`hit_loss.py:46`), not the new in-batch code. No driver-reset event was logged; the laptop had resumed
from sleep three minutes before the run. Then the actual cause surfaced: the NVIDIA driver had
**auto-updated mid-session, 610.62 → 617.14** (CUDA 13.3 → 13.4) — replacing the kernel driver kills
every live CUDA context. Checked before retrying: `torch 2.14.0+cu130` still sees the GPU on the new
driver, and 200 large matmuls matched an fp64 reference to 0.001. The crashed attempt is kept as
`data/runs/full_split_inbatch_crashed_step357/`; the rerun reproduced its losses exactly (11.6920 at
step 1, 1.8305 at step 100, 1.5516 at step 200) and finished cleanly. Worth pausing automatic driver
updates during long runs. The session now also holds the machine awake while runs are going.

**Did the in-batch term actually fire?** Yes, counted rather than assumed: `rows_with_negative` 6,632 of
15,522 sampled exist rows (`full`) and 6,547 (`no_geometry`) — ~43% of all exist rows, i.e. ~85% of the
eligible ABox rows (the other half are TBox rows, excluded on purpose). Slightly fewer in `no_geometry`
because identical symmetry-only crystal texts are never paired. Step-1 loss went 8.04 → 11.69 with the
term on — the extra term contributing, not a worse start. Cost: none measurable (1.058 vs 1.064 s/step).

**Results, all four split runs** — held-out = the 51 materials in no training row; "among test
crystals" ranks each held-out material's crystal among only the 51 held-out crystals:

| held-out (51) | `full` base | `full` **in-batch** | `no_geometry` base | `no_geometry` **in-batch** |
|---|---|---|---|---|
| materials typed `substance` | 100% | 100% | 100% | 100% |
| crystals typed `crystal structure` | 76% | 76% | 76% | 76% |
| hasStructure via role, among test crystals (MRR) | 0.139 | **1.000** | 0.278 | **0.660** |
| — plain distance, no role | 0.274 | 1.000 | 0.409 | 0.651 |
| — hard-negative (random 0.679) | 0.715 | 1.000 | 0.896 | 1.000 |
| — among all 250 crystals | 0.045 | 1.000 | 0.106 | 0.302 |
| untrained, plain, among test crystals | 0.537 | 0.537 | 0.174 | 0.174 |
| *train side*, via role, among train crystals | 0.063 | 1.000 | 0.157 | 0.435 |
| median s/step · training time | 1.064 · 8.6 min | 1.058 · 8.9 min | 0.608 · 5.2 min | 0.617 · 5.1 min |

Random among 51 test crystals: MRR 0.089. Nothing collapsed in any run (0 confusable pairs within
1e-3; in-batch runs spread same-family materials further apart — within-family median ~3.6 → ~10.9).

**`full` in-batch scored a perfect 1.000 on unseen materials — too good to accept without checking.**
Step 16 deliberately left one shared channel in `full`: a material's sentence repeats its crystal's
exact `Lattice:` / `Volume:` / `Sites:` lines, and in-batch negatives are exactly the pressure that could
teach string-matching them. Tested on the held-out 51, perturbing sentences at evaluation time only:
- material's geometry lines **removed**: MRR 1.000 → **0.990** — so it isn't matching the material's
  lattice string;
- crystals' **lattice + volume** lines shuffled between crystals: **50/51** picks stay on the true
  crystal, **0/51** follow the moved numbers — the lattice numbers are ignored;
- crystals' **`Sites:` composition line** shuffled: only **12/51** stay — that's the signal.

So `full` learned to match a formula to its crystal's unit-cell composition (LiTi₂O₄ ↔ `Li:2, O:8,
Ti:4` — a ratio mapping, not a string copy), supported by the symmetry both sentences share, and does
it on materials it never saw. The honest caveat: every one of the 250 formulas in this KG is unique,
so composition alone identifies the crystal — `full`'s 1.000 shows the relation is now *learned*, but
it's an easy version of the task. A hard version needs polymorphs (same formula, different crystals),
which this KG doesn't contain. The baseline model, same probe: 2/51 follow the shuffled block — it
never learned this link at all.

**`no_geometry` is the informative one** — its crystals carry no composition, only family, space group
and crystal system (28 distinct texts among the 51 held-out crystals). The best any model can do there
is rank by symmetry perfectly and guess within same-symmetry groups: computed ceiling MRR **0.724**
held-out (0.506 train side). In-batch reaches **0.660 held-out — 91% of the achievable maximum** —
against the baseline's 0.278 (38%). That can't come from composition, because there isn't any: it's
the relation working as intended on symmetry alone.

**Conclusion.** Step 16's diagnosis was right: with contrast only against class labels, hasStructure
wasn't learned (held-out 0.139 / 0.278, *below* the untrained model in `full`). Giving the exist loss
crystal-vs-crystal negatives fixes it — on held-out materials, at no cost to typing (100% / 76%
unchanged in every run), no collapse, no speed cost. Recommended default for further runs:
`--in-batch-negs` on.

Commands used (from the repo root, ~9 / 5 / 5.5 min each on the RTX 4060):

```
.venv\Scripts\python train_ont.py --variant full --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --in-batch-negs --output data/runs/full_split_inbatch
.venv\Scripts\python train_ont.py --variant no_geometry --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --output data/runs/no_geometry_split_base
.venv\Scripts\python train_ont.py --variant no_geometry --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --in-batch-negs --output data/runs/no_geometry_split_inbatch
.venv\Scripts\python abox/phase5_checks.py --run data/runs/<run> --variant <variant> --split data/split.json
```

*(Pipeline status: leak fixed; held-out split in place; hasStructure learned and generalizing with
in-batch negatives. Still open: the spec's type-loss ablation; a real `val.json` — `best_lambda`
still comes from 2 TBox queries; refit the numeric scaler (Step 9) on `split.json`'s train materials
before export; and polymorphs, if hasStructure is ever to be tested harder than composition matching.
The 250-material KG was rebuilt on this machine in Step 18.)*


## Step 18 — Rebuilt the 250-material KG on this machine: identical material data (2026-09-23)

The KG K-Ont was built from (`battGPT/output/battgpt_kg_ont/battery_kg.ttl`) only ever existed on the
Mac — `battGPT` never committed it (Step 5), and the `output/battgpt_kg` folder on GitHub is the older
12-material KG (83,977 triples), not this one. Our own record says which file it was:
`entities.json`'s `meta.source_ttl` is `../output/battgpt_kg_ont/battery_kg.ttl`, 325,138 triples.

**Picked the right code version first.** `battGPT`'s latest commit (`f8a9958`, 23 Sep) postdates the KG
(extracted 22 Sep 09:31 UTC) and edits the ontology itself (`battgpt.ttl` +1,169/−540 lines), so it
would build a genuinely different KG. Checked out `50e635c` (22 Sep 09:18 UTC, "Add space-group/
stoichiometry heuristic ... candidate search") instead, cloned as a sibling folder. Python 3.12.10
(matching the Mac's `myenv312`) in `battGPT/.venv` with rdflib 7.6.0, pymatgen 2026.9.24, smact
4.0.0, mp-api 0.46.5. The MP key went in `battGPT/.env` (gitignored), checked with one live lookup
(`mp-5670` → LiTi2O4) without printing it.

**Same materials, checked before ingesting.** `populate_kg_ont.py --use-cache` reads the saved candidate
search (`K-Ont/data/candidate_materials.json`, copied to where the script expects it and hidden from
battGPT's git via `.git/info/exclude`). Calling the script's own `build_candidate_list` on it: the
**identical 250 material IDs** as the original KG — none added, none missing. The run: 250/250 ingested,
validation PASSED, exit 0.

**Compared, not assumed.** Validation report vs the original's: materials 250 = 250, crystals 250 =
250, sites 4,744 = 4,744, bonds 19,942 = 19,942, property nodes 2,118 = 2,118 — but triples **325,031 vs
325,138**. Re-extracted the rebuild with `abox/extract.py` and diffed it field by field against the
`entities.json` K-Ont trained on: **every material, crystal, unit cell, element, species, battery cell
and every property value is identical** (0 differences in any field). The only difference: three
space-group individuals present only in the original (`Fd3mIndividual`, `Ia3dIndividual`,
`PnmaIndividual`), referenced by no crystal. Traced them: they're defined in `f8a9958`'s ontology
(`hasCharacteristicSpaceGroup` for spinel/garnet/olivine) and absent from `50e635c`'s — so the original
was exported while that ontology edit was still uncommitted on the Mac, and the 107 missing triples are
ontology-level definitions carried in the export's ontology closure, not material data. (The original
`.ttl` isn't on this machine, so a triple-by-triple diff wasn't possible; the entity-level diff is the
evidence.) The newer pymatgen reproduced every site and all 19,942 bonds.

**Conclusion:** the rebuilt KG carries exactly the material data K-Ont was built and trained on; nothing
downstream needs regenerating. It lives in `battGPT/output/battgpt_kg_ont/` (uncommitted there, like the
original).

## Step 19 — Correction: what the no-geometry result actually shows (2026-09-23)

Step 17 called `no_geometry`'s held-out MRR (0.660 of a possible 0.724) "the relation working as intended
on symmetry alone" and said it "can't come from composition". The second half is true; the first
overclaims, and a probe shows why. The *material* sentence also states its own space group, crystal
system and structure family (they come from its crystal) — the same three fields that make up the
whole `no_geometry` crystal sentence. Removing those lines from the material sentences at test time
only (held-out 51, among 51 test crystals):

| material sentence at test time | in-batch model | standard model |
|---|---|---|
| as trained | 0.651 | 0.272 |
| without its space-group line | 0.333 | 0.131 |
| without its structure-family line | 0.524 | 0.268 |
| without all three symmetry lines | **0.093** (random 0.089) | 0.080 |

So the in-batch model learned to match the symmetry labels that a material and its crystal both state
— the only information that version shares between the two sentences — just as the `full` model
matches formula ↔ unit-cell composition (Step 17's probe). Both are genuine learned alignment, and
both generalize to unseen materials, but neither is the model inferring a structure from chemistry
alone. The deck's "strongest evidence" slide was reworded to say this.

## Step 20 — Compared with a teammate's parallel implementation, on the same data (2026-09-24)

A teammate built the same extension independently: [akshayks13/FYP_K-OnT](https://github.com/akshayks13/FYP_K-OnT).
Full comparison, measurements and a scale-up plan: [`COMPARISON_FYP_K-OnT.md`](COMPARISON_FYP_K-OnT.md).
In short:

- **Same data, verified.** His streaming extractor, pointed at our KG, returned the same 250 materials
  with zero field mismatches against our `entities.json`, so the pipelines could be compared directly.
- **His pipeline trained fine on this GPU** (220 s, 3 epochs, unmodified) and reported 100% held-out
  typing, hasStructure MRR 1.000 and 100% electrode-role accuracy.
- **Both relation numbers are leaks, measured:** the untrained encoder already scores hasStructure MRR
  0.895 (his crystal sentence is `"Crystal structure of <formula>"`); with the `Battery role:` line
  removed, his role accuracy falls to 0.895, exactly the majority-class share (17/19 cathodes).
  Typing is real in both.
- **His strengths:** a 5,000-material KG (8.37M triples), a streaming reader that handles it in about a
  minute, explicit hard-negative is-a rows for hasStructure, and fresh negatives per oversampled copy.
- **Next:** a same-data head-to-head of the two hasStructure training methods on leak-free sentences,
  then scale to 1,000 and 5,000 materials with polymorphs (estimates in the comparison doc).

Evidence kept: `data/fyp_compare/abox/phase5b_report.json`, `data/fyp_compare/leak_probe.py` + `.txt`.

## Step 21 — Head-to-head: our in-batch negatives vs the teammate's is-a rows, same data (2026-09-24)

**Setup.** Three ways of training hasStructure, on identical data: our KG, the Step 17 split (199/51),
and our leak-free sentences. Everything else unchanged (galen base, batch 32, 3 epochs, last epoch).
- **M1 ours:** relation rows (Concept = material sentence) + in-batch negatives (Step 17 runs, reused).
- **M2 his:** is-a rows `material ⊑ "has structure some <crystal>"` with explicit crystal wrong answers
  (one same-chemistry, then two others), relation rows with Concept = that text concept, no in-batch
  term. Ported into `abox/rows.py --hasstructure isa`, one change from his code: a wrong answer whose
  text equals the true crystal's is skipped (Step 16).
- **M3 both:** our relation rows + his is-a rows + in-batch negatives (`--hasstructure both`).

Checked before training: the default `--hasstructure relation` still reproduces the Step 17 rows
byte for byte; M2 and M3 add 474 (`full`) / 464 (`no_geometry`) is-a rows; no wrong answer equals its
right one; no held-out-only crystal text anywhere. One side effect: the merge rebalances the ontology
rows 1:1 against the larger material side (×17 instead of ×15), so M2 and M3 also see ~350 more
ontology rows than M1. M2 and M3 take ~20% longer (564 steps vs 486).

**Scoring** ([`abox/h2h_eval.py`](../abox/h2h_eval.py) → `data/runs/h2h_results.json`): each
held-out material ranks the 51 held-out crystals three ways, so no method is favoured by the rule it
trained for: via the rotation (ours), via the encoded text concept `"has structure some <crystal>"`
(his), and plain distance. Hard negatives are same-chemistry crystals drawn from all 250 (24
held-out materials have one). The scorer reproduces the Step 17 numbers exactly.

| held-out MRR (random 0.089) | `full`: rotation / text | `full`: hard-neg | `no_geometry`: rotation / text | `no_geometry`: hard-neg |
|---|---|---|---|---|
| untrained galen | — / 0.588 | 0.910 | — / 0.166 | 0.868 |
| standard OnT | 0.139 / 0.238 | 0.715 | 0.273 / 0.363 | 0.896 |
| **M1 ours** | **1.000 / 1.000** | 1.000 | **0.651 / 0.651** | 1.000 |
| **M2 his** | 0.863 / 0.948 | 1.000 | 0.637 / 0.651 | 1.000 |
| **M3 both** | 1.000 / 1.000 | 1.000 | 0.651 / 0.651 | 1.000 |

Typing is 100% for every trained model. `no_geometry`'s maximum is 0.724 (Step 17).

**What each relies on** (test sentences changed only; `data/runs/h2h_field_probe_full.txt`,
`abox/h2h_field_probe.py`):

| `full`, rotation / text | as is | crystal Sites shuffled | Lattice + Volume shuffled | material geometry removed |
|---|---|---|---|---|
| M1 ours | 1.000 / 1.000 | 0.317 / 0.306 | 1.000 / 1.000 | 0.990 / 0.971 |
| M2 his | 0.863 / 0.948 | 0.736 / 0.720 | 0.905 / 0.915 | 0.858 / 0.944 |
| M3 both | 1.000 / 1.000 | 0.352 / 0.371 | 1.000 / 1.000 | 0.980 / 0.990 |

For `no_geometry`, removing the three symmetry lines from the material sentence drops **every** method
to random (0.08–0.11), as in Step 19.

**Conclusions:**
- **`no_geometry`: no difference.** All three reach 0.651 of a possible 0.724, and all three match
  the symmetry labels both sentences state.
- **`full`: ours ≥ his.** M1 and M3 score 1.000; M2 scores 0.86–0.95. M2 isn't copying lattice
  strings (shuffling or removing them barely matters); it leans less on unit-cell composition (0.72
  when shuffled, vs 0.32 for M1) and spreads its matching over several shared fields, a bit less
  precisely.
- **Combining adds nothing:** M3 = M1, at ~20% more training time.
- **Keep M1** (no extra rows, top score). M2 is a valid alternative that needs no change to OnT's loss
  code, if a patch-free setup ever matters.
- **Limits:** one seed, 51 test materials. And none of the three shows more than matching information
  both sentences state, so the next step is still data with polymorphs (see
  [`COMPARISON_FYP_K-OnT.md`](COMPARISON_FYP_K-OnT.md) §5).

## Step 22 — Grew the KG: 1,000 materials with polymorph groups, on battGPT latest main (2026-10-01)

**Why:** Steps 17–21 all end the same way: hasStructure is only ever solved by matching information
both sentences state (formula, composition, symmetry labels). With one crystal per formula, that is
enough. The fix is data where it isn't: several crystals of the **same formula** (polymorphs).

**What changed (battGPT, branch `kont-grow-kg` from latest main `f8a9958`; not committed yet):**
- `pipeline/ingest/candidate_search.py`: `sweep_broad_pool()` (every Li/Na material on Materials
  Project with ≤0.15 eV/atom above hull, ≤40 sites, 2–5 elements) and `pick_polymorph_group()` (up to
  2 extra polymorphs per formula, within 0.05 eV/atom of the hull, most stable first).
- `scripts/populate_kg_ont.py --grow`: opt-in; the old 250-material path is unchanged without it.
  Fill order up to `--cap`: curated materials → every candidate the structure-family classifier
  labels → polymorph groups until 25% of the KG is in a group → the rest by energy above hull.
  Writes `selected_material_ids.json` (ids, groups, summary) next to the KG.
- Latest main changes the ontology (Garnet/Perovskite under OxideStructureFamily,
  `hasCharacteristicSpaceGroup`, BatteryCell `hasRole`), but **our TBox needs no change**: checked
  with rdflib against `f8a9958`'s `battgpt.ttl`, all 13 structure-family and 5 battery-role parents
  in `tbox/build_schema.py` match, no subclass is missing, and the 3 relations' domain/range match
  (belongsToElectrode's range is the documented simplification). `build_schema.py` already had
  Garnet/Perovskite under OxideStructureFamily.

**Measured:**

| | 250 KG (Step 18) | 1,000 KG (this step) |
|---|---|---|
| candidate formulas found | 248 | 9,386 |
| ingestion | 370 s | 1,386 s (1.39 s/material, 0 failures) |
| triples | 325k | 1,202,425 |
| validation | passed | passed, 0 errors, 0 warnings |
| structure family labelled | — | 436 (564 unlabelled) |
| battery role | — | 538 (462 none) |
| K-Ont extraction (`abox/extract.py`) | 9 s | 37 s, peak RAM 1.55 GB |

- **Polymorphs:** 102 formulas appear 2–3 times (251 materials). 96 of the 102 differ in space group,
  76 in crystal system and 29 in structure family. 16 groups have two members with identical
  (space group, system, family); only their geometry tells them apart.
- **Only 362 of 9,386 candidate formulas get a structure-family label.** Every one is included, but at
  5,000 materials ~90% will have no family. Family typing stops growing with the KG; material
  typing, hasStructure and the numeric vector do.
- The 5,000 selection (dry run): 527 polymorph groups, 1,250 materials. The 1,000 selection is nested
  inside it.

**K-Ont side, done:**
- **UTF-8 bug on Windows:** `extract.py`, `verbalize.py`, `rows.py`, `numeric.py`, `split.py` and
  `merge_datasets.py` used the platform default encoding, so on this machine they wrote cp1252
  (`g/cm³`, `μB`) and would crash on anything outside it. The 250-material files were UTF-8 only
  because they were built on the Mac. Now explicit `encoding="utf-8"` everywhere; re-extracting the
  1,000 KG gives a file identical to a converted copy except the timestamp.
- **`abox/split.py --group-polymorphs`:** a formula is one unit, so each polymorph group sits wholly
  on one side; stratified by family ("mixed" when polymorphs differ). Off by default, and the
  250-material `data/split.json` regenerates identically. On the 1,000 KG
  (`data/split_grow_1000.json`): 803 train / 197 test, 16 of 102 polymorph groups in test, 0 formulas
  split across sides.

- **Training data for the 1,000 KG** (`data/battgpt_merged_grow_1000_split_{full,no_geometry}/`):
  803 training materials → 9,248 ABox is-a rows + 1,574 ABox relation rows (803 hasStructure, 351
  hasStructureFamily, 420 belongsToElectrode) per variant; vocabulary check passed. Only 10 of 1,000
  materials have real electrode data (battery cells). **The oversampling problem arrived early:**
  balancing 1:1 now repeats the 174 TBox is-a rows 53× and the 3 TBox relation rows **525×**
  (merge_datasets.py warns). At 5,000 that would be ~260× / ~2,600×.

**First training on the 1,000 KG** (TBox kept 1:1 as in the 250 runs, so results compare; in-batch
negatives; 3 epochs, batch 32; `data/runs/{full,no_geometry}_grow1000_inbatch/`):

| | `full` | `no_geometry` |
|---|---|---|
| steps / wall time | 1,734 / 38 min | 1,734 / 19 min |
| median step (KG build sharing the CPU) | 1.27 s | 0.63 s |
| peak GPU memory | 1.25 GB | 0.93 GB |

Held-out (197 materials, in no training row), untrained → trained; `abox/phase5_checks.py`:

| | `full` | `no_geometry` | random |
|---|---|---|---|
| material typed "substance" | 0% → **100%** | 0% → **100%** | — |
| crystal typed "crystal structure" | 98% → 71% | 93% → 77% | — |
| hasStructure among the 197 test crystals (MRR) | 0.33 → **0.985** | 0.08 → 0.45 | 0.03 |
| hasStructure, same chemical system | 0.71 → 0.94 | 0.63 → 0.87 | 0.55 |
| **polymorphs (42 materials), as is** | 0.79 → **0.976** | 0.71 → **0.984** | 0.651 |
| **polymorphs, structure lines removed** | 0.66 → **0.651** | 0.63 → **0.647** | **0.651** |

- **New check F** (`phase5_checks.py`): rank each material's crystal among only its same-formula
  polymorphs. The material sentence restates its crystal's space group, crystal system, lattice,
  volume and site counts verbatim (e.g. LiVO2 `Fd-3m (#227)` vs `P2/c (#13)`), so "as is" can be
  solved by string matching; "structure lines removed" drops those lines from the material sentence
  at test time, leaving properties (band gap, formation energy, magnetization, stability, role).
- **Result: polymorph identification is exactly random once the shared lines are gone**, in both
  variants and on training materials too (`full` 0.671 vs 0.676 random). The model matches what both
  sentences state; it has not learned property → structure. This is the cleanest form yet of
  Steps 17/19/21's conclusion, and the 250 KG could not show it (no polymorphs).
- Material typing generalises (100% held-out); no collapse (class median distance 13.5 → 20.8,
  material nearest-neighbour min 0.37).
- **Crystal typing regresses** (`full` 99% → 65% over all crystals; 181 now nearest "polyanion
  structure family"): crystals drift toward their family class. Same direction as the 250 run (78%),
  stronger here.
- **Implication:** more materials alone won't produce structure learning. Before training at 5,000,
  either the material sentence must stop restating its crystal, or training needs a signal that
  forces property → structure.

**First 5,000 build: invalid, rebuilding.** It ran 13:18–15:46 (ingestion 2 h 4 min, 5,546,356
triples) but **validation failed**: 52 unit cells without lattice parameters. Cause: a ~7 min DNS
outage (15:00–15:07, `getaddrinfo failed` for api.materialsproject.org). For every material fetched
in that window `MPIngester` silently returned a placeholder record (formula = the material id, no
composition, no structure) and the script kept it. Fix in `populate_kg_ont.py`: a placeholder is
retried with backoff (30/60/120/240 s), and if any material still fails the script exits before
building the graph instead of writing fake materials. The 1,000 KG had 0 placeholders. Bad output
kept as `output/battgpt_kg_grow_5000_INVALID_dns_outage/`; rebuild started 15:49.

**Next:** build 5,000 (extraction should need ~8 GB if linear in triples); rows with a held-out split that keeps each polymorph group on one side; the scale
fixes from [`COMPARISON_FYP_K-OnT.md`](COMPARISON_FYP_K-OnT.md) §5 Step 3; retrain at 1,000.

**5,000 rebuild (same day):** passed, 0 errors, 0 warnings, 5,602,266 triples, no placeholders, no
retries needed. `abox/extract.py` on it: 210 s, peak RAM 7.31 GB (`data/battgpt_abox_grow_5000/`).
527 polymorph groups (1,250 materials; 489 differ in space group, 37 in structure family); all
1,000 materials of the 1,000 KG are inside it.

## Step 23 — Material sentences that don't restate their crystal: still no structure learning (2026-10-01)

**Why:** Step 22's check F showed polymorph identification is pure matching of the lines the
material sentence copies from its crystal. Test: train without them.

`abox/strip_material_structure.py` drops every structure line (space group, crystal system,
structure family, `Structure:`, lattice, volume, density, sites) from the material sentences of the
1,000 KG (`data/battgpt_abox_grow_1000_nostruct/`); crystal sentences unchanged; same split, rows,
settings (`data/runs/{full,no_geometry}_grow1000_nostruct/`, 20.5 / ~15 min).

| held-out (197) | `full` crystals | `no_geometry` crystals | random |
|---|---|---|---|
| material typing | 100% | 99.5% | — |
| hasStructure among the 197 test crystals | 0.881 | 0.091 | 0.03 |
| same chemical system | 0.60 | 0.54 | ~0.55 |
| **polymorphs (42)** | **0.651** | **0.613** | **0.651** |

- With `full` crystals the model still finds the crystal among all candidates (0.88), but only by
  composition (formula vs the crystal's `Sites:` counts); among same-chemistry crystals it is at
  chance. Polymorph scores equal random EXACTLY on test and train: since every member of a group is
  scored, that means every sibling gets the same ranking -- the material's properties (band gap,
  formation energy, magnetisation, stability) don't move it at all.
- **Conclusion:** with this setup OnT does not learn property → structure.

## Step 24 — battGPT pipeline: the 8 ontology terms MP could fill but the pipeline didn't (2026-10-01/06)

Ontology coverage, measured on the 1,000 KG against `battgpt.ttl`'s own 91 terms: 71 used.
8 more were available from Materials Project but never written. Fixed on battGPT branch
`kont-grow-kg` (not committed there yet):
- 5 property classes (`BandGapProperty` … `ShearModulusProperty`): nodes were typed only as
  `emmo:Property`; now also as their own class (`triple_generator.py`).
- `hasOxidationState`: live MP structures carry none; pymatgen's `BVAnalyzer` (bond valence, MP's
  own method) now assigns them; metals/intermetallics stay without (`pymatgen_processor.py`).
- `hasSmactValidity`: never written, and the old logic defaulted to "valid" whenever oxidation
  states were missing (= every live material). Now SMACT's own `smact_validity()`; unknown → nothing.
- `belongsToCrystalSystem`: space group → crystal system link was never asserted.
- Thin before: voltage/capacity came from a 10-entry cache; now MP's insertion-electrode endpoint is
  queried live (cache first), with an explicit rule when a material sits in several electrodes.

Rebuilt 1,000 KG (`output/battgpt_kg_grow_1000_v2`): valid, 1,212,386 triples; property classes
1,000 each (elastic 76); oxidation states on 18,370 / 20,374 sites (90%); SMACT valid 732, not 268;
voltage/capacity 296 materials (was 10). Now 79 / 91 terms used; the rest: 6 cell-operation terms
(need lab data, not in MP), 5 abstract parents, 1 schema axiom (`hasCharacteristicSpaceGroup`).
The 5,000 KG has not been rebuilt with these fixes.

## Step 25 — Towards novel cathode CIFs: Phase A (structure from composition) and Phase C (CrystaLLM) (2026-10-06)

**Goal reframed:** generate CIFs for NEW cathode compositions. A new composition has only its
formula, elements, guessable oxidation states and intended role -- no structure and none of MP's
structure-derived properties.

**Phase A** (`abox/phaseA_features.py` with battGPT's venv for pymatgen, `abox/phaseA_eval.py`):
predict the ground-state space group of 719 held-out formulas of the 5,000 KG (none in any OnT
training row) from composition only.

| method | SG top 1 | top 5 | cathodes (174), top 5 |
|---|---|---|---|
| most common | 0.17 | 0.38 | 0.37 |
| same stoichiometry pattern | 0.37 | 0.66 | 0.64 |
| nearest compositions | 0.27 | 0.55 | 0.53 |
| **random forest, composition features** | **0.50** | **0.73** | **0.78** |
| best OnT embedding (no_geometry, linear) | 0.36 | 0.59 | 0.60 |
| random forest + OnT embedding | 0.37–0.43 | 0.65–0.69 | 0.65–0.72 |

OnT embeddings of composition-only sentences add nothing (they make the random forest worse).
On CrystaLLM's own held-out test set (external; `abox/phaseC_predict_sg.py`) the same random
forest is much weaker: top-5 0.47 on 55 cathode-like, 0.26 on all 971 Li/Na compositions.

**Phase C** (`abox/phaseC_generate.py`, CrystaLLM small, `.venv-crystallm`): 100 held-out
compositions (55 cathode-like), 10 CIFs per prompt, StructureMatcher vs the known structure.

| prompt | valid | matches | found / 100 | cathodes found / 55 |
|---|---|---|---|---|
| composition | 95% | 2.7% | 5 | 1 |
| + RF top-1 space group | 83% | 0.8% | 1 | 1 |
| + RF top-5 | 82% | 0.2% | 1 | 1 |
| + true space group | 99% | 5.7% | 6 | 1 |

A correct space group helps (Na2LaOs 3→10 of 10; Li2ZnCr 0→10); our predicted ones hurt (wrong for
the findable materials, biased to 225). The small model finds 1 of 55 cathodes under any prompt.
Fixed on the way: current pymatgen writes monoclinic groups in full form (`P12_1/c1`) that
CrystaLLM's vocabulary lacks -- they were silently dropped until mapped to short form.

**Next:** the large CrystaLLM (2.2 GB, ~10 s/sample on the 4060, ~4.5 h for 55 × 3 prompts × 10)
moves to a workstation: [`WORKSTATION_SETUP.md`](WORKSTATION_SETUP.md).

**Same day, later: K-Ont only, reproducible.** Phase C no longer imports CrystaLLM from the
teammate's repo: the authors' `crystallm` package is sparse-cloned into `K-Ont/CrystaLLM`
(github.com/lantunes/CrystaLLM, commit 8e81a86, gitignored) and used unmodified. It calls
`SymmOp.as_xyz_string()`, renamed `as_xyz_str()` in current pymatgen -- unpatched, every generated
CIF fails post-processing (0 valid); `phaseC_generate.py` now adds the alias itself. Sampling was
unseeded (LiFeSO4F's oracle prompt: 8/10 one run, 2/10 the next); it is now seeded per material and
prompt (`--seed`, default 1337) and two identical runs gave identical results. The small-model table
above came from the unseeded run, so its per-material counts are one sample of that noise.

## Step 26 — Phase C scoring bug: the reference structures were incomplete (2026-10-07)

**Every Phase C match count so far is invalid.** CrystaLLM's test-set CIFs store only the asymmetric
unit with a placeholder `'x, y, z'` operator; they must go through the same `postprocess()` as
generated CIFs. Our script read them directly, so references were missing most atoms (Li2ZnSiO4: 8
sites instead of 32, volume per atom 4x too large) and the matcher compared full generated crystals
with fragments. Only space-group-1 references survive intact -- which is why LiFeSO4F (P1) was the
only material ever matched. A second trap on the way: the test CIFs are indented by pymatgen, and
CrystaLLM's `replace_symmetry_operators()` only recognises the compact layout the model writes, so
whitespace is collapsed first. `phaseC_generate.load_truth()` now does both; all 100 laptop
references rebuild to their full cell composition. Generated CIFs were always fine and are kept, so
re-scoring needs no regeneration (`abox/phaseC_nearmiss.py`, which also reports right composition,
right space group, loose match and volume ratio). New: a sanity set of 20 well-known materials
CrystaLLM trained on (`abox/phaseC_make_sanity.py`; LiCoO2 3/3 in a smoke test), CHGNet stability
scoring against the known structure (`abox/phaseC_chgnet.py`, ~15 s per relaxation on 4 CPU threads)
and an overnight queue for the workstation (`abox/phaseC_overnight.py`).

## Step 27 — Phase C, corrected: CrystaLLM recovers most unseen cathode structures (2026-10-08)

Workstation overnight queue (`abox/phaseC_overnight.py`, finished 03:26, all jobs exit 0), with the
Step 26 reference fix. Strict match = CrystaLLM benchmark tolerances; "found" = at least one of 10
samples matches.

| 55 unseen cathode-like | large: found | large: CIFs matching | small: found | small: CIFs matching |
|---|---|---|---|---|
| composition only | **44/55** | 64% | 37/55 | 47% |
| + RF top-5 space groups | 28/55 | 11% | 20/55 | 8% |
| + true space group | **50/55** | 81% | 43/55 | 58% |

Calibration on 95 non-cathode Li/Na test compositions: large 87/95 (composition) and 93/95 (true
space group), small 81/95 and 93/95 -- the pipeline works. CHGNet (5 candidates per prompt, relaxed):
89-94% of the large model's candidates end within 0.05 eV/atom of the known structure; picking the
lowest-energy candidate gives the right structure for 39/52 materials (composition) and 47/54 (true
space group); for 6-8 materials CHGNet finds a candidate below the known structure.

- **Steps 25's Phase C table and "1 of 55" were artefacts of the Step 26 bug.**
- A correct space group helps (cathodes 64% -> 81% of CIFs; others 46% -> 93%); our random forest's
  guesses hurt, because CrystaLLM's own space-group choice (70% right on cathodes) beats them (RF top-1
  ~24%). A KG signal helps only if it identifies the structure better than CrystaLLM already does.
- The sanity set (training-set materials) scored low (4-13/20) because several references were
  primitive or non-standard MP cells: 6/7 matched where the cell was the conventional standard cell.
  It is a flawed check, superseded by the calibration set.
- Next: a novel-cathode pilot (substitution -> CrystaLLM large -> CHGNet), Step 28.

## Step 28 — Novel-cathode pilot, set up (2026-10-08)

`abox/novel_propose.py` (laptop, needs the MP key): 658 cathode parents from the 5,000 KG (<= 0.05
eV/atom, conventional cell <= 40 atoms, a redox transition metal, no precious/toxic elements) ->
1,276 SMACT-valid substitution children (TM swap at the same oxidation state, or Li <-> Na) -> 311
already in Materials Project -> 965 novel; the 200 with the smallest ionic-radius mismatch (<= 2 per
parent, 145 parents) in `data/crystallm/novel_candidates.json` (108 Na, 92 Li; median 28 atoms;
median theoretical capacity 115 mAh/g, an upper bound). Each keeps its parent's conventional cell and
space group. Workstation queue `abox/novel_queue.py`: large CrystaLLM, 10 samples each for
composition-only and parent-space-group prompts (the latter a KG analogue), CHGNet relaxation of up to
5 per prompt (`abox/novel_chgnet.py`), lowest-energy structure kept per composition.

## Step 29 — KG-steered generation: chemistry priors from the KG, plausibility scoring (2026-10-08)

Goal restated: use the KG/ontology to steer CrystaLLM's output quality. Text hints about the structure
don't beat CrystaLLM's own judgement (Step 27; a KG parent's space group is right for only 10/22 test
cathodes with a plausible parent, and never where CrystaLLM failed), so the KG supplies what CrystaLLM
lacks: chemistry statistics. `abox/kg_priors.py` -> `data/crystallm/kg_priors.json` from the 5,000 KG:
bond-length distributions for 875 element pairs (433,674 CrystalNN bonds; Li-O median 2.10 A, Co-O
1.95 A), coordination-number frequencies per element (Mn: 6-fold at 706 of 897 sites), per-element
volumes (cell volume predicted to 6.7% median error). `abox/kg_steer.py` scores a generated
structure (bond z, coordination likelihood, volume fit, clashes; weights fixed a priori) and evaluates
on the Phase C runs whether the KG's top pick is the right structure more often than a random pick,
and how it compares with CHGNet's lowest-energy pick. ~5 s per structure (CrystalNN); runs on the
workstation.

## Step 30 — KG-steering results; steered sampling and ontology requirements (2026-10-08)

Workstation evaluation (`data/crystallm/kg_steer_eval.json`), Phase C generations, truth known:

| | random pick | KG top-1 | match rate, all / KG-plausible | KG-plausible share |
|---|---|---|---|---|
| large, composition (55 cathode-like) | 62.9% | **71.2%** | 63.7% / 69.7% | 75% |
| small, composition | 45.5% | **55.8%** | 47.8% / 63.3% | 57% |
| large, true SG | 81.6% | 83.3% | 81.6% / 89.4% | 74% |
| small, true SG | 58.8% | 61.1% | 59.7% / **79.8%** | 56% |
| large, composition (95 non-cathode) | 46.0% | 41.1% | 46.0% / 63.4% | 18% |

- In the cathode domain the KG's chemistry priors select the right structure 8-10 points more often
  and raise filter precision by 6-20 points. On the non-cathode (mostly intermetallic) set, the
  ionic-compound priors mislead the top-1 pick: the steering is domain knowledge, valid where the KG
  represents the domain.
- CHGNet's lowest-energy pick is still better on the same 5-candidate pools (75% vs 67%, large).
  Correction: the KG score is NOT much cheaper (CrystalNN ~5 s/structure vs CHGNet ~8 s).
- Next (`abox/steer_queue.py`): combined KG-filter -> CHGNet selection; KG-steered sampling
  (`abox/kg_steer_sample.py`: generate until 5 KG-plausible, max 40, vs the first 5 unsteered); the
  pilot's novel structures checked for KG plausibility and for keeping the parent's framework
  (`abox/novel_steer_analysis.py`, StructureMatcher.fit_anonymous vs the parent; 31 of 200 parents
  carry an ontology structure family), giving a KG-steered shortlist.

## Step 31 — Combined selection (laptop s1) and a validated stability screen (2026-10-08)

**Combined KG filter -> CHGNet** (`kg_steer.py --save-scores`, cathode-like, composition prompt; the
non-cathode re-scoring crashed on laptop memory and is not needed): large 75.0% (= CHGNet alone 75.0%)
with 10% fewer relaxations; small 59.6% vs CHGNet 57.7% with 22% fewer; true-SG prompt mixed (large 83.3
vs 87.0, small 72.2 vs 68.5). The KG's clearest value stays selection without a physics model (+8-10
points over random). Per-CIF scores: `data/crystallm/kg_scores_{large,small_seeded}.json`.

**Stability screen** (`abox/stability_hull.py`): CHGNet-relaxed energy on the Materials Project phase
diagram (MP's corrected entries for the chemical system). First run was wrong: it applied MP2020
corrections to CHGNet energies, which already include them (raw CHGNet vs MP corrected: median +0.026
eV/atom on 12 known cathodes) -- 58/60 landed on the hull. Fixed (raw CHGNet energy). On 60 known
cathodes from the KG: all 10 MP-unstable (> 0.03 eV/atom) flagged > 0.03; 21/24 MP-stable within 0.05;
93% within 0.05 of MP; MAE 0.022; bias +0.017 (reads slightly less stable); rank correlation 0.39 over
a narrow 0-0.15 range. Rule for novel candidates: predicted e_above_hull <= 0.05 eV/atom = plausible;
it separates stable from unstable but cannot finely rank near-stable ones.

## Step 32 — KG-steered sampling does not help; KG selection: small, real but uncertain gain (2026-10-08)

`abox/kg_steer_sample.py` on the laptop (large CrystaLLM, batches of 5 -- batches of 10 ran out of the
4060's 8 GB on 32-atom cells; 55 cathode-like test compositions, 103 min): generate until 5 candidates
are KG-plausible (max 40). On the 47 compositions where both sets exist, first-5 unsteered 67.7% vs
first-5 KG-plausible 67.9% correct (7 differ: 4 better, 3 worse) at 13.6 samples per composition instead
of 5. 3 compositions never produced the right composition (Na4Al3Fe(SiO3)8, Li4Mn2Fe3Te3O16,
K4Na4Mo5(WO8)3). Resampling until plausible is dropped as a mechanism: ~75% of candidates pass anyway.

KG top-1 minus random pick, bootstrap 95% CI over compositions:

| sample set | gain | 95% CI |
|---|---|---|
| Phase C large (composition) | +8.2 points | [+0.9, +16.6] |
| Phase C small (composition) | +10.3 | [+1.5, +19.4] |
| steered-sampling run, large | +1.7 | [-6.1, +9.5] |

The KG's value is in selecting among CrystaLLM's samples, and it is modest: positive in all three
independent sample sets, clearly above zero in two, uncertain in size at ~52 compositions per run.

## Step 33 — Task A-test: seeding CrystaLLM with a KG analogue steers its output (2026-10-08)

`abox/seeded_generate.py --set test` (laptop, large model, 82 min): the 22 test cathodes with a
chemically plausible one-swap KG analogue (`test_analogues.json`); target = the analogue's conventional
cell with the swap, volume scaled by the KG volume model; 10 samples per level, same seeds.

| | composition | + analogue SG | + analogue lattice | substitution (control) |
|---|---|---|---|---|
| valid | 94.1% | 99.5% | 100% | 100% |
| right composition | 93.2% | 99.1% | 100% | 100% |
| keeps the analogue's framework | 42.9% | **68.3%** | 67.3% | 100% (by construction) |
| matches the true structure (per CIF) | 34.6% | **44.5%** | 41.8% | 40.9% |
| compositions found | 11/22 | 11/22 | 10/22 | 9/22 |
| KG-plausible | 72% | 72% | 73% | 95% |

Paired per composition, + analogue SG vs composition, bootstrap 95% CI: **framework compliance +26.9
points [+13.7, +40.8] (12 compositions better, 1 worse)**; validity +5.5 [0.0, +14.1] (3 better, 0
worse); true-structure share +11.0 [+0.3, +24.7] (6 better, 2 worse). The lattice seed adds nothing
over the space group. When the analogue's SG is the true one (10/22) the SG seed finds 10/10
(composition 9/10); when it isn't (12/22), 1/12 (composition 2/12) -- seeding follows the KG
analogue, right or wrong. Best-energy structures are within a few meV/atom of plain substitution.

First direct evidence for the claim: conditioning on a KG-retrieved analogue makes CrystaLLM comply with
the ontology requirement (keep the framework) far more often and stay valid, at no loss in finding the
true structure. Limits: 22 compositions; whether the requirement is *right* depends on the analogue.
