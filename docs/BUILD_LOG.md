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
