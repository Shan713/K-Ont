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

## Step 17 — A held-out split, and in-batch negatives for the exist loss (2026-09-23, in progress)

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

**Stopped here on request** — the other three split runs (`full` + in-batch, and both `no_geometry`
runs) were queued and deliberately not trained. They were blocked with a placeholder that makes
`train_ont.py` refuse to start (it won't overwrite a run folder holding different data); all three
refused with 0 steps and the GPU went idle, then the placeholders were removed. To run them:

```
.venv\Scripts\python train_ont.py --variant full --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --in-batch-negs --output data/runs/full_split_inbatch
.venv\Scripts\python train_ont.py --variant no_geometry --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --output data/runs/no_geometry_split_base
.venv\Scripts\python train_ont.py --variant no_geometry --epochs 3 --batch-size 32 --data-prefix data/battgpt_merged_split --in-batch-negs --output data/runs/no_geometry_split_inbatch
.venv\Scripts\python abox/phase5_checks.py --run data/runs/<run> --variant <variant> --split data/split.json
```

(~9 / 5 / 5.5 min on the RTX 4060.) What to look at: held-out `hasStructure_via_role_among_test_crystals`
against the baseline's 0.139 and the untrained 0.537; the train.log line `In-batch exist negatives:`
(`rows_with_negative` should be a large share of `rows_seen` — near 0 would mean the term never fired);
and that held-out typing stays at 100%.

*(Pipeline status: leak fixed; held-out split in place; hasStructure measured honestly and currently
not learned, on train and held-out alike. Next: the three runs above. Still open: the type-loss
ablation, and a real `val.json` — `best_lambda` still comes from 2 TBox queries. The numeric scaler
(Step 9) should be refit on `split.json`'s train materials before any export.)*
