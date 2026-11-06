# P3 — Mixed-site templates (design, 2 Nov 2026)

## Why

P1 found that the 6 senary and septenary systems (Li + 4–5 metals + O, 900 formulas) have **no template at all**. A plain template needs a known material with the same formula pattern, and our pool has none with 5–6 cations. About 300 ternary and quaternary formulas also lack a template.

Multi-metal cathodes such as NMC (LiNi⅓Mn⅓Co⅓O₂), LNMO (LiNi½Mn₃⁄₂O₄) and high-entropy oxides are made the same way in practice: one known framework, with several metals sharing one cation sublattice. P3 builds structures exactly like that.

## Idea in one line

Take a known single-metal Li oxide framework ("host"), enlarge its cell, and spread the system's metals over the host's metal sites in low-electrostatic-energy orderings. K-Ont chooses the hosts.

## Data check (2 Nov)

- **Hosts.** 349 stable-ish (E_hull ≤ 0.05 eV/atom) single-metal Li–M–O oxides in the 10,267 pool, where M is a transition metal or Al. The common ratios are:

  | Ratio | Hosts | Type |
  |---|---|---|
  | LiMO₂ | 37 | layered / rock-salt type |
  | Li₂MO₃ | 27 | Li-rich layered |
  | LiM₂O₄ | 18 | spinel |
  | Li₂MO₂ | 15 | |
  | Li₇M₅O₁₂ | 15 | |
  | Li₈MO₆ | 14 | |
  | LiM₃O₄ | 13 | |
  | Li₃M₄O₈, Li₂M₃O₆, Li₅MO₄, Li₃MO₄, … | | |

- **Backtest set.** 1,094 multi-metal Li–M–O oxides already in the pool: 936 with 2 metals, 158 with 3. Their space groups are mostly the low-symmetry ordered supercells of these frameworks (C2/m 209, Cm 189, P-1 186, C2 95), which is exactly what P3 produces.

## Method (host-first)

Formulas for these systems are generated **from the hosts**, not taken from P1. A P1 formula such as LiAlFeMnNiO₆ (Li:M:O = 1:4:6) has no realistic framework. A host-derived formula such as Li₆(Al Fe Mn₂ Ni V)O₁₂ sits in the layered LiMO₂ framework by construction.

1. **Hosts.** Take the 349 single-metal Li–M–O oxides above, keep the lowest-energy one per (reduced formula, space group), with at most 24 atoms in the primitive cell.

2. **Supercells.** For each host, use diagonal and simple non-diagonal supercells (pymatgen `make_supercell`). Keep cells with
   - total atoms ≤ 48;
   - a number of metal sites N_M that can hold every metal of the system at least once (N_M ≥ number of metals);
   - the most compact shape for each N_M (smallest ratio of longest to shortest lattice vector).

3. **Formulas per host supercell.** Split the N_M sites among the system's metals, each at least once. Keep a split when it charge-balances the host's fixed Li and O counts, using the same oxidation-state rule as P1 (Al³⁺, Fe²⁺/³⁺, Mn²⁺/³⁺/⁴⁺, Ni²⁺/³⁺, V³⁺/⁴⁺/⁵⁺, mixed valence allowed). This decides which hosts suit which chemistry:
   - LiMO₂ needs an average of +3, and every system can reach that;
   - LiM₂O₄ needs +3.5, so it needs enough Mn or V;
   - Li₂MO₃ needs +4, so it needs mostly Mn or V.

   Prefer near-equal splits (high-entropy-like), then splits close to known cathode ratios.

4. **K-Ont ranks hosts per formula.** This is the same retriever and query as P2: the composition-only sentence of the new formula against each host's full sentence. Use the top hosts by score, so K-Ont decides which framework a chemistry should take.

5. **Ordering on the metal sublattice.** Li and O stay on the host's Li and O sites; only the metal sites are decorated.
   - **Charges.** Each element gets one average oxidation state, chosen inside its allowed range so that the cell is neutral and as close as possible to the host's metal valence.
   - **Energy.** The Ewald energy is quadratic in the charges, E = ½ qᵀ A q. A (the Ewald interaction matrix for the fixed site positions) is computed once per host supercell with pymatgen `EwaldSummation`. After that, a swap of two metals costs O(N).
   - **Search.** For each formula: 64 random decorations, each refined by swap Monte Carlo (lowest-energy swap first, then a few random restarts).
   - **Keep.** The 3 lowest-energy decorations that are distinct under StructureMatcher (0.2 / 0.3 / 5°).
   - Low Ewald energy is the standard first filter for orderings (as in pymatgen `OrderDisorderedStructureTransformation`) and is cheap. Relaxation is not ours.

6. **Cell size.** Scale the cell volume by the atomic-volume ratio, exactly as in P2. Atom positions stay at the host's positions.

7. **Checks (P5).** Right composition; no two atoms closer than 0.5 Å; not a duplicate within the formula.

## Backtest (acceptance fixed before running)

To show that the method is sound, rebuild known multi-metal oxides from single-metal hosts:

- **Targets.** The 1,094 multi-metal Li–M–O oxides in the pool with ≤ 48 atoms. Hosts have a single metal, so no host can be the answer itself. A host containing one of the target's metals is allowed, because that is how such structures are actually derived (LiNiO₂ → LiNi½Mn½O₂).
- **Generation.** Run steps 2–5 for each target's exact composition and cell size.
- **Score.** found@k: one of the top k (host, ordering) candidates matches the known structure (StructureMatcher 0.2 / 0.3 / 5°, unrelaxed, as for P2).
- **References.**
  - random ordering on the same host (no Ewald);
  - host ranked by a composition rule instead of K-Ont;
  - P2 plain templates where an exact template exists.
- **Accept** if Ewald ordering beats random ordering on found@3, and the result goes into the paper.
- **Report** found@1/3/10 split by number of metals and by host type.

## Output

- **CIFs.** `data/gen30k/mixed/<system>/<formula>__h<rank>_o<k>__<host id>.cif`
- **Manifest rows.** Source `mixed_site_template`, host ID, host formula and space group, K-Ont rank and score, supercell matrix, Ewald energy rank, oxidation states used.
- **Quota (P6).** About 970 per system. For the 6 senary and septenary systems, P3 provides most of it and CrystaLLM tops it up. For the other systems, P3 adds diversity next to the P2 templates.

## Cost

- **Ewald matrix:** once per host supercell, at most a few seconds each, a few hundred supercells in total.
- **Monte Carlo:** O(N) per swap, so well under a second per formula. CPU only.
- **Full P3:** about 1 hour on 3 CPU workers, never alongside GPU jobs. The backtest is similar.

## Risks and choices

- **Li on metal sites** (Li-excess, e.g. Li[Li⅓Mn⅔]O₂) is not in v1: Li stays on Li sites. It could be added later as an extra "metal" in step 3.
- **Average charges in the Ewald model are an approximation.** Real mixed-valence ordering (e.g. Mn³⁺/Mn⁴⁺ charge ordering) is not resolved. This is acceptable for a pre-relaxation candidate generator, and it is stated in the manifest.
- **Senary and septenary formulas come from host ratios, not P1.** P1 keeps its list for the other systems, and the manifest records which source each formula came from.
