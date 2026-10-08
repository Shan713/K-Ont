# Task D notes: KG-conditioned diffusion (MatterGen), workstation

Working notes for Task D of `docs/HANDOFF_KG_STEERING.md`. To be folded into BUILD_LOG by the laptop session.
Status at the time of writing: D0 done, environment isolated and verified, D2 data prepared, `battery_role`
embedding written and unit-tested, official `mattergen_base` downloaded and verified, D4 protocol fixed.
**No training has been run** (waiting for the novel pilot to finish and push).

## D0. Model choice (MatterGen), from the official sources

- Repo `github.com/microsoft/mattergen` (3.5 MB), MIT licence for code and for the weights (README and the
  Hugging Face card `microsoft/mattergen`).
- Released checkpoints (Hugging Face, `checkpoints/<name>/{config.yaml, checkpoints/last.ckpt}`):
  `mattergen_base` (Alex-MP-20), `mp_20_base`, and property models `chemical_system`, `space_group`,
  `dft_mag_density`, `dft_band_gap`, `ml_bulk_modulus`, `dft_mag_density_hhi_score`,
  `chemical_system_energy_above_hull`. File sizes: `mattergen_base` 439.86 MB, `mp_20_base` 439.85 MB, property
  models 488-537 MB each.
- Fine-tuning on a new property: add the name to `PROPERTY_SOURCE_IDS` (`mattergen/common/utils/globals.py`),
  put it as a column in `train.csv` / `val.csv`, run `csv-to-dataset`, add a `property_embeddings/<name>.yaml`;
  `mattergen-finetune adapter.pretrained_name=mattergen_base data_module=<dm> ...`. Float properties can reuse a
  config; anything else needs a custom `PropertyEmbedding` subclass (as `space_group`, `chemical_system`).
- Limits (model card): up to **20 atoms per unit cell**, no noble gases, Z > 84, Tc, Pm.
- Requirements in the README: Python >= 3.10, torch 2.2.x with PyG wheels, Linux + CUDA (Windows not mentioned).
- Alternatives checked: DiffCSP (MIT; composition/type-conditioned only; torch 1.9 stack; <= 20 atoms on MP-20),
  DiffCSP++ (MIT; space-group / Wyckoff conditioning; no custom-label fine-tuning documented).
  CrystalFormer not checked.

## Decisions (from the user)

1. Task D is restricted to <= 20 atoms in the **primitive standard cell**. Main label `battery_role`
   (PositiveElectrode / NegativeElectrode / none); `structure_family` is too sparse (exploratory later).
   KG splits as given; never train on `test` or `excluded`. Later comparisons with CrystaLLM use the same
   <= 20-atom targets.
2. Platform: the existing WSL2 Ubuntu (`D:\Battery\WSL\Ubuntu`), not native Windows.
3. Main starting point: the 2026-09-29 `mp20_oxi` adapter run (base frozen). The 2026-06-19 `li_cathode_oxi`
   full fine-tune is a comparison only.
4. No training until the pilot's CHGNet jobs have finished. Everything on D:.

## Environment and isolation

The teammate's `~/mattergen`, its `.venv` and its `outputs/` are read-only for us. Our work:

- `~/taskD/mattergen`: copy of the package source (no `.venv`, no `outputs`, no checkpoints) with its own git
  repo (baseline commit = the teammate's working tree a245cf2 + their local patches).
- `~/taskD/.venv`: `rsync` copy of the teammate's venv (Python 3.10.20, torch 2.2.1+cu118, mattergen 1.0.3,
  mattersim 1.1.2, numpy 1.26.4, pymatgen 2024.10.29, pytorch-lightning 2.0.6), re-pointed with
  `pip install -e ~/taskD/mattergen --no-deps --no-build-isolation --no-index --no-cache-dir` (offline).
- Run everything from `~/taskD` with `~/taskD/.venv/bin/python -I ...` (not the venv's console scripts: their
  shebangs still name the old path). Leftover old path in the copy: `bin/jp.py` only (Jupyter helper, unused).
- Checks after the install: `import mattergen` from `~/taskD` resolves to
  `/home/cbscu4cse23142/taskD/mattergen/mattergen/__init__.py`; the teammate's editable finder still maps to
  `~/mattergen/mattergen` with unchanged timestamps; the teammate repo's HEAD, `git status` hash and `git diff`
  hash equal the pre-work baseline (`a245cf2b...`, `95a84a0f0e0be9c7`, `6a6530fcbb915e61`). One directory,
  `~/mattergen/.git/lfs/tmp`, has a recent mtime (probably git-lfs, from an ordinary `git status`/`git diff`
  before the `--no-optional-locks` rule); no file contents changed and `.git/index` is unchanged (2026-09-29).
- GPU inside WSL: `nvidia-smi` sees the RTX A2000 12 GB. WSL warns about an invalid `wsl2.swap` line in
  `C:\Users\...\.wslconfig` (left alone, as instructed).

## The oxidation-state adapter checkpoints (read only)

| run | data | setting | steps | file |
|---|---|---|---|---|
| `outputs/singlerun/2026-09-29/20-25-39` | `mp20_oxi` | `full_finetuning=false` (adapter, base frozen) | 5,800 | 219 MiB, `epoch=189-loss_val=0.36` |
| `outputs/singlerun/2026-06-19/06-28-11` | `li_cathode_oxi` | `full_finetuning=true` | 600 | 489 MiB, `epoch=174-loss_val=0.35` |

Findings for the 2026-09-29 checkpoint: the `state_dict` has 303 tensors = 281 base GemNet tensors (53.74 M
parameters) + 4 `cond_mixin_layers.oxidation_states.*` tensors (4 x 512 x 512 = 1.05 M parameters) + 16 adapter
tensors (`cond_adapt_layers.oxidation_states.*`, 3.15 M parameters) + 2 property-embedding tensors.
(An earlier version of this note counted 285 "base" tensors; the 4 extra are the mix-in layers.) It loads
standalone, offline and on CPU (`MatterGenCheckpointInfo(model_path=<run dir>, load_epoch="last")` +
`CrystalGenerator.prepare()`); nothing was written into the run folder. `model.parameters()` counts 48.8 M
against 58.0 M in the `state_dict` (difference not explained). The files in `~/mattergen/checkpoints/*` are
134-byte Git-LFS stubs (expected).

**Official `mattergen_base` downloaded** (approved) into `~/taskD/checkpoints_hf/checkpoints/mattergen_base/`
(`config.yaml` 5,485 bytes, `checkpoints/last.ckpt` 461,370,350 bytes; sha256
`81668ee12afc1ee1b037f362420730de3460bfd2d36e547585fdcb911a3dfdef`, equal to the oid in the Git-LFS pointer).
It has 281 tensors, 53.74 M parameters, epoch 2109, step 2,504,570. Tensor comparison with the 2026-09-29
adapter run: 281 of 281 common tensors **bit-identical, max abs difference 0.0** (the adapter's frozen base is
the official base). The 2026-06-19 full fine-tune: 105 of 281 identical, max abs difference 2.9e-3
(`fc_atom.weight`), as expected for a trained base.

## D2. Data (`abox/taskd_prepare_data.py`, stats in `data/crystallm/taskD_data_stats.json`)

Source `kg_training_source.json.gz` (5,000 materials). Kept: ordered, primitive standard cell
(`SpacegroupAnalyzer(symprec=0.1)`) with <= 20 atoms, MatterGen-supported elements. Only `train` and `val` are
written (`data/crystallm/taskD/{train,val}.csv`, gitignored; columns `material_id, formula_pretty,
battery_role, structure_family, cif, nsites, energy_above_hull, space_group`).

| split | total | kept | dropped (> 20 atoms / unsupported element) | positive | negative | none |
|---|---|---|---|---|---|---|
| train | 3,905 | 2,239 | 1,559 / 107 | 559 | 98 | 1,582 |
| val | 497 | 304 | 180 / 13 | 62 | 14 | 228 |
| test (counted, not written) | 493 | 277 | 198 / 18 | 55 | 16 | 206 |
| excluded (counted, not written) | 105 | 73 | 26 / 6 | 20 | 1 | 52 |

`structure_family` kept in train: LayeredOxide 102, Garnet 34, Spinel 8, Argyrodite 4, Perovskite 4, RockSalt 1,
none 2,086 (too sparse, as decided). NASICON, olivine and most garnets drop out because their primitive cells have
more than 20 atoms.

`csv-to-dataset` (MatterGen's own) accepts the CSVs; the cached `battery_role.json` counts equal the table
(cache in `~/taskD/dataset/kg_battery_role/{train,val}`).

Points to be aware of: only **98** negative-electrode training examples (14 in val); 71% of the data is `none`.

## `battery_role` embedding (our copy only; `git diff` against the baseline, 5 files, +105/-2)

- `globals.py`: `"battery_role"` added to `PROPERTY_SOURCE_IDS`.
- `dataset.py`: string-property pass-through now `if prop in ("oxidation_states", "battery_role")`.
- `conf/.../property_embeddings/battery_role.yaml`: `PropertyEmbedding`, conditional module
  `BatteryRoleOneHotEmbedding`, unconditional module `ZerosEmbedding`, as `oxidation_states.yaml`.
- `conf/data_module/kg_battery_role.yaml`: copy of `mp20_oxi.yaml` without the oxidation transform, pointing at
  `~/taskD/dataset/kg_battery_role`.
- `property_embeddings.py`: new class, inserted before `PropertyEmbedding`:

```python
class BatteryRoleOneHotEmbedding(torch.nn.Module):
    CLASSES: tuple = ("none", "PositiveElectrode", "NegativeElectrode")

    def __init__(self, emb_size: int = 512):
        super().__init__()
        self.emb_size = emb_size
        self._class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}
        self.vocab_size = len(self.CLASSES)
        self.embedding = torch.nn.Linear(in_features=self.vocab_size, out_features=emb_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def _one_hot(self, label) -> torch.Tensor:
        if isinstance(label, (list, tuple)):
            if len(label) != 1:
                raise ValueError(f"battery_role takes exactly one label per structure, got {label!r}")
            label = label[0]
        if label not in self._class_to_idx:
            raise ValueError(f"Unknown battery_role label {label!r}; expected one of {self.CLASSES}")
        one_hot = torch.zeros(self.vocab_size, device=self.device)
        one_hot[self._class_to_idx[label]] = 1.0
        return one_hot.reshape(1, -1)

    def forward(self, x) -> torch.Tensor:
        one_hots = torch.cat([self._one_hot(label) for label in x], dim=0)
        return self.embedding(one_hots)
```

Design: "none" is a real label (the KG records no battery role); "unconditioned" (classifier-free-guidance
dropout, unconditional sampling) is the `ZerosEmbedding`. Using an explicit "none" class also keeps every row
through the data module's `filter_sparse_properties`. Unit test (CPU): shape `(n, emb)`, different roles give
different vectors, same role the same vector, unconditional output is zeros, bad labels raise `ValueError`.
Not tested: training, generation, or the Hydra composition of the new configs.

## Decisions, round 2 (from the user)

1. Train a `battery_role` adapter from the **official** `mattergen_base` (`adapter.model_path` pointing at
   `~/taskD/checkpoints_hf/checkpoints/mattergen_base`, `full_finetuning=false`). No stacking on the teammate's
   oxidation-state adapter. Later experiment, not now: `battery_role` + `oxidation_states` together, both from
   the base.
2. No reweighting, no oversampling; the three classes stay as they are (98 negative vs 559 positive vs 1,582
   none in train). The target class is **PositiveElectrode**.
3. Before any long run: wait until the pilot has finished and pushed (including commit 5f8a424), then do a
   100-step timing run and report seconds per step, GPU memory and the projected time for ~5,000 steps. The full
   run only starts on the user's OK.
4. The D4 evaluation is fixed below, before training.
5. Isolation rules stay; the teammate baseline is re-recorded and compared at the end of each work block.

## D3 plan (nothing run yet)

- Command shape (to be confirmed by the timing run; not yet executed, the Hydra composition of the new configs is
  untested): `~/taskD/.venv/bin/python -I -m mattergen.scripts.finetune adapter.model_path=<official base dir>
  adapter.full_finetuning=false data_module=kg_battery_role data_module.properties=[battery_role]
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.battery_role=battery_role
  ~trainer.logger trainer.accumulate_grad_batches=<k> data_module.batch_size.{train,val,test}=<b>`, run from
  `~/taskD` so the Hydra outputs land there. The base `finetune.yaml` has lr 5e-6, `precision: 32`, DDP strategy,
  `check_val_every_n_epoch: 5`, a W&B logger (removed with `~trainer.logger`).
- Steps vs epochs: 2,239 training structures, so an optimizer step with micro-batch `b` and accumulation `k`
  sees `b*k` structures (proposal for the timing run: b = largest that fits, k so that b*k = 64, i.e. about 35
  steps per epoch and about 143 epochs for 5,000 steps; to be reported with the timing).
- Checkpoint used for D4: `last.ckpt` of the agreed run, fixed in advance (not chosen on D4 metrics).

## D4 evaluation protocol (fixed before training)

**Question.** Does conditioning on `battery_role=PositiveElectrode` make the generated structures more often
cathode-like than unconditional sampling, without losing validity or stability?

**Model.** The trained `battery_role` adapter for all sets below (so "unconditional" = the same network with the
property left out, i.e. the zero embedding, which by construction is the base model's unconditional score).
Optional reference set (needs the user's OK): the official `mattergen_base`, unconditional, to show the adapter
does not degrade the base.

**Sets (256 structures each).**

| set | `battery_role` | `diffusion_guidance_factor` |
|---|---|---|
| S0 unconditional | not given | 0.0 (not used) |
| S1 positive | PositiveElectrode | 1.0 |
| S2 positive (headline) | PositiveElectrode | 2.0 |
| S3 control | `none` | 2.0 (assumption: same factor as the headline; say if another is wanted) |

All sets: same batch size B and `num_batches = 256 / B`, same `num_atoms_distribution` (the default
`ALEX_MP_20`, at most 20 atoms), and the same seed (`pytorch_lightning.seed_everything(<seed>)` immediately
before each set, same seed value for every set) so initial noise and atom counts line up across sets. B is set
by the timing run (GPU memory). `CrystalGenerator` has no seed argument, hence the explicit seeding.

**Metrics per set** (valid = the denominator for everything except validity itself):

1. **valid** = `structure_validity` (pairwise-distance cutoff 0.5 A, MatterGen's own) **and** `is_smact_valid`
   (SMACT, `use_pauling_test=True, include_alloys=True`), both from `mattergen.evaluation.metrics.structure`.
   Denominator 256.
2. **cathode-like share** (primary) = valid structures that contain Li or Na **and** at least one redox-active
   transition metal from `TMS = [Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Nb, Mo]` (the list in `abox/novel_propose.py`).
   Reported two ways: share among valid, and yield over all 256. Secondary, stricter variant: additionally
   contains O, S or F (the Phase C "cathode-like" notion).
3. **KG plausibility** = `abox/kg_steer.kg_score(s)["plausible"]` share and mean score, on the valid cathode-like
   structures (the KG priors are for ionic Li/Na compounds; trap 6, so the all-valid number is shown only for
   reference).
4. **E_hull** = CHGNet-relaxed energy placed on the MP phase diagram, as in `abox/stability_hull.py` (CHGNet
   energies already include the MP2020 corrections; never apply `MaterialsProject2020Compatibility`). Reported on
   the valid cathode-like structures: share with e_hull <= 0.05 eV/atom and the median. `stability_hull.py` knows
   only the sets `calib` and `novel`; D4 needs a third set read from the D4 structures, added as a **new** file,
   not by editing that script. The MP fetch (chemical systems of the D4 structures) happens on the laptop with
   the MP key; the computation can run anywhere.
5. **unique** = fraction of valid structures that are distinct under `StructureMatcher(ltol=0.2, stol=0.3,
   angle_tol=5)` within the set.
6. **novel** = fraction of valid structures with no `StructureMatcher` match (same tolerances, reduced-composition
   prefilter) among (a) the 5,000 KG structures (`kg_training_source.json.gz`, all splits) and (b) MP. The MP
   part needs MP structures for the generated compositions, fetched on the laptop; (a) is computed here.

**Statistics.** Bootstrap 95% CIs (10,000 resamples, seeded), resampling structures within each set; ratio
metrics (shares among valid or among cathode-like) are recomputed on each resample. Differences between sets use
independent bootstraps of the two sets and the percentile interval of the difference. Report n for every share,
since the cathode-like subset may be small.

**Main claim and decision rule.** PositiveElectrode (S2) beats unconditional (S0) on the cathode-like share, at
equal or better validity and E_hull. Proposed pre-registered reading (the margins are my proposal, to be
confirmed or changed by the user before training):

- (a) the 95% CI of cathode-like share S2 - S0 has a lower bound above 0;
- (b) validity S2 - S0 has a lower bound >= -0.05 (non-inferiority margin of 5 points);
- (c) the share with e_hull <= 0.05 among valid cathode-like, S2 - S0, has a lower bound >= -0.10.

S1 vs S0 (guidance 1.0) is a dose-response check; S3 vs S0 is the control: if `none` gives the same lift, the
effect is not the label. A pass needs (a), (b), (c) together; anything else is reported as it is.

## Other notes

- Memory: no training until the pilot's CHGNet jobs are done; WSL sees 31 GB RAM, the card has 12 GB.
- The teammate's `generator.py` patch (bypass of the CSP assertions when `target_compositions_dict` is set) is
  inherited by our copy; irrelevant for property-conditioned generation.
- Not done: Hydra composition of the new configs, the timing run, any training or generation.
