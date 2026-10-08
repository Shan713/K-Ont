# Task D notes: KG-conditioned diffusion (MatterGen), workstation

Working notes for Task D of `docs/HANDOFF_KG_STEERING.md`. To be folded into BUILD_LOG by the laptop session.
Status at the time of writing: D0 done, environment isolated and verified, D2 data prepared, `battery_role`
embedding written and unit-tested. **No training has been run** (waiting for the novel pilot's CHGNet jobs).

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

Findings for the 2026-09-29 checkpoint: the `state_dict` has 303 tensors = 285 base GemNet tensors (54.8 M
parameters, 209 MB) + 16 adapter tensors (`cond_adapt_layers.oxidation_states.*`, 3.15 M parameters) + 2
property-embedding tensors. It therefore **holds the full base weights**. It loads standalone, offline and on
CPU (`MatterGenCheckpointInfo(model_path=<run dir>, load_epoch="last")` + `CrystalGenerator.prepare()`),
giving a `DiffusionLightningModule`; nothing was written into the run folder. `model.parameters()` counts
48.8 M against 58.0 M in the `state_dict` (difference not explained; probably buffers). The files in
`~/mattergen/checkpoints/*` are 134-byte Git-LFS stubs (expected), so the 440 MB `mattergen_base` download is
**not needed** to start from this checkpoint. Not verified: that its frozen base equals the official
`mattergen_base` weights (it should, the base is frozen, but the official file is not available to compare).

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

## Open points before D3 (training)

1. Start from the adapter checkpoint as a second property next to `oxidation_states`, or alone? Whether
   MatterGen can stack a second property adapter on an already-adapted checkpoint (`adapter.model_path` to the
   run folder) has not been tested. An alternative that avoids stacking: extract the 285 base tensors into a
   `mattergen_base`-style checkpoint folder and fine-tune from that.
2. Class balance: 98 negative vs 559 positive vs 1,582 none; consider weighting or oversampling.
3. Memory: the training must wait for the pilot (GPU and CHGNet memory); WSL sees 31 GB RAM.
4. The teammate's `generator.py` patch (bypass of the CSP assertions when `target_compositions_dict` is set) is
   inherited by our copy; it is irrelevant for property-conditioned generation.
