# Running Phase C (CrystaLLM) on a workstation

Phase C asks: does a space-group hint help CrystaLLM generate the real structure of compositions it
never saw? The small model ran on the laptop (BUILD_LOG Step 25); the **large** model (≈200M
parameters, 2.2 GB) needs ≈10 s per sample on an RTX 4060 laptop, ≈4.5 h for the planned run, so it
moves here. Nothing in this run needs the Materials Project API key.

## 1. Code

K-Ont, plus the `crystallm` package from the CrystaLLM authors' repository (MIT licence), cloned
into `K-Ont/CrystaLLM` (gitignored). A sparse clone fetches only that package (~300 KB):

```bash
git clone https://github.com/Shan713/K-Ont.git
cd K-Ont
git clone --depth 1 --filter=blob:none --sparse https://github.com/lantunes/CrystaLLM.git CrystaLLM
git -C CrystaLLM sparse-checkout set crystallm      # laptop used commit 8e81a86
```

The CrystaLLM code is used unmodified; `abox/phaseC_generate.py` adds one compatibility alias
(current pymatgen renamed `SymmOp.as_xyz_string` to `as_xyz_str`).

## 2. Python environment (3.11)

```bash
python3.11 -m venv .venv-crystallm
source .venv-crystallm/bin/activate            # Windows: .venv-crystallm\Scripts\activate
pip install --upgrade pip
# PyTorch with CUDA: pick the index URL for the workstation's CUDA version from pytorch.org
# (the laptop used cu130: https://download.pytorch.org/whl/cu130)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install pymatgen chgnet omegaconf pandas tqdm pyyaml smact scikit-learn pyzmq
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

The generation code samples in bfloat16, so the GPU needs bf16 support (NVIDIA Ampere or newer:
RTX 30xx/40xx, A100, …).

Versions on the laptop: torch 2.14.1+cu130, pymatgen 2026.9.24, chgnet 0.4.2, numpy 2.4.6.

## 3. Data and model

The test set and our space-group predictions are in the repo; unpack the test CIFs:

```bash
cd data/crystallm
mkdir -p test_cifs && tar -xzf cifs_v1_test.tar.gz -C test_cifs     # 10,286 CIFs
```

Download the large model from the CrystaLLM authors' Zenodo record and check it:

```bash
curl -L -C - --retry 30 -o crystallm_v1_large.tar.gz \
  https://zenodo.org/records/10642388/files/crystallm_v1_large.tar.gz
md5sum crystallm_v1_large.tar.gz    # must be 7229ae633f832a935eb30dc7f58a8830
tar -xzf crystallm_v1_large.tar.gz  # -> crystallm_v1_large/ckpt.pt
cd ../..
```

(Optional, to re-run the small model: `crystallm_v1_small.tar.gz`, md5
`0221fbcd166bddb17f75be8a610892f3`.)

## 4. Run

From the K-Ont folder, with the venv active:

```bash
python -u abox/phaseC_generate.py --model crystallm_v1_large --n-materials 55 --samples 10 \
  --conditions composition,rf_top5,oracle --gen-dir phaseC_gen_large \
  --out data/crystallm/phaseC_results_large.json | tee data/crystallm/phaseC_large_run.log
```

- The 55 cathode-like compositions come first, so `--n-materials 55` is exactly those.
- Prompts: `composition` (plain CrystaLLM), `rf_top5` (our random forest's top 5 space groups,
  2 samples each), `oracle` (the true space group: the ceiling for any hint).
- Results are rewritten after every material, so an interrupted run keeps what it finished.
- Sampling is seeded per material and prompt (`--seed`, default 1337), so a rerun gives the same CIFs.
- Each line printed is one material: matches per prompt out of 10.

## 5. Send back

`data/crystallm/phaseC_results_large.json` and `phaseC_large_run.log` (small). The generated CIFs
(`data/crystallm/phaseC_gen_large/`) are only needed if we want to inspect structures.
