# Setting up the RTX 4060 laptop for OnT training

Checklist for moving from the Mac (CPU-only, ~90s/training-step measured — see `BUILD_LOG.md`
Step 14) to a machine with a real GPU. Written for **native Windows** (no WSL2) since nothing in
this stack actually needs Linux — every package here (`torch`, `geoopt`, `sentence-transformers`,
`deeponto`, `rdflib`, `transformers`, `spacy`) ships Windows wheels, and DeepOnto's JVM bridge
(JPype) works natively on Windows too. If the laptop is Linux instead, every command below is the
same except venv activation (`source .venv/bin/activate` instead of `.venv\Scripts\activate`).

Work through this top to bottom — each step assumes the one before it succeeded. **Don't skip
straight to training and debug backwards; verify the GPU is actually visible to PyTorch (Step 4)
before installing anything else**, since that's the one failure mode that silently wastes the most
time if caught late.

---

## 1. Confirm the GPU + driver are ready

```powershell
nvidia-smi
```

Should print the RTX 4060 and a driver/CUDA version (e.g. "CUDA Version: 12.x"). If this command
isn't found, install/update the NVIDIA driver first (nvidia.com/drivers) — everything else in this
checklist depends on it. Note the CUDA version shown; you'll need it in Step 3.

## 2. Install prerequisites

- **Python 3.11** specifically (not 3.12/3.13) — matches what this project was built and tested
  against on the Mac. https://www.python.org/downloads/, check "Add to PATH" during install. On the
  RTX 4060 laptop (which only had 3.13) we used `winget install --id Python.Python.3.11 --exact
  --scope user`, then built the venv from that exact interpreter
  (`%LOCALAPPDATA%\Programs\Python\Python311\python.exe -m venv .venv`).
- **A JDK** (any recent LTS — 17 or 21 is fine; we used 23 on the Mac with no issues) — needed for
  DeepOnto's JVM bridge (JPype). https://adoptium.net/ is a straightforward choice on Windows.
- **Git** — https://git-scm.com/download/win

## 3. Clone the repos

```powershell
git clone https://github.com/Shan713/K-Ont.git
cd K-Ont
git clone https://github.com/HuiYang1997/OnT.git
git clone https://github.com/lantunes/CrystaLLM.git
```

`OnT/` and `CrystaLLM/` are gitignored inside `K-Ont` on purpose (third-party code, `CrystaLLM`
alone carries ~150MB of benchmark data) — see `BUILD_LOG.md` Step 1. `CrystaLLM` isn't needed for
today's training run specifically (that's a later phase — export + CrystaLLM wiring, not started
yet); clone it now anyway so it's there when you get to it.

**Apply the device-handling patch** (fixes a real bug found on the Mac — full story in
`BUILD_LOG.md` Step 14 and `patches/README.md`):

```powershell
cd OnT
git apply ../patches/ont_pipeline_device_fix.patch
git apply ../patches/ont_gpu_training_fixes.patch
cd ..
```

The second patch (added on the RTX 4060, `BUILD_LOG.md` Step 15) is what makes training fit on an
8 GiB GPU at all — without it the first step OOMs — plus a NumPy 2.4 fix for the end-of-epoch eval.

On a CUDA machine the original bug (Apple MPS vs CPU device disagreement) wouldn't actually have
fired — it's Mac-specific — but the patch also adds a `device=` override to `fit()` that's useful
for forcing/debugging device placement, so apply it anyway.

## 4. Set up the venv, install PyTorch WITH CUDA first, verify the GPU is visible

**Order matters here.** `requirements.txt` pulls in plain `torch` as one of `deeponto`'s
dependencies, with no CUDA pin — if you `pip install -r requirements.txt` first, you risk pip
grabbing a CPU-only build. Install the CUDA build of torch *first*, so it's already satisfied by
the time the rest of `requirements.txt` installs:

```powershell
python -m venv .venv
.venv\Scripts\activate

# Go to https://pytorch.org/get-started/locally/ and use the exact command it gives you for
# your CUDA version from Step 1 (Windows / Pip / Python / CUDA 12.x). It'll look like:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# (RTX 4060 laptop, driver CUDA 13.3: we used .../whl/cu130 -> torch 2.14.0+cu130. Check which
# cuXXX indexes actually carry a cp311 win_amd64 wheel rather than copying the example above.)

# STOP AND VERIFY before installing anything else:
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"
```

This must print `CUDA available: True` and the RTX 4060's name before you continue. If it prints
`False`, stop here and fix it (usually a CUDA-version mismatch between the driver from Step 1 and
the `cuXXX` wheel you installed) rather than proceeding — every later step will silently run on CPU
otherwise, exactly what we're trying to get away from.

Now the rest:

```powershell
pip install -r requirements.txt

# The spaCy English model deeponto's verbalizer needs. Its OWN auto-download looked like it
# succeeded but silently didn't install anything (see BUILD_LOG.md Step 12) -- install the wheel
# directly instead of trusting `python -m spacy download`:
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Verify:
python -c "import spacy; spacy.load('en_core_web_sm'); print('spacy model OK')"
```

## 5. Bring over the already-generated data (don't regenerate it)

Everything through the merge step (extract → verbalize → numeric → rows → TBox prep → merge) is
already built, run, and verified on the Mac. Regenerating it here would mean re-running Materials
Project searches, re-running DeepOnto, etc. for no benefit — none of that depends on which machine
runs it, only the actual OnT *training* does. Most of `K-Ont/data/` is deliberately gitignored
(it's large and regenerable — see `.gitignore`'s comments), so `git clone` alone won't bring it
over.

**Copy the whole `K-Ont/data/` folder from the Mac to this laptop** (13MB total as of the last
check — small enough for a USB drive, AirDrop-to-cloud-to-download, or `scp` over the local
network if both machines are on the same Wi-Fi):

```powershell
# from the Mac, whatever transfer method is convenient, e.g.:
#   scp -r data/ user@windows-laptop-ip:/path/to/K-Ont/
# then on Windows, confirm it landed in the right place:
dir data\battgpt_merged_full
```

You should see `train.jsonl`, `train_exist.jsonl`, `train_conj.jsonl`, `val.json`,
`concept_names.json`, `role_names.json`, `role_inverse.json`, `merge_summary.json` inside
`data\battgpt_merged_full\` (and the same again in `data\battgpt_merged_no_geometry\`) — those are
the two directories a training run actually points at.

*(If you'd rather regenerate from scratch for some reason — e.g. testing the full pipeline on this
machine too — you'd additionally need `battGPT/output/battgpt_kg_ont/battery_kg.ttl` (21MB, lives
in the separate `battGPT` repo, not committed anywhere) and that repo's own Python environment.
Not needed just to run training.)*

## 6. Smoke-test training for real

```powershell
.venv\Scripts\python train_ont.py --variant full --epochs 1 --batch-size 64 --base-model sentence-transformers/all-MiniLM-L12-v2 --output data\runs\smoke
```

`train_ont.py` copies the merged data into `<output>\data` (so `fit()` reuses it instead of
running DeepOnto on the TBox alone), turns on gradient checkpointing, logs the real device and
checkpointing state at step 1, and times every step with `cuda.synchronize()` into
`<output>\step_times.json`. *(An earlier version of this step used an inline `python -c` snippet —
it never configured logging, so `fit()`'s "on device: ..." line was silently dropped.)*

**Measured on the RTX 4060 laptop (Step 15):** median **2.54 s/step**, 98 steps in 4 min 8 s, peak
GPU memory 2.04 GiB, vs ~90 s/step on the Mac's CPU. If a run is anywhere near 90 s/step, the GPU
isn't being used — recheck Step 4.

## 7. Known gotchas from setting this up the first time (Mac, but some apply everywhere)

- **HuggingFace downloads are unauthenticated by default** and can be slow/rate-limited. If model
  downloads drag, set an `HF_TOKEN` environment variable (free HuggingFace account) before running
  anything — `pipeline.py` will pick it up automatically.
- **The spaCy model's own auto-downloader can silently "succeed" without installing anything** —
  always verify with the `python -c "import spacy; spacy.load(...)"` check in Step 4, don't trust
  the log line alone.
- **`ont/__init__.py` eagerly imports the entire training stack** (`ont.model` → `ont.hit` →
  `torch` + `geoopt` + `sentence_transformers`) the moment anything under `ont.*` is imported, even
  for something as small as TBox prep. Not a problem once everything's installed, just explains why
  a "lightweight" script can pull in ~2GB of dependencies.
