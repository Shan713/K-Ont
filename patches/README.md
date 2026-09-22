# Patches for the vendored `OnT/` checkout

`OnT/` is gitignored (it's a third-party clone — see the root `.gitignore` and `BUILD_LOG.md`
Step 1), so any change made directly inside it is invisible to `K-Ont`'s own git history and would
be silently lost the moment someone re-clones `OnT/` fresh. Patches applied to `OnT/` in the course
of this project are kept here instead, as plain `git diff` output, so they survive a re-clone.

**After cloning `OnT/` fresh, apply every patch here before running anything in `tbox/` or a
training script:**

```bash
cd OnT
git apply ../patches/ont_pipeline_device_fix.patch
```

| Patch | What it fixes | Why (full story in BUILD_LOG.md) |
|---|---|---|
| `ont_pipeline_device_fix.patch` | `ont/pipeline.py`'s device selection only ever checked for an NVIDIA GPU (`cuda`), defaulting to CPU even when Apple Silicon's `mps` backend is available — and the training loop independently re-detected the device on its own, disagreeing with the model's placement and crashing ("Passed CPU tensor to MPS op"). Fixed both to agree, and added an explicit `device=` override to `fit()`. | Step 14 |

MPS itself still isn't usable end-to-end in this codebase as of this patch — it now loads onto
`mps` cleanly but hits an out-of-memory error on the very first real training step (tried to
allocate 20GB for a 33M-parameter model), a separate, deeper issue this patch does not fix. Pass
`device="cpu"` to `fit()` until that's root-caused.
