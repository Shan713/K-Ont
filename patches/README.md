# Patches for the vendored `OnT/` checkout

`OnT/` is gitignored (it's a third-party clone — see the root `.gitignore` and `BUILD_LOG.md`
Step 1), so any change made directly inside it is invisible to `K-Ont`'s own git history and would
be silently lost the moment someone re-clones `OnT/` fresh. Patches applied to `OnT/` in the course
of this project are kept here instead, as plain `git diff` output, so they survive a re-clone.

**After cloning `OnT/` fresh, apply every patch here, in this order, before running anything in
`tbox/` or a training script:**

```bash
cd OnT
git apply ../patches/ont_pipeline_device_fix.patch
git apply ../patches/ont_gpu_training_fixes.patch
git apply ../patches/ont_exist_inbatch_negatives.patch
```

All three were checked (BUILD_LOG.md Steps 15 and 17) to apply cleanly, in this order, to upstream
`82ef384` ("Release ontology-transformer 0.1.7") and reproduce the patched files exactly. Each patch
is a diff *on top of* the one before it — apply them in order.

| Patch | What it fixes | Why (full story in BUILD_LOG.md) |
|---|---|---|
| `ont_pipeline_device_fix.patch` | `ont/pipeline.py`'s device selection only ever checked for an NVIDIA GPU (`cuda`), defaulting to CPU even when Apple Silicon's `mps` backend is available — and the training loop independently re-detected the device on its own, disagreeing with the model's placement and crashing ("Passed CPU tensor to MPS op"). Fixed both to agree, and added an explicit `device=` override to `fit()`. | Step 14 |
| `ont_gpu_training_fixes.patch` | Three things, all found on the first real GPU run: (1) a `gradient_checkpointing=` option on `fit()` — without it one step on our long ABox sentences holds ~14 GiB of activations and OOMs an 8 GiB GPU (this was also the real cause of the Mac's MPS "OOM"); enabled on the HF encoder directly because the Trainer's own flag crashes with transformers 5.17 + sentence-transformers 6.1. (2) a `select_best_epoch=` option — upstream reloads the best-val-MRR epoch at the end, which on our 2-query `val.json` is noise. (3) `np.trapz` → `np.trapezoid` in `ont/evaluation/ranking.py` (removed in NumPy 2.4; crashed the end-of-epoch eval). | Step 15 |
| `ont_exist_inbatch_negatives.patch` | Opt-in `exist_in_batch_negatives=` on `fit()` / `LogicalConstraintLoss`: an extra in-batch contrastive term in the existential loss — each exist row whose filler is an *instance* sentence is contrasted against another same-role row's filler (∃r.D_j), never a TBox class-name filler and never identical text. Upstream borrows exist negatives from the concurrent type-row batch, which on our data are 96% class labels — so hasStructure never had to tell one crystal from another. Default off = upstream behaviour. | Steps 16–17 |

The MPS out-of-memory noted after the first patch is now understood (Step 15): it was activation
memory from ~200-token ABox sentences, not an MPS bug. `gradient_checkpointing=True` should fix it
on a Mac too, but that hasn't been tested there.
