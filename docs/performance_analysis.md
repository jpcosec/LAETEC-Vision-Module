# Training Pipeline Performance Analysis

## Current Setup (2026-03-15)

### Hardware
| Resource | Spec |
|----------|------|
| GPU | NVIDIA RTX 2060 — 6 GB VRAM |
| CPU | 6 cores |
| RAM | 31 GB total, ~24 GB available |
| Disk dataset | 137 GB (271k PNG images) |

### Current Config
- `batch_size: 28`
- `num_workers: 4`
- No `persistent_workers`, no `prefetch_factor`
- Images loaded from disk on-the-fly: PNG decode → resize 224×224 → normalize (per sample, per epoch)

---

## Bottleneck Diagnosis

**GPU utilization: 3%** — the GPU is starving for data.

The pipeline is CPU-bound: each training step the DataLoader must decode PNG files,
resize them, and normalize them before the GPU can do any work. The GPU finishes a
batch faster than the CPU can prepare the next one.

Additional issue: with 2 training processes running simultaneously, each with
`num_workers=4`, we have 10 processes competing for 6 CPU cores (oversubscribed).

---

## Improvement Options

### Option 1 — Preprocess to disk as `.pt` tensors (highest impact)
Run a one-time script that:
1. Loads each PNG
2. Resizes to 224×224
3. Converts to float32 tensor and normalizes (ImageNet stats)
4. Saves as `frame_XXXX.pt` next to the original PNG (or in a parallel directory)

Training then becomes: `torch.load()` → GPU. No decode, no resize, no normalize at runtime.

**Pros:** eliminates the main bottleneck entirely
**Cons:** one-time cost of ~1–2 hours processing; disk usage goes from 137 GB → ~163 GB (float32) or ~41 GB (uint8, normalize at load time which is cheap)
**Recommendation:** save as uint8 to keep disk footprint small, normalize in the DataLoader (single tensor op, negligible cost)

Dataset class change: replace `read_image() / 255.0` + transforms with `torch.load()` + normalize.

---

### Option 2 — Tune `num_workers` and `batch_size` (quick win)
With 6 CPUs and at most 2 concurrent training runs:
- **`num_workers: 2`** per process (avoids CPU oversubscription)
- **`batch_size: 96`** — GPU has ~4.4 GB free; current 764 MB usage for batch=28 suggests
  headroom for 3–4× larger batches per process. Start at 96, monitor VRAM.

Add to DataLoader:
- `persistent_workers=True` — avoids respawning worker processes each epoch
- `prefetch_factor=4` — queues more batches ahead of time

Expected effect: reduces GPU idle time between batches. Will not fully solve the
bottleneck if disk I/O is the limiting factor, but meaningfully improves throughput.

---

### Option 3 — RAM cache (not viable for full dataset)
Available RAM: 24 GB
Full dataset as float32: ~163 GB — does not fit
Full dataset as uint8: ~41 GB — does not fit

Partial caching (e.g. most-used subjects) is complex and not worth the implementation
cost given Options 1 and 2 are simpler and more impactful.

---

## Recommended Plan

**Step 1 (now):** Apply Option 2 — change `batch_size`, `num_workers`,
add `persistent_workers` and `prefetch_factor`. Zero risk, immediate effect.

**Step 2 (next):** Implement Option 1 — write a preprocessing script that converts
the dataset to uint8 `.pt` tensors. Update `CustomDataset` to load from `.pt` when
available, falling back to PNG otherwise (backward compatible).
Update `local.yaml` with the preprocessed dataset path.

**Step 3:** Re-run `nvidia-smi` during training to confirm GPU utilization improves.
Target: >60% sustained GPU utilization.

---

## Two Parallel Runs

Currently running 2 training processes simultaneously (764 MB each = 1528 MB total).
With batch_size=96, each process would use ~2.6 GB → 5.2 GB total, leaving ~0.9 GB
headroom. Tight but feasible. Monitor carefully.

If running a single process, batch_size can go up to ~160–180.
