# Changelog

## 2026-03-14

- Centralized all machine-specific dataset paths into `configs/paths/local.yaml` (gitignored), eliminating hardcoded paths across config files.
- Added `configs/paths/` as a Hydra config group; `local.yaml` is the default and must be created manually on each machine.
- Updated `configs/data/dataloader.yaml` and `configs/run/triplets.yaml` to reference paths via Hydra interpolation (`${paths.*}`).
- Updated README with setup instructions for `configs/paths/local.yaml` and correct usage for both classification and triplet pipelines.
- Fixed subject-based data splitting: created pre-split `train.csv`, `val.csv`, `test.csv` with zero subject overlap (42/5/6 subjects respectively).
- Added `get_data_loaders_from_split_files()` to replace runtime splitting that caused subject leakage.

## 2026-03-13

- Added Hydra run group under `configs/run/` with `default` and `triplets` pipelines, keeping baseline behavior unchanged via `run=default`.
- Updated `configs/config.yaml` to compose `run: default` and route output folders through `run.output_subdir` so each pipeline controls its own output layout.
- Added `src/engine/triplet_train.py` to package the notebook triplet experiment into a reusable training pipeline (subject-balanced batches, dual/shared embedding heads, batch-hard triplet losses, and history/checkpoint export).
- Updated `src/train.py` to dispatch by `cfg.run.pipeline`, preserving legacy classification flow and enabling triplets through `python src/train.py run=triplets`.
- Updated `src/engine/train.py` to support both split-file and legacy loader signatures for baseline runs while explicitly guarding against triplet mode in that entry point.
- Added `docs/entrypoint.md` as a new-user starting guide and referenced it from `README.md`, including the source-file differences between baseline and triplet data inputs.

## 2026-03-11

- Added `docs/recreation.md` with per-experiment run requirements for dataset preparation, splitting, dataloader checks, training, evaluation, and metrics workflows.
- Documented current environment gaps and path requirements needed to run the pipeline on this machine.
- Added `notebooks/triplet_multiloss_workbench.ipynb` to prototype batch-hard triplet training with MobileNetV3 bottleneck embeddings, dual/shared heads, and triplet mining visualizations.
- Added notebook cells to visualize mined triplets as real anchor/positive/negative images with subject/stress metadata and hard distances.
- Moved and documented triplet dataset construction flow at the top of the notebook, including data stack overview and batch-level diagnostics for valid ID/stress anchors.
