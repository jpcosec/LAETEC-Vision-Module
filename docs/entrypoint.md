# New User Entrypoint

This guide gets you from clone to first run with the two supported Hydra
pipelines:

- `run=default` for the baseline binary classifier flow.
- `run=triplets` for the triplet multi-loss experiment.

## 1) Install

```bash
uv venv .venv
source .venv/bin/activate
uv sync --group dev
```

## 2) Pick your pipeline

### Baseline classifier (`run=default`)

Input source:
- `configs/data/dataloader.yaml`
- Required keys: `train_csv`, `val_csv`, `test_csv`
- CSV format expected by current loaders: semicolon-separated (`;`)

Run:

```bash
python src/train.py
```

Optional explicit form:

```bash
python src/train.py run=default
```

### Triplet experiment (`run=triplets`)

Input source:
- `configs/run/triplets.yaml`
- Uses `run.triplets.manifest_path` and manifest columns (`image_col`,
  `subject_col`, `stress_col`)
- Default manifest separator: `,` (`run.triplets.manifest_sep`)

Run:

```bash
python src/train.py run=triplets
```

Useful overrides:

```bash
python src/train.py run=triplets \
  run.triplets.manifest_path=/path/to/images_manifest_labeled.csv \
  run.triplets.mode=dual \
  run.triplets.epochs=10
```

## 3) Outputs

- Baseline: `outputs/{model}/{seed}/`
- Triplets: `outputs/triplets/{mode}/{seed}/`

Triplet runs save:
- Best checkpoint (`triplet_{mode}_{seed}_{timestamp}.pth`)
- Epoch history CSV (`triplet_history_{mode}_{seed}.csv`)

## 4) Source file differences (important)

- `src/train.py` is the canonical entry point for both run pipelines.
- `src/engine/train.py` remains for baseline-only modular training and blocks
  `run=triplets` by design.

This separation keeps older workflows working while allowing triplet runs
through Hydra config selection.
