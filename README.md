# Vision-Module

**Repository:** https://github.com/271828D/Vision-Module

A PyTorch-based computer vision project for stress identification using pretrained models (EfficientNet, MobileNet). This module provides a configurable training pipeline with Hydra configuration management and Weights & Biases integration for experiment tracking.

New here? Start with `docs/entrypoint.md`.

## Features

- Multiple pretrained architectures (EfficientNet-B2, EfficientNetV2-S, MobileNetV3-Small)
- Hydra-based configuration system for easy experimentation
- Weights & Biases integration for experiment tracking
- Early stopping with patience-based validation
- CUDA 12.6 support for GPU acceleration
- Automated checkpoint saving with timestamps

## Prerequisites

- Python 3.12
- CUDA 12.6 (for GPU support)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/271828D/Vision-Module.git
   cd Vision-Module
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. (Optional) Set up Weights & Biases:
   ```bash
   wandb login
   ```

## Configuration

### Machine-specific paths (required)

All dataset paths are centralized in a single gitignored file:

```
configs/paths/local.yaml
```

This file is not committed to the repository. Each machine must create its own. Copy the template and fill in your paths:

```yaml
# configs/paths/local.yaml
train_csv: /path/to/your/train.csv
val_csv: /path/to/your/val.csv
test_csv: /path/to/your/test.csv
triplet_manifest: /path/to/your/images_manifest_labeled.csv
```

All other configs (`dataloader.yaml`, `triplets.yaml`) reference these paths via Hydra interpolation (`${paths.train_csv}`, etc.) — no hardcoded paths anywhere else.

### Configuration Files

- `configs/config.yaml` - Main config (seed, model, pipeline selection)
- `configs/paths/local.yaml` - **Machine-specific paths (gitignored, create manually)**
- `configs/model/` - Model architectures (`effb2`, `effv2s`, `mobilev3s`, etc.)
- `configs/data/dataloader.yaml` - Data loading parameters
- `configs/training/default.yaml` - Training hyperparameters
- `configs/run/default.yaml` - Classification pipeline config
- `configs/run/triplets.yaml` - Triplet loss pipeline config

## Usage

### Basic Training

```bash
python src/train.py
```

### Triplet Loss Pipeline

```bash
python src/train.py run=triplets
```

Triplet training uses `paths.triplet_manifest` from your `configs/paths/local.yaml`. The manifest must be a semicolon-separated CSV with these columns:

| Column | Description |
|--------|-------------|
| `path` | Absolute path to the image |
| `subject` | Subject identifier |
| `label` | Stress label (0 or 1) |

This matches the standard `dataframe.csv` produced by this project. Other formats are not supported — adapt your manifest to this schema before running.

To run with shared embedding head instead of dual:
```bash
python src/train.py run=triplets run.triplets.mode=shared
```

### Custom Configuration

Override any parameter at runtime:

```bash
# Change model
python src/train.py model=mobilev3s

# Change hyperparameters
python src/train.py training.epochs=50 training.lr=0.001

# Change seed
python src/train.py seed=123

# Combine multiple overrides
python src/train.py model=effv2s seed=999 training.epochs=75
```

### Source Entry Points

- `src/train.py` is the main Hydra entry point and now supports both `run=default` and `run=triplets`.
- `src/engine/train.py` remains available for the modular baseline pipeline and explicitly guards against `run=triplets`.
- Baseline and triplet experiments use different input sources:
  - Baseline (`run=default`) reads `configs/data/dataloader.yaml` (`train_csv`, `val_csv`, semicolon-separated files).
  - Triplets (`run=triplets`) reads `run.triplets.manifest_path` from `configs/run/triplets.yaml` (manifest-style CSV).

### Available Models

- `effb2` - EfficientNet-B2 (default)
- `effv2s` - EfficientNetV2-Small
- `mobilev3s` - MobileNetV3-Small

## Output

Training outputs are saved to:
```
outputs/{model_name}/{seed}/
```

Model checkpoints are saved as:
```
model_{model}_{seed}_{timestamp}.pth
```

## Project Structure

```
Vision-Module/
├── configs/           # Hydra configuration files
│   ├── data/         # Data loading configs
│   ├── model/        # Model architecture configs
│   ├── run/          # Pipeline selection configs
│   ├── training/     # Training hyperparameters
│   └── config.yaml   # Main config
├── src/
│   ├── data/         # Dataset and dataloader modules
│   ├── engine/       # Training and validation loops
│   ├── models/       # Model definitions
│   ├── utils/        # Utility functions
│   └── train.py      # Main training script
├── notebooks/        # Jupyter notebooks for experimentation
└── pyproject.toml    # Project dependencies
```

## Development

Development dependencies are included in the `dev` dependency group:

```bash
# Already installed with uv sync
# Includes: jupyter, matplotlib, pandas, pytest, pre-commit, black
```

### Code Formatting

The project uses Black with line length 79:
```bash
black src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Training Features

- **Early Stopping:** Stops training if validation loss doesn't improve for 5 epochs
- **Best Model Saving:** Automatically saves the model with the lowest validation loss
- **Progress Tracking:** Real-time progress bars with tqdm
- **Experiment Logging:** All metrics logged to Weights & Biases

## License

See repository for license information.

## Citation

If you use this code, please cite the repository:
```
https://github.com/271828D/Vision-Module
```
