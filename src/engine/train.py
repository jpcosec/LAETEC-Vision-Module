import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch as th
import wandb
from datetime import datetime

from src.data.dataloader import (
    get_data_loaders,
    get_data_loaders_from_split_files,
)
from src.models.model import PretrainedModel
from src.utils.utils import (
    build_optimizer,
    train_epoch,
    validate,
    save_best_model,
    set_seed,
)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if "run" in cfg and cfg.run.pipeline == "triplets":
        raise ValueError(
            "Triplet pipeline is supported via src/train.py only. "
            "Use: python src/train.py run=triplets"
        )

    # Setup
    set_seed(cfg.seed)
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    start_time = datetime.now()
    print(f"Starting training at: {start_time.strftime('%Y-%m-%d_%H:%M:%S')}")

    # Data
    if "train_csv" in cfg.data and "val_csv" in cfg.data:
        train_loader, val_loader = get_data_loaders_from_split_files(
            train_csv=cfg.data.train_csv,
            val_csv=cfg.data.val_csv,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
    else:
        train_loader, val_loader = get_data_loaders(**cfg.data)

    # Model
    model = PretrainedModel(
        cfg.model.model,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = th.nn.BCEWithLogitsLoss()

    # Training setup
    best_val_loss = float("inf")
    output_dir = HydraConfig.get().runtime.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{output_dir}/model_{cfg.model.model}_{cfg.seed}_{timestamp}.pth"

    # Wandb
    safe_cfg = OmegaConf.to_container(cfg, resolve=True)
    project = "stress-project"
    if "run" in cfg and "wandb" in cfg.run and "project" in cfg.run.wandb:
        project = cfg.run.wandb.project
    wandb.init(project=project, config=safe_cfg)

    # Training loop
    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Early stopping & saving
        best_val_loss = save_best_model(val_loss, best_val_loss, model, checkpoint_path)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

    wandb.finish()

    # End time
    end_time = datetime.now()
    print(f"Finished training at: {end_time.strftime('%Y-%m-%d_%H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")


if __name__ == "__main__":
    main()
