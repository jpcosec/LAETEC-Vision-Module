import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch as th
from tqdm import tqdm
import wandb
from datetime import datetime
import random

from src.data.dataloader import get_data_loaders_from_split_files
from src.engine.triplet_train import train_triplet_experiment
from src.models.model import PretrainedModel


def train_classification_experiment(cfg: DictConfig) -> None:
    th.manual_seed(cfg.seed)
    th.cuda.manual_seed(cfg.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    random.seed(cfg.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    start_time = datetime.now()
    print(f"Starting training at: {start_time.strftime('%Y-%m-%d_%H:%M:%S')}")

    train_loader, val_loader = get_data_loaders_from_split_files(
        train_csv=cfg.data.train_csv,
        val_csv=cfg.data.val_csv,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = PretrainedModel(
        cfg.model.model,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    model.to(device)

    optimizer = th.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.99, 0.999),
        eps=1e-8,
    )
    criterion = th.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience = 5
    trigger_times = 0
    min_delta = 0.01

    output_dir = HydraConfig.get().runtime.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{output_dir}/model_{cfg.model.model}_{cfg.seed}_{timestamp}.pth"

    safe_cfg = OmegaConf.to_container(cfg, resolve=True)
    run = None
    if "run" in cfg and "wandb" in cfg.run and cfg.run.wandb.enabled:
        run = wandb.init(project=cfg.run.wandb.project, config=safe_cfg)

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with th.no_grad():
            progress_bar_val = tqdm(
                val_loader,
                desc="Validating",
                total=len(val_loader),
            )
            for images, labels in progress_bar_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                progress_bar_val.set_postfix(val_loss=loss.item())

        val_loss /= len(val_loader)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            th.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model at epoch {epoch + 1}: {checkpoint_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if run is not None:
            wandb.log(
                {
                    "train_loss": train_loss / len(train_loader),
                    "val_loss": val_loss,
                }
            )
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

    end_time = datetime.now()
    print(f"Finished training at: {end_time.strftime('%Y-%m-%d_%H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")

    if run is not None:
        wandb.finish()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pipeline = "classification"
    if "run" in cfg and "pipeline" in cfg.run:
        pipeline = cfg.run.pipeline

    if pipeline == "triplets":
        train_triplet_experiment(cfg)
        return

    train_classification_experiment(cfg)


if __name__ == "__main__":
    main()
