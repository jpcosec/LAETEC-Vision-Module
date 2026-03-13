from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm

from src.utils.utils import set_seed


class StressTripletDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = read_image(str(row[self.image_col])).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)

        subject_id = th.tensor(int(row["subject_id"]), dtype=th.long)
        stress = th.tensor(int(row["stress_label"]), dtype=th.long)
        return image, subject_id, stress, str(row[self.image_col])


class SubjectBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        df: pd.DataFrame,
        subjects_per_batch: int,
        images_per_subject: int,
        seed: int,
    ):
        self.df = df.reset_index(drop=True)
        self.subjects_per_batch = subjects_per_batch
        self.images_per_subject = images_per_subject
        self.batch_size = subjects_per_batch * images_per_subject
        self.rng = np.random.default_rng(seed)

        self.subject_to_indices: Dict[int, np.ndarray] = {}
        for subject_id, group in self.df.groupby("subject_id"):
            self.subject_to_indices[int(subject_id)] = group.index.to_numpy()

        self.subject_ids = np.array(sorted(self.subject_to_indices.keys()))
        self.num_batches = max(len(self.df) // self.batch_size, 1)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_subjects = self.rng.choice(
                self.subject_ids,
                size=self.subjects_per_batch,
                replace=len(self.subject_ids) < self.subjects_per_batch,
            )

            batch_indices: List[int] = []
            for subject_id in chosen_subjects:
                pool = self.subject_to_indices[int(subject_id)]
                sample_idx = self.rng.choice(
                    pool,
                    size=self.images_per_subject,
                    replace=len(pool) < self.images_per_subject,
                )
                batch_indices.extend(sample_idx.tolist())

            self.rng.shuffle(batch_indices)
            yield batch_indices


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = None
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        net = models.mobilenet_v3_small(weights=weights)
        self.features = net.features
        self.avgpool = net.avgpool
        self.out_dim = net.classifier[0].in_features

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return th.flatten(x, 1)


class SharedEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: th.Tensor) -> Dict[str, th.Tensor]:
        features = self.backbone(x)
        embedding = F.normalize(self.head(features), p=2, dim=1)
        return {"id_emb": embedding, "stress_emb": embedding}


class DualEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        self.id_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.stress_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: th.Tensor) -> Dict[str, th.Tensor]:
        features = self.backbone(x)
        id_emb = F.normalize(self.id_head(features), p=2, dim=1)
        stress_emb = F.normalize(self.stress_head(features), p=2, dim=1)
        return {"id_emb": id_emb, "stress_emb": stress_emb}


def load_triplet_manifest(triplet_cfg: DictConfig) -> pd.DataFrame:
    df = pd.read_csv(triplet_cfg.manifest_path, sep=triplet_cfg.manifest_sep)

    if triplet_cfg.label_found_col in df.columns:
        df = df[df[triplet_cfg.label_found_col] == triplet_cfg.label_found_value]

    required_cols = [
        triplet_cfg.image_col,
        triplet_cfg.subject_col,
        triplet_cfg.stress_col,
    ]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in manifest: {missing_str}")

    df = df.copy()
    df[triplet_cfg.image_col] = df[triplet_cfg.image_col].astype(str)
    df = df[df[triplet_cfg.image_col].map(lambda path: Path(path).exists())].copy()

    df["stress_label"] = df[triplet_cfg.stress_col].astype(int)
    df["subject_id"] = pd.Categorical(df[triplet_cfg.subject_col]).codes
    return df.reset_index(drop=True)


def split_by_subject(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["subject_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def build_triplet_loaders(
    triplet_cfg: DictConfig,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    manifest_df = load_triplet_manifest(triplet_cfg)
    train_df, val_df = split_by_subject(manifest_df, triplet_cfg.val_size, seed)

    train_tf = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = StressTripletDataset(train_df, triplet_cfg.image_col, train_tf)
    val_ds = StressTripletDataset(val_df, triplet_cfg.image_col, val_tf)

    train_sampler = SubjectBalancedBatchSampler(
        train_df,
        subjects_per_batch=triplet_cfg.subjects_per_batch,
        images_per_subject=triplet_cfg.images_per_subject,
        seed=seed,
    )
    val_batch_size = triplet_cfg.subjects_per_batch * triplet_cfg.images_per_subject

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=triplet_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=triplet_cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(triplet_cfg: DictConfig) -> nn.Module:
    if triplet_cfg.mode == "dual":
        return DualEmbeddingModel(
            embedding_dim=triplet_cfg.embedding_dim,
            pretrained=True,
        )
    if triplet_cfg.mode == "shared":
        return SharedEmbeddingModel(
            embedding_dim=triplet_cfg.embedding_dim,
            pretrained=True,
        )
    raise ValueError("run.triplets.mode must be either 'dual' or 'shared'.")


def pairwise_dist(embeddings: th.Tensor) -> th.Tensor:
    return th.cdist(embeddings, embeddings, p=2)


def build_id_masks(subject_ids: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    batch_size = subject_ids.shape[0]
    eye = th.eye(batch_size, dtype=th.bool, device=subject_ids.device)
    same_subject = subject_ids[:, None] == subject_ids[None, :]
    pos_mask = same_subject & ~eye
    neg_mask = ~same_subject & ~eye
    return pos_mask, neg_mask


def build_stress_masks(
    stress_labels: th.Tensor,
    subject_ids: th.Tensor,
    require_diff_subject_for_positive: bool,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    batch_size = stress_labels.shape[0]
    eye = th.eye(batch_size, dtype=th.bool, device=stress_labels.device)

    stressed = stress_labels == 1
    not_stressed = stress_labels == 0

    pos_mask = stressed[:, None] & stressed[None, :] & ~eye
    if require_diff_subject_for_positive:
        diff_subject = subject_ids[:, None] != subject_ids[None, :]
        pos_mask = pos_mask & diff_subject

    neg_mask = stressed[:, None] & not_stressed[None, :]
    anchor_mask = stressed
    return pos_mask, neg_mask, anchor_mask


def batch_hard_triplet_loss(
    embeddings: th.Tensor,
    pos_mask: th.Tensor,
    neg_mask: th.Tensor,
    margin: float,
    anchor_mask: th.Tensor | None = None,
) -> Tuple[th.Tensor, Dict[str, float]]:
    dist = pairwise_dist(embeddings)
    hard_pos = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values
    hard_neg = dist.masked_fill(~neg_mask, 1e9).min(dim=1).values

    valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
    if anchor_mask is not None:
        valid = valid & anchor_mask

    per_anchor = F.relu(hard_pos - hard_neg + margin)
    if valid.any():
        loss = per_anchor[valid].mean()
        stats = {
            "active_anchors": float(valid.float().mean().item()),
            "mean_hard_pos": float(hard_pos[valid].mean().item()),
            "mean_hard_neg": float(hard_neg[valid].mean().item()),
        }
    else:
        loss = embeddings.sum() * 0.0
        stats = {
            "active_anchors": 0.0,
            "mean_hard_pos": float("nan"),
            "mean_hard_neg": float("nan"),
        }
    return loss, stats


def run_triplet_epoch(
    model: nn.Module,
    loader: DataLoader,
    triplet_cfg: DictConfig,
    device: str,
    optimizer: th.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    meter = {
        "loss_total": 0.0,
        "loss_id": 0.0,
        "loss_stress": 0.0,
        "id_active": 0.0,
        "stress_active": 0.0,
    }
    step_count = 0

    context = th.enable_grad() if is_train else th.no_grad()
    with context:
        desc = "Training triplets" if is_train else "Validating triplets"
        for images, subject_ids, stress_labels, _ in tqdm(loader, desc=desc):
            images = images.to(device)
            subject_ids = subject_ids.to(device)
            stress_labels = stress_labels.to(device)

            outputs = model(images)
            id_emb = outputs["id_emb"]
            stress_emb = outputs["stress_emb"]

            pos_id, neg_id = build_id_masks(subject_ids)
            loss_id, stats_id = batch_hard_triplet_loss(
                id_emb,
                pos_mask=pos_id,
                neg_mask=neg_id,
                margin=triplet_cfg.margin_id,
            )

            pos_st, neg_st, anchor_st = build_stress_masks(
                stress_labels,
                subject_ids,
                triplet_cfg.require_diff_subject_for_positive,
            )
            loss_stress, stats_stress = batch_hard_triplet_loss(
                stress_emb,
                pos_mask=pos_st,
                neg_mask=neg_st,
                margin=triplet_cfg.margin_stress,
                anchor_mask=anchor_st,
            )

            loss_total = (
                triplet_cfg.weight_id * loss_id
                + triplet_cfg.weight_stress * loss_stress
            )

            if is_train:
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            meter["loss_total"] += float(loss_total.item())
            meter["loss_id"] += float(loss_id.item())
            meter["loss_stress"] += float(loss_stress.item())
            meter["id_active"] += stats_id["active_anchors"]
            meter["stress_active"] += stats_stress["active_anchors"]
            step_count += 1

    if step_count == 0:
        return meter

    for key in meter:
        meter[key] /= step_count
    return meter


def init_wandb(cfg: DictConfig):
    if not cfg.run.wandb.enabled:
        return None
    safe_cfg = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(project=cfg.run.wandb.project, config=safe_cfg)


def train_triplet_experiment(cfg: DictConfig) -> None:
    triplet_cfg = cfg.run.triplets
    set_seed(cfg.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    start_time = datetime.now()
    print(f"Starting training at: {start_time.strftime('%Y-%m-%d_%H:%M:%S')}")

    train_loader, val_loader = build_triplet_loaders(triplet_cfg, cfg.seed)
    model = build_model(triplet_cfg).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=triplet_cfg.lr)

    output_dir = HydraConfig.get().runtime.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (
        f"{output_dir}/{triplet_cfg.checkpoint_prefix}_"
        f"{triplet_cfg.mode}_{cfg.seed}_{timestamp}.pth"
    )

    run = init_wandb(cfg)
    history: List[Dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(triplet_cfg.epochs):
        train_stats = run_triplet_epoch(
            model,
            train_loader,
            triplet_cfg,
            device,
            optimizer,
        )
        val_stats = run_triplet_epoch(model, val_loader, triplet_cfg, device)

        metrics = {f"train_{k}": v for k, v in train_stats.items()}
        metrics.update({f"val_{k}": v for k, v in val_stats.items()})
        metrics["epoch"] = epoch + 1
        history.append(metrics)

        if run is not None:
            wandb.log(metrics)

        if val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]
            th.save(model.state_dict(), checkpoint_path)

        print(
            f"epoch={epoch + 1} "
            f"train_total={train_stats['loss_total']:.4f} "
            f"val_total={val_stats['loss_total']:.4f} "
            f"train_id_active={train_stats['id_active']:.3f} "
            f"train_stress_active={train_stats['stress_active']:.3f}"
        )

    history_path = f"{output_dir}/triplet_history_{triplet_cfg.mode}_{cfg.seed}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved best triplet model to: {checkpoint_path}")
    print(f"Saved triplet training history to: {history_path}")

    if run is not None:
        wandb.finish()

    end_time = datetime.now()
    print(f"Finished training at: {end_time.strftime('%Y-%m-%d_%H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")
