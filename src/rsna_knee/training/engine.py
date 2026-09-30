"""Shared training/eval plumbing used by all three train_*.py scripts.

Keeping this in one place is what lets train_baseline.py, train_teacher.py
and train_student.py stay short and readable — each one is really just
"which model, which dataset, which loss" wired into the same loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna_knee.data.cv_split import load_cv_splits
from rsna_knee.data.dataset import KneeImageDataset, KneeMultimodalDataset, multimodal_collate
from rsna_knee.models.losses import compute_pos_weight
from rsna_knee.utils.metrics import macro_auc


class EarlyStopper:
    """Stops training when val macro-AUC hasn't improved for `patience` epochs.

    Higher is better for AUC, so we track the max.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Returns True if this is a new best score."""
        if np.isnan(value):
            self.bad_epochs += 1
            return False
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def get_fold_study_uids(cfg, fold: int) -> tuple[list[str], list[str]]:
    splits = load_cv_splits(cfg)
    train_uids = splits.loc[splits.fold != fold, "StudyInstanceUID"].tolist()
    val_uids = splits.loc[splits.fold == fold, "StudyInstanceUID"].tolist()
    return train_uids, val_uids


def load_labels(cfg) -> pd.DataFrame:
    raw_root = Path(cfg.data.raw_root)
    df = pd.read_csv(raw_root / cfg.data.train_csv)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    return df


def load_reports(cfg) -> pd.DataFrame:
    raw_root = Path(cfg.data.raw_root)
    df = pd.read_csv(raw_root / cfg.data.reports_csv)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    return df


def build_image_loaders(cfg, fold: int) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    labels_df = load_labels(cfg)
    train_uids, val_uids = get_fold_study_uids(cfg, fold)

    train_ds = KneeImageDataset(cfg, train_uids, labels_df, split="train", train=True)
    val_ds = KneeImageDataset(cfg, val_uids, labels_df, split="train", train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )

    train_labels = labels_df.set_index("StudyInstanceUID").loc[train_uids, list(cfg.labels)]
    prevalence = torch.tensor(train_labels.fillna(0).mean().values, dtype=torch.float32)
    pos_weight = compute_pos_weight(prevalence)
    return train_loader, val_loader, pos_weight


def build_multimodal_loaders(cfg, fold: int) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    labels_df = load_labels(cfg)
    reports_df = load_reports(cfg)
    train_uids, val_uids = get_fold_study_uids(cfg, fold)

    train_ds = KneeMultimodalDataset(cfg, train_uids, labels_df, reports_df, split="train", train=True)
    val_ds = KneeMultimodalDataset(cfg, val_uids, labels_df, reports_df, split="train", train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
        drop_last=True, pin_memory=True, collate_fn=multimodal_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=True, collate_fn=multimodal_collate,
    )

    train_labels = labels_df.set_index("StudyInstanceUID").loc[train_uids, list(cfg.labels)]
    prevalence = torch.tensor(train_labels.fillna(0).mean().values, dtype=torch.float32)
    pos_weight = compute_pos_weight(prevalence)
    return train_loader, val_loader, pos_weight


@torch.no_grad()
def collect_predictions_image(model, loader, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Runs the image-only model (baseline or student) over a loader.
    Returns (study_uids, y_true (N,12) possibly all-NaN, y_pred (N,12))."""
    model.eval()
    uids, y_true, y_pred = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        out = model(image, view_mask)
        probs = torch.sigmoid(out["logits"]).cpu().numpy()
        y_pred.append(probs)
        uids.extend(batch["study_uid"])
        if "label" in batch:
            y_true.append(batch["label"].numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0) if y_true else np.full_like(y_pred, np.nan)
    return uids, y_true, y_pred


def save_oof_predictions(cfg, uids: list[str], y_true: np.ndarray, y_pred: np.ndarray, run_name: str) -> Path:
    """Save out-of-fold predictions to CSV — this is your real, leak-free
    estimate of leaderboard performance (see README's 'reading the OOF score')."""
    out_dir = Path(cfg.output.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(cfg.labels)
    df_pred = pd.DataFrame(y_pred, columns=[f"pred_{c}" for c in cols])
    df_true = pd.DataFrame(y_true, columns=[f"true_{c}" for c in cols])
    df = pd.concat([pd.DataFrame({"StudyInstanceUID": uids}), df_true, df_pred], axis=1)
    path = out_dir / f"{run_name}_oof.csv"
    df.to_csv(path, index=False)

    score, per_label = macro_auc(y_true, y_pred, label_names=cols)
    print(f"[{run_name}] OOF macro-AUC = {score:.4f}")
    for name, auc in per_label.items():
        print(f"    {name:<20s} {auc:.4f}" if not np.isnan(auc) else f"    {name:<20s} n/a")
    return path
