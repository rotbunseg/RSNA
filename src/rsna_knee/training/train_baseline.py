"""Stage 1: image-only baseline (Track A, week 2).

Purpose per the playbook: prove the full pipeline works end-to-end and get a
scored submission out FAST — not to be the best model. Run this before
anything fancier.

Usage:
    python -m rsna_knee.training.train_baseline --config configs/default.yaml --fold 0
    # or all folds:
    python -m rsna_knee.training.train_baseline --config configs/default.yaml --all-folds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rsna_knee.models.losses import masked_bce
from rsna_knee.models.student import StudentModel  # same image-only architecture as the baseline
from rsna_knee.training.engine import (
    EarlyStopper,
    build_image_loaders,
    collect_predictions_image,
    save_oof_predictions,
)
from rsna_knee.utils.config import load_config, parse_cli_overrides
from rsna_knee.utils.logging_utils import RunLogger
from rsna_knee.utils.metrics import macro_auc
from rsna_knee.utils.seed import set_seed


def train_one_fold(cfg, fold: int, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    train_loader, val_loader, pos_weight = build_image_loaders(cfg, fold)
    pos_weight = pos_weight.to(device)

    model = StudentModel(cfg, n_labels=len(cfg.labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
    stopper = EarlyStopper(cfg.train.early_stopping_patience)

    logger = RunLogger(cfg.output.log_dir, run_name=f"baseline_fold{fold}")
    ckpt_path = Path(cfg.output.checkpoint_dir) / f"baseline_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.train.epochs_baseline):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[baseline fold{fold} epoch{epoch}] train", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                out = model(image, view_mask)
                loss = masked_bce(out["logits"], labels, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        uids, y_true, y_pred = collect_predictions_image(model, val_loader, device)
        val_auc, per_label = macro_auc(y_true, y_pred, label_names=list(cfg.labels))

        logger.log_epoch({"fold": fold, "epoch": epoch, "train_loss": running_loss / len(train_loader), "val_macro_auc": val_auc})
        logger.print_epoch(f"baseline fold{fold}", epoch=epoch, train_loss=round(running_loss / len(train_loader), 4), val_macro_auc=round(val_auc, 4) if not np.isnan(val_auc) else "n/a")

        if stopper.step(val_auc):
            torch.save(model.state_dict(), ckpt_path)
        if stopper.should_stop:
            print(f"[baseline fold{fold}] early stopping at epoch {epoch} (best val_macro_auc={stopper.best:.4f})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return collect_predictions_image(model, val_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true", help="Train every fold and report the combined OOF score")
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides, e.g. train.batch_size=16")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.set))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[baseline] WARNING: no GPU found — this will be very slow. On Kaggle, set Accelerator=GPU.")

    folds = range(cfg.cv.n_folds) if args.all_folds else [args.fold]
    all_uids, all_true, all_pred = [], [], []
    for fold in folds:
        uids, y_true, y_pred = train_one_fold(cfg, fold, device)
        all_uids += uids
        all_true.append(y_true)
        all_pred.append(y_pred)

    if len(folds) > 1:
        save_oof_predictions(
            cfg, all_uids, np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0), run_name="baseline"
        )


if __name__ == "__main__":
    main()
