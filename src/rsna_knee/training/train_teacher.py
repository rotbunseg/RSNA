"""Stage 2: multimodal teacher (image + report). Track A + Track B, week 4.

Trains on BOTH the image and the radiology report, with two losses:
  - masked_bce on the 12 labels (the "real" task)
  - contrastive_loss between image and text embeddings (the "squeeze extra
    signal out of the report" trick from the kickoff plan)

This model is never shipped — see train_student.py for the distillation step
that produces the actual submission model.

Usage:
    python -m rsna_knee.training.train_teacher --config configs/default.yaml --fold 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rsna_knee.models.fusion import TeacherModel
from rsna_knee.models.losses import contrastive_loss, masked_bce
from rsna_knee.training.engine import EarlyStopper, build_multimodal_loaders
from rsna_knee.utils.config import load_config, parse_cli_overrides
from rsna_knee.utils.logging_utils import RunLogger
from rsna_knee.utils.metrics import macro_auc
from rsna_knee.utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    uids, y_true, y_pred = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        out = model(image, view_mask, batch["report_text"])
        y_pred.append(torch.sigmoid(out["logits"]).cpu().numpy())
        uids.extend(batch["study_uid"])
        if "label" in batch:
            y_true.append(batch["label"].numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0) if y_true else np.full_like(y_pred, np.nan)
    return uids, y_true, y_pred


def train_one_fold(cfg, fold: int, device: torch.device):
    train_loader, val_loader, pos_weight = build_multimodal_loaders(cfg, fold)
    pos_weight = pos_weight.to(device)

    model = TeacherModel(cfg, n_labels=len(cfg.labels)).to(device)
    # The text encoder is frozen by default (see models/text_encoder.py) —
    # only its projection head + everything else in the teacher is trainable.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
    stopper = EarlyStopper(cfg.train.early_stopping_patience)

    logger = RunLogger(cfg.output.log_dir, run_name=f"teacher_fold{fold}")
    ckpt_path = Path(cfg.output.checkpoint_dir) / f"teacher_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.train.epochs_teacher):
        model.train()
        running_bce, running_contrastive = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"[teacher fold{fold} epoch{epoch}] train", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                out = model(image, view_mask, batch["report_text"])
                bce = masked_bce(out["logits"], labels, pos_weight=pos_weight)
                cst = contrastive_loss(out["image_embedding"], out["text_embedding"])
                loss = bce + cfg.train.contrastive_weight * cst
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_bce += bce.item()
            running_contrastive += cst.item()

        uids, y_true, y_pred = evaluate(model, val_loader, device)
        val_auc, _ = macro_auc(y_true, y_pred, label_names=list(cfg.labels))

        logger.log_epoch({
            "fold": fold, "epoch": epoch,
            "train_bce": running_bce / len(train_loader),
            "train_contrastive": running_contrastive / len(train_loader),
            "val_macro_auc": val_auc,
        })
        logger.print_epoch(
            f"teacher fold{fold}", epoch=epoch,
            bce=round(running_bce / len(train_loader), 4),
            contrastive=round(running_contrastive / len(train_loader), 4),
            val_macro_auc=round(val_auc, 4) if not np.isnan(val_auc) else "n/a",
        )

        if stopper.step(val_auc):
            torch.save(model.state_dict(), ckpt_path)
        if stopper.should_stop:
            print(f"[teacher fold{fold}] early stopping at epoch {epoch} (best val_macro_auc={stopper.best:.4f})")
            break

    print(f"[teacher fold{fold}] done. Best checkpoint: {ckpt_path}")
    print("Next: python -m rsna_knee.training.train_student --config configs/default.yaml --fold", fold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.set))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = range(cfg.cv.n_folds) if args.all_folds else [args.fold]
    for fold in folds:
        train_one_fold(cfg, fold, device)


if __name__ == "__main__":
    main()
