"""Stage 3: distill the teacher into the image-only student.

THIS is the model that becomes the submission notebook (see
src/rsna_knee/inference/predict.py). It trains on the same image+report
batches as the teacher (report text is needed to run the teacher forward
pass and get its soft targets), but the student itself only ever looks at
the image — report text is never fed to the student.

Requires a trained teacher checkpoint for the same fold (run train_teacher.py
first).

Usage:
    python -m rsna_knee.training.train_student --config configs/default.yaml --fold 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rsna_knee.models.fusion import TeacherModel
from rsna_knee.models.losses import distillation_loss, masked_bce
from rsna_knee.models.student import StudentModel
from rsna_knee.training.engine import EarlyStopper, build_multimodal_loaders, save_oof_predictions
from rsna_knee.utils.config import load_config, parse_cli_overrides
from rsna_knee.utils.logging_utils import RunLogger
from rsna_knee.utils.metrics import macro_auc
from rsna_knee.utils.seed import set_seed


@torch.no_grad()
def evaluate_student(model, loader, device):
    """Evaluates the student using ONLY the image — this is the honest,
    submission-realistic number, unlike anything that peeks at report text.
    """
    model.eval()
    uids, y_true, y_pred = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        out = model(image, view_mask)
        y_pred.append(torch.sigmoid(out["logits"]).cpu().numpy())
        uids.extend(batch["study_uid"])
        if "label" in batch:
            y_true.append(batch["label"].numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0) if y_true else np.full_like(y_pred, np.nan)
    return uids, y_true, y_pred


def load_teacher(cfg, fold: int, device: torch.device) -> TeacherModel:
    ckpt_path = Path(cfg.output.checkpoint_dir) / f"teacher_fold{fold}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No teacher checkpoint at {ckpt_path}. Run "
            f"`python -m rsna_knee.training.train_teacher --config {Path('configs/default.yaml')} --fold {fold}` first."
        )
    teacher = TeacherModel(cfg, n_labels=len(cfg.labels)).to(device)
    teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def train_one_fold(cfg, fold: int, device: torch.device):
    train_loader, val_loader, pos_weight = build_multimodal_loaders(cfg, fold)
    pos_weight = pos_weight.to(device)

    teacher = load_teacher(cfg, fold, device)
    student = StudentModel(cfg, n_labels=len(cfg.labels)).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
    stopper = EarlyStopper(cfg.train.early_stopping_patience)

    logger = RunLogger(cfg.output.log_dir, run_name=f"student_fold{fold}")
    ckpt_path = Path(cfg.output.checkpoint_dir) / f"student_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = cfg.train.distill_alpha
    temperature = cfg.train.distill_temperature

    for epoch in range(cfg.train.epochs_distill):
        student.train()
        running_kd, running_hard = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"[student fold{fold} epoch{epoch}] train", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                teacher_out = teacher(image, view_mask, batch["report_text"])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                student_out = student(image, view_mask)
                kd = distillation_loss(
                    student_out["logits"], teacher_out["logits"],
                    student_out["image_embedding"], teacher_out["image_embedding"],
                    temperature=temperature,
                )
                hard = masked_bce(student_out["logits"], labels, pos_weight=pos_weight)
                loss = alpha * kd + (1 - alpha) * hard
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_kd += kd.item()
            running_hard += hard.item()

        uids, y_true, y_pred = evaluate_student(student, val_loader, device)
        val_auc, _ = macro_auc(y_true, y_pred, label_names=list(cfg.labels))

        logger.log_epoch({
            "fold": fold, "epoch": epoch,
            "train_kd_loss": running_kd / len(train_loader),
            "train_hard_loss": running_hard / len(train_loader),
            "val_macro_auc": val_auc,
        })
        logger.print_epoch(
            f"student fold{fold}", epoch=epoch,
            kd=round(running_kd / len(train_loader), 4),
            hard=round(running_hard / len(train_loader), 4),
            val_macro_auc=round(val_auc, 4) if not np.isnan(val_auc) else "n/a",
        )

        if stopper.step(val_auc):
            torch.save(student.state_dict(), ckpt_path)
        if stopper.should_stop:
            print(f"[student fold{fold}] early stopping at epoch {epoch} (best val_macro_auc={stopper.best:.4f})")
            break

    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    return evaluate_student(student, val_loader, device)


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
    all_uids, all_true, all_pred = [], [], []
    for fold in folds:
        uids, y_true, y_pred = train_one_fold(cfg, fold, device)
        all_uids += uids
        all_true.append(y_true)
        all_pred.append(y_pred)

    if len(folds) > 1:
        save_oof_predictions(
            cfg, all_uids, np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0), run_name="student"
        )
    print("\nThis student checkpoint is what scripts/build_submission.py loads. "
          "Compare its OOF macro-AUC above against the baseline's — that comparison is your "
          "evidence the report-guided distillation approach is actually earning its complexity "
          "(per the playbook: 'complexity is earned, not assumed').")


if __name__ == "__main__":
    main()
