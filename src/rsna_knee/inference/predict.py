"""Generate submission.csv from one or more trained student checkpoints.

This is the ONLY script that needs to run inside the actual Kaggle
submission notebook — everything else (preprocessing, CV split, all three
training stages) happens beforehand, offline, with internet and time to
spare. This script must:
  - never touch the network (no downloads — all weights come from an
    attached Kaggle Dataset/Model, per the competition's code-competition rules)
  - finish well within the 9-hour runtime cap
  - only import the image-only StudentModel — never TeacherModel / TextEncoder,
    which would needlessly pull in the multilingual text stack at inference

If multiple fold checkpoints are given, their predictions are averaged
(a simple, cheap ensemble across folds) — see README for the tradeoff against
the Efficiency Prize's runtime penalty before adding more than a couple.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna_knee.data.dataset import KneeImageDataset
from rsna_knee.models.student import StudentModel
from rsna_knee.utils.config import load_config, parse_cli_overrides


@torch.no_grad()
def predict_with_checkpoint(cfg, ckpt_path: Path, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray]:
    model = StudentModel(cfg, n_labels=len(cfg.labels)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    uids, preds = [], []
    for batch in tqdm(loader, desc=f"predict [{ckpt_path.name}]"):
        image = batch["image"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        out = model(image, view_mask)
        preds.append(torch.sigmoid(out["logits"]).cpu().numpy())
        uids.extend(batch["study_uid"])
    return uids, np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="One or more student_fold*.pt paths to ensemble by averaging.",
    )
    parser.add_argument("--split", default="test", choices=["test", "train"], help="'train' is useful for a local smoke test")
    parser.add_argument("--out", default=None, help="Overrides cfg.output.submission_path")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_root = Path(cfg.data.raw_root)
    sub_df = pd.read_csv(raw_root / cfg.data.sample_submission_csv)
    study_uids = sub_df["StudyInstanceUID"].astype(str).tolist()

    dataset = KneeImageDataset(cfg, study_uids, labels_df=None, split=args.split, train=False)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    all_preds = []
    uid_order = None
    for ckpt in args.checkpoints:
        uids, preds = predict_with_checkpoint(cfg, Path(ckpt), loader, device)
        if uid_order is None:
            uid_order = uids
        elif uids != uid_order:
            raise RuntimeError("Checkpoint prediction order mismatch — this shouldn't happen with shuffle=False.")
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)

    out_df = pd.DataFrame(avg_preds, columns=list(cfg.labels))
    out_df.insert(0, "StudyInstanceUID", uid_order)

    # Preserve the sample_submission's row order/columns exactly — Kaggle is
    # strict about this.
    out_df = sub_df[["StudyInstanceUID"]].merge(out_df, on="StudyInstanceUID", how="left")
    out_df = out_df[["StudyInstanceUID"] + list(cfg.labels)]
    if out_df[list(cfg.labels)].isna().any().any():
        n_missing = int(out_df[list(cfg.labels)].isna().any(axis=1).sum())
        print(f"[predict] WARNING: {n_missing} studies missing predictions — filling with 0.5 (neutral). "
              "This should not happen in a real run; investigate before submitting.")
        out_df = out_df.fillna(0.5)

    out_path = Path(args.out or cfg.output.submission_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[predict] wrote {out_path} ({len(out_df)} rows, {len(args.checkpoints)} checkpoint(s) averaged)")


if __name__ == "__main__":
    main()
