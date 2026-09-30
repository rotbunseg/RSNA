"""Patient-level stratified K-fold split — the one CV harness the whole team uses.

Per the playbook: "never trust a single split for a dataset this
heterogeneous," and critically, split by PATIENT/GROUP, not by study or
image, so the same patient's data never appears in both train and
validation within a fold (that leakage would make CV scores look better
than the true leaderboard score).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsna_knee.utils.config import Config


def make_cv_splits(cfg: Config) -> pd.DataFrame:
    """Assign every training study to one of cfg.cv.n_folds folds.

    Returns a DataFrame with columns [StudyInstanceUID, fold] and also
    writes it to cfg.data.raw_root/cfg.cv.splits_csv (actually written next
    to the cache — see scripts/make_cv_splits.py) so every training script
    reads the SAME split rather than re-randomizing per run.

    Stratification key: for a multi-label problem, sklearn's StratifiedGroupKFold
    needs a single label per row. We use each study's TOTAL number of positive
    findings (0..12) as a stratification proxy — it keeps "easy" (few/no
    findings) and "hard" (many findings) studies balanced across folds, which
    is a reasonable proxy for keeping the harder-to-predict rare classes from
    landing disproportionately in one fold. It is not perfect per-label
    stratification (true multi-label stratified splitting is more involved);
    if a specific rare label ends up badly imbalanced across folds in your
    EDA, consider upgrading to `iterstrat`'s MultilabelStratifiedGroupKFold.
    """
    raw_root = Path(cfg.data.raw_root)
    df = pd.read_csv(raw_root / cfg.data.train_csv)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)

    label_cols = list(cfg.labels)
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"train.csv is missing expected label columns {missing}. "
            "Update `labels:` in configs/default.yaml to match the real column names."
        )

    if cfg.cv.group_column in df.columns:
        groups = df[cfg.cv.group_column].astype(str)
    else:
        print(
            f"[cv_split] WARNING: '{cfg.cv.group_column}' not found in train.csv — "
            "falling back to StudyInstanceUID as the grouping key. If the real data has "
            "multiple studies per patient, this UNDER-protects against leakage; find the "
            "real patient id column and update cv.group_column in the config."
        )
        groups = df["StudyInstanceUID"]

    strat_key = df[label_cols].fillna(0).sum(axis=1).clip(upper=6).astype(int)

    skf = StratifiedGroupKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    fold_assignment = np.full(len(df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(df, strat_key, groups)):
        fold_assignment[val_idx] = fold

    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"], "fold": fold_assignment})
    assert (out["fold"] >= 0).all(), "Every study must be assigned to a fold"
    return out


def load_cv_splits(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.cache_dir) / cfg.cv.splits_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/make_cv_splits.py` first "
            "(see README, 'Step 2: build the CV split')."
        )
    return pd.read_csv(path, dtype={"StudyInstanceUID": str})
