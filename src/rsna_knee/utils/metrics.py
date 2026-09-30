"""The competition metric: macro-averaged AUC-ROC across all 12 labels.

This is the ONE function every training/eval loop calls to decide "did this
change actually help" — per the team playbook, never trust anything else
(accuracy, loss curves alone, a single label's AUC) to make that call.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str] | None = None):
    """Compute per-label AUC and the macro average, the way the competition scores it.

    Args:
        y_true: (N, 12) array of 0/1 ground-truth labels. May contain NaN for
            missing labels (some studies may not report every finding) — rows
            with NaN for a given label are dropped from that label's AUC only.
        y_pred: (N, 12) array of predicted probabilities in [0, 1].
        label_names: optional list of 12 label names, for the per-label dict.

    Returns:
        (macro_auc: float, per_label_auc: dict[str, float])
        A label is skipped (excluded from the macro average, and reported as
        NaN) if it has fewer than 2 positive or 2 negative examples in y_true
        — AUC is undefined in that case. This matters most in tiny debug runs
        and early folds for rare classes like Fracture; watch for NaNs there.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError("label_names length must match number of label columns")

    per_label: dict[str, float] = {}
    scores = []
    for i, name in enumerate(label_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        mask = ~np.isnan(col_true)
        col_true, col_pred = col_true[mask], col_pred[mask]

        n_pos = int((col_true == 1).sum())
        n_neg = int((col_true == 0).sum())
        if n_pos < 2 or n_neg < 2:
            per_label[name] = float("nan")
            continue

        auc = roc_auc_score(col_true, col_pred)
        per_label[name] = float(auc)
        scores.append(auc)

    macro = float(np.mean(scores)) if scores else float("nan")
    return macro, per_label


def format_per_label(per_label: dict[str, float]) -> str:
    """Pretty one-line string for logging, e.g. for tqdm postfix / console."""
    parts = [f"{name}={score:.3f}" if not np.isnan(score) else f"{name}=n/a" for name, score in per_label.items()]
    return " | ".join(parts)
