"""Run with: pytest tests/ -v
No GPU or competition data required — these are fast sanity checks that catch
basic mistakes (shape bugs, NaN handling) before you burn GPU quota on a real run.
"""
import numpy as np

from rsna_knee.utils.metrics import macro_auc


def test_macro_auc_perfect_separation():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=float)
    score, per_label = macro_auc(y_true, y_pred, label_names=["a", "b"])
    assert score == 1.0
    assert per_label["a"] == 1.0
    assert per_label["b"] == 1.0


def test_macro_auc_handles_nan_targets():
    y_true = np.array([[1, np.nan], [0, 1], [1, 0], [0, 1]], dtype=float)
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=float)
    score, per_label = macro_auc(y_true, y_pred, label_names=["a", "b"])
    assert not np.isnan(score)
    assert per_label["a"] == 1.0


def test_macro_auc_skips_undefined_label():
    # label "b" has only positives -> AUC undefined -> should be NaN, excluded from macro avg
    y_true = np.array([[1, 1], [0, 1], [1, 1], [0, 1]], dtype=float)
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=float)
    score, per_label = macro_auc(y_true, y_pred, label_names=["a", "b"])
    assert np.isnan(per_label["b"])
    assert score == per_label["a"]
