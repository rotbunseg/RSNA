"""PyTorch Dataset classes.

Two datasets, matching the two training stages that need images:
  - KneeImageDataset      -> baseline model AND the distilled student (image only)
  - KneeMultimodalDataset -> the teacher (image + report text)

Both read from the cache built by data/preprocess.py, never from raw DICOM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_knee.utils.config import Config


def _augment(volume: np.ndarray, train: bool) -> np.ndarray:
    """Light, cheap augmentation applied per-study at train time.

    Kept deliberately simple (flips + small intensity jitter) — the playbook
    says earn complexity via measured CV improvement, and elaborate 3D-aware
    augmentation is exactly the kind of thing to add later ONLY if it moves
    the CV score. volume shape: (n_series, n_slices, H, W).
    """
    if not train:
        return volume
    if np.random.rand() < 0.5:
        volume = volume[:, :, :, ::-1]  # horizontal flip, applied per series/slice
    if np.random.rand() < 0.3:
        jitter = np.random.uniform(0.9, 1.1)
        volume = np.clip(volume * jitter, 0.0, 1.0)
    return np.ascontiguousarray(volume)


class KneeImageDataset(Dataset):
    """Image-only dataset for the baseline model and the distilled student.

    Each item's image tensor has shape (n_series, n_slices, H, W) — the model
    (see models/image_encoder.py) is responsible for pooling over slices and
    series with attention; this class just loads and (optionally) augments.
    """

    def __init__(self, cfg: Config, study_uids: list[str], labels_df: pd.DataFrame | None, split: str, train: bool):
        self.cfg = cfg
        self.study_uids = list(study_uids)
        self.labels_df = labels_df
        self.cache_dir = Path(cfg.data.cache_dir) / split
        self.train = train
        self.label_cols = list(cfg.labels)

        if labels_df is not None:
            self._label_lookup = labels_df.set_index("StudyInstanceUID")[self.label_cols]
        else:
            self._label_lookup = None

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx]
        arr_path = self.cache_dir / f"{study_uid}.npy"
        meta_path = self.cache_dir / f"{study_uid}_meta.json"

        volume = np.load(arr_path)  # (n_series, n_slices, H, W)
        with open(meta_path) as f:
            meta = json.load(f)
        view_mask = np.array([0.0 if s.get("is_padding") else 1.0 for s in meta["series"]], dtype=np.float32)

        volume = _augment(volume, self.train)
        image_tensor = torch.from_numpy(volume).float()
        view_mask_tensor = torch.from_numpy(view_mask).float()

        item = {
            "study_uid": study_uid,
            "image": image_tensor,          # (n_series, n_slices, H, W)
            "view_mask": view_mask_tensor,  # (n_series,) — 1 = real series, 0 = padding
        }

        if self._label_lookup is not None:
            row = self._label_lookup.loc[study_uid]
            # NaN allowed here (see utils/metrics.py) — some studies may not
            # report every finding; the loss function must handle NaN targets
            # (see training/engine.py: masked BCE).
            label = torch.tensor(row.values.astype(np.float32))
            item["label"] = label

        return item


class KneeMultimodalDataset(KneeImageDataset):
    """Image + radiology-report dataset, used ONLY to train the teacher.

    Adds tokenized/embedded report text. We embed with a frozen sentence
    encoder at __getitem__ time here for simplicity; if this turns out to be
    a training-speed bottleneck, precompute all report embeddings once
    (they don't change) and cache them the same way we cache images — see
    the README's "speeding up the teacher" note.
    """

    def __init__(
        self,
        cfg: Config,
        study_uids: list[str],
        labels_df: pd.DataFrame | None,
        reports_df: pd.DataFrame,
        split: str,
        train: bool,
    ):
        super().__init__(cfg, study_uids, labels_df, split, train)
        text_col = cfg.data.report_text_column
        self._report_lookup = reports_df.set_index("StudyInstanceUID")[text_col]

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        study_uid = item["study_uid"]
        report_text = self._report_lookup.get(study_uid, "")
        if not isinstance(report_text, str) or not report_text.strip():
            report_text = ""  # handled by the text encoder as an empty-report case
        item["report_text"] = report_text
        return item


def multimodal_collate(batch: list[dict]):
    """Custom collate: everything stacks normally except `report_text`, which
    stays a list of raw strings — the text encoder (a sentence-transformers
    model) tokenizes a whole batch of strings itself inside the training step.
    """
    out = {}
    for key in batch[0]:
        if key == "report_text":
            out[key] = [b[key] for b in batch]
        elif key == "study_uid":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
