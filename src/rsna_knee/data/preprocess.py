"""Build the preprocessed image cache (Track C's week-1/2 deliverable).

Run once via scripts/build_cache.py. For every study, for up to
`max_series_per_study` series/views, samples `slices_per_series` slices,
resizes+normalizes them, and stacks them into one array saved as
`<cache_dir>/<StudyInstanceUID>.npy` of shape
    (n_series, slices_per_series, H, W)
plus a sidecar `<cache_dir>/<StudyInstanceUID>_meta.json` recording which
series (and, if available, which MRI sequence/plane) filled each slot — the
image encoder's multi-view attention pooling (see models/image_encoder.py)
reads that metadata to know what it's attending over.

This intentionally does the slow part (DICOM parsing, resizing) exactly once
per study, per the playbook's #1 rule ("preprocess once, iterate many
times") — every training run after this reads small .npy files, not DICOM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rsna_knee.data.dicom_utils import dicom_to_array, load_dicom_series, sample_slice_indices
from rsna_knee.utils.config import Config


def iter_series_dirs(study_dir: Path) -> list[Path]:
    """Return this study's series subfolders.

    ASSUMPTION: study_dir/<SeriesInstanceUID or sequence-name>/*.dcm.
    Track C: adjust this one function if the real layout differs (e.g. if
    series are distinguished only by a DICOM tag rather than by folder) —
    everything else in this file is layout-agnostic.
    """
    if not study_dir.exists():
        return []
    return sorted(p for p in study_dir.iterdir() if p.is_dir())


def build_cache_for_study(
    study_uid: str,
    study_dir: Path,
    out_dir: Path,
    image_size: int,
    slices_per_series: int,
    max_series: int,
    overwrite: bool = False,
) -> bool:
    """Process one study. Returns True on success, False if skipped/failed."""
    out_path = out_dir / f"{study_uid}.npy"
    meta_path = out_dir / f"{study_uid}_meta.json"
    if out_path.exists() and meta_path.exists() and not overwrite:
        return True

    series_dirs = iter_series_dirs(study_dir)[:max_series]
    if not series_dirs:
        print(f"[build_cache] WARNING: no series found for {study_uid} at {study_dir}")
        return False

    series_stack = []
    series_meta = []
    for series_dir in series_dirs:
        try:
            slices = load_dicom_series(series_dir)
            if not slices:
                continue
            idxs = sample_slice_indices(len(slices), slices_per_series)
            arrays = [dicom_to_array(slices[i], image_size) for i in idxs]
            series_stack.append(np.stack(arrays, axis=0))  # (slices_per_series, H, W)

            first = slices[0]
            series_meta.append({
                "series_dir": series_dir.name,
                "sequence": str(getattr(first, "SeriesDescription", "unknown")),
                "n_raw_slices": len(slices),
            })
        except Exception as exc:  # noqa: BLE001 - a single bad series must not kill the whole build
            print(f"[build_cache] WARNING: failed on series {series_dir} for {study_uid}: {exc}")
            continue

    if not series_stack:
        print(f"[build_cache] WARNING: every series failed for {study_uid}")
        return False

    # Pad/truncate to a fixed number of series so every cached array has the
    # same shape (simplifies batching later). Missing "view" slots are zero-
    # filled and flagged as `is_padding` in the sidecar metadata so the model
    # can mask them out instead of learning from blank images.
    n_series = len(series_stack)
    h = w = image_size
    fixed = np.zeros((max_series, slices_per_series, h, w), dtype=np.float32)
    for i, arr in enumerate(series_stack):
        fixed[i] = arr
    for i in range(n_series, max_series):
        series_meta.append({"series_dir": None, "sequence": "pad", "n_raw_slices": 0, "is_padding": True})
    for m in series_meta[:n_series]:
        m["is_padding"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fixed)
    with open(meta_path, "w") as f:
        json.dump({"study_uid": study_uid, "series": series_meta, "n_real_series": n_series}, f)
    return True


def build_cache(cfg: Config, split: str = "train", limit: int | None = None) -> None:
    """Build the cache for every study in `train` or `test`.

    `limit`: process only the first N studies — use this for a fast smoke
    test (e.g. limit=20) before committing to a multi-hour full run.
    """
    raw_root = Path(cfg.data.raw_root)
    images_dir = raw_root / (cfg.data.train_images_dir if split == "train" else cfg.data.test_images_dir)
    out_dir = Path(cfg.data.cache_dir) / split

    if split == "train":
        import pandas as pd
        df = pd.read_csv(raw_root / cfg.data.train_csv)
    else:
        import pandas as pd
        df = pd.read_csv(raw_root / cfg.data.sample_submission_csv)
    study_uids = df["StudyInstanceUID"].astype(str).unique().tolist()
    if limit:
        study_uids = study_uids[:limit]

    n_ok, n_fail = 0, 0
    for study_uid in tqdm(study_uids, desc=f"Building {split} cache"):
        ok = build_cache_for_study(
            study_uid=study_uid,
            study_dir=images_dir / study_uid,
            out_dir=out_dir,
            image_size=cfg.data.cache_image_size,
            slices_per_series=cfg.data.slices_per_series,
            max_series=cfg.data.max_series_per_study,
        )
        n_ok += int(ok)
        n_fail += int(not ok)

    print(f"[build_cache] {split}: {n_ok} succeeded, {n_fail} failed/skipped, cache at {out_dir}")
    if n_fail > 0:
        print(
            "[build_cache] Some studies failed — this is almost always a data-layout "
            "assumption mismatch (see the ASSUMPTION notes in configs/default.yaml and "
            "iter_series_dirs() in this file). Inspect one failing study_dir by hand before "
            "running the full build."
        )
