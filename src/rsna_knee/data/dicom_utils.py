"""DICOM loading and slice-selection helpers.

Per the playbook's "preprocess once, iterate many times" rule, everything in
this file is meant to run ONCE per study inside scripts/build_cache.py — not
inside the training loop. Training reads the small cached .npy files that
build_cache.py produces (see preprocess.py), never raw DICOM directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom


def load_dicom_series(series_dir: str | Path) -> list[pydicom.FileDataset]:
    """Load every DICOM file in a series folder, sorted into anatomical order.

    ASSUMPTION: one folder per series, containing one .dcm file per slice.
    If the real data nests differently (e.g. all series flattened into one
    study folder with a SeriesInstanceUID DICOM tag to group by), adjust
    `iter_series_dirs` in preprocess.py rather than this function.
    """
    series_dir = Path(series_dir)
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        # Some DICOM exports omit the .dcm extension.
        paths = sorted(p for p in series_dir.iterdir() if p.is_file())
    slices = [pydicom.dcmread(str(p), force=True) for p in paths]
    slices = _sort_slices(slices)
    return slices


def _sort_slices(slices: list[pydicom.FileDataset]) -> list[pydicom.FileDataset]:
    """Order slices along the through-plane axis.

    Prefers ImagePositionPatient (projected onto the slice normal) when
    present, since InstanceNumber is occasionally unreliable in de-identified
    research exports; falls back to InstanceNumber, then filename order.
    """
    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            row, col = iop[:3], iop[3:]
            normal = np.cross(row, col)
            pos = np.array(ds.ImagePositionPatient, dtype=float)
            return float(np.dot(normal, pos))
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        return 0.0

    try:
        return sorted(slices, key=sort_key)
    except Exception:
        # Never let a malformed tag kill preprocessing for one study — fall
        # back to file order and let downstream QA flag it if it looks wrong.
        return slices


def dicom_to_array(ds: pydicom.FileDataset, target_size: int) -> np.ndarray:
    """Convert one DICOM slice to a normalized, resized float32 array in [0, 1].

    Applies the DICOM windowing (WindowCenter/WindowWidth) when present, which
    matters a lot for MRI — raw pixel intensities are not comparable across
    scanners/protocols without it.
    """
    import cv2  # local import: keep cv2 out of the hot path for callers that
                # only need pydicom (e.g. quick metadata scans in EDA)

    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        center = float(center[0] if isinstance(center, (list, pydicom.multival.MultiValue)) else center)
        width = float(width[0] if isinstance(width, (list, pydicom.multival.MultiValue)) else width)
        lo, hi = center - width / 2, center + width / 2
    else:
        # No windowing metadata: fall back to a robust percentile window so a
        # few bright/dark outlier pixels don't wash out the whole image.
        lo, hi = np.percentile(arr, [0.5, 99.5])

    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32)


def sample_slice_indices(n_slices: int, n_wanted: int) -> list[int]:
    """Pick `n_wanted` evenly-spaced slice indices out of `n_slices` available.

    Pads by repeating the last index if the series has fewer slices than
    `n_wanted` (short series happen — don't crash, just repeat).
    """
    if n_slices <= 0:
        return [0] * n_wanted
    if n_slices >= n_wanted:
        return list(np.linspace(0, n_slices - 1, n_wanted).round().astype(int))
    idx = list(range(n_slices))
    idx += [n_slices - 1] * (n_wanted - n_slices)
    return idx
