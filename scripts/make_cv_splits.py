#!/usr/bin/env python
"""CLI: build the shared patient-level CV split. Run this ONCE and commit the
resulting cv_splits.csv (or re-run identically thanks to the fixed seed) so
every team member trains/validates on the exact same folds.

Usage:
    python scripts/make_cv_splits.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.data.cv_split import make_cv_splits
from rsna_knee.utils.config import load_config, parse_cli_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.set))
    splits = make_cv_splits(cfg)

    out_dir = Path(cfg.data.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.cv.splits_csv
    splits.to_csv(out_path, index=False)

    print(f"[make_cv_splits] wrote {out_path}")
    print(splits["fold"].value_counts().sort_index().rename("n_studies"))


if __name__ == "__main__":
    main()
