#!/usr/bin/env python
"""CLI: build the preprocessed image cache. Track C, week 1-2.

Examples:
    # Fast smoke test on 20 studies first — ALWAYS do this before a full run
    python scripts/build_cache.py --split train --limit 20

    # Full build
    python scripts/build_cache.py --split train
    python scripts/build_cache.py --split test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.data.preprocess import build_cache
from rsna_knee.utils.config import load_config, parse_cli_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N studies (smoke test)")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.set))
    build_cache(cfg, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
