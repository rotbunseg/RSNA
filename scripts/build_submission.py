#!/usr/bin/env python
"""CLI wrapper around rsna_knee.inference.predict — this is the script the
actual Kaggle submission notebook's final cell should call (or copy the
handful of lines from rsna_knee/inference/predict.py directly into a notebook
cell — see README, 'Submitting to Kaggle').

Example:
    python scripts/build_submission.py \\
        --checkpoints checkpoints/student_fold0.pt checkpoints/student_fold1.pt \\
        --out /kaggle/working/submission.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.inference.predict import main

if __name__ == "__main__":
    main()
