# RSNA Knee Abnormality Detection — Rotbun Data Labs

Team pipeline for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition. This implements the **Report-Guided Distillation**
approach from our kickoff plan: a "teacher" model trains on both the knee
MRI and its paired radiology report, then a lightweight "student" is
distilled from it to run on the image alone — so the submitted model never
depends on the reports' multilingual text at inference time, and stays fast
enough to be competitive for the Efficiency Prize.

If you haven't read the kickoff plan or sat through the kickoff meeting,
read that first — this README assumes you know the *why*; it only covers
the *how*.

## Before you touch any code: one important caveat

**The `data:` section of `configs/default.yaml` is our best guess at the
competition's file layout**, based on the public competition description —
we have not yet inspected the actual downloaded `train.csv` / DICOM folder
structure ourselves. Every place in the code that depends on an assumption
we haven't verified is marked with an `ASSUMPTION:` comment. Track C: your
very first task is to open the real data on Kaggle and fix these — search
the repo for `ASSUMPTION` to find every spot:

```bash
grep -rn "ASSUMPTION" configs/ src/
```

Everything downstream reads its file paths and column names from
`configs/default.yaml`, so in most cases fixing the config file alone fixes
the whole pipeline — you shouldn't need to touch the code itself unless the
folder layout (not just names) is structurally different from what's
described here.

## Architecture recap

```
TRAINING TIME                                    INFERENCE TIME (submission notebook)
MRI series ──► Image encoder ──┐
                                 ├─► Cross-attention ──► 12-label head   (never shipped)
Radiology report ─► Text encoder┘    fusion + contrastive align
        │
        └── knowledge distillation ──►  Student: image-only encoder ──► submission.csv
                                          (distilled, no text pipeline)
```

Three training stages, run in order:

1. **Baseline** (`train_baseline.py`) — image-only, no report text, no
   distillation. Exists purely to prove the pipeline works end-to-end and
   get a scored submission out fast. Uses the *same* architecture as the
   student (see below), so its numbers are a fair "before" comparison for
   the distillation step.
2. **Teacher** (`train_teacher.py`) — image + report, trained with a hard-
   label loss plus a CLIP-style contrastive loss aligning image and report
   embeddings. Never shipped.
3. **Student** (`train_student.py`) — distilled from the teacher: same
   image-only architecture as the baseline, but trained with a distillation
   loss (soft targets + feature matching from the teacher) in addition to
   its own hard-label loss. **This is the model that generates the
   submission.**

## Repository layout

```
configs/default.yaml         All settings — paths, columns, model, training hyperparameters
src/rsna_knee/
  data/
    dicom_utils.py           DICOM loading, windowing, slice ordering/sampling
    preprocess.py            Builds the cache (run via scripts/build_cache.py)
    cv_split.py               Patient-level stratified K-fold split
    dataset.py                 PyTorch Datasets (image-only, image+report)
  models/
    image_encoder.py          Backbone + multi-view/multi-slice attention pooling
    text_encoder.py            Frozen multilingual sentence encoder (teacher-only)
    fusion.py                    Cross-attention fusion + TeacherModel
    student.py                    StudentModel (image-only — this is what ships)
    losses.py                      masked BCE, contrastive loss, distillation loss
  training/
    engine.py                      Shared dataloader/eval/logging plumbing
    train_baseline.py               Stage 1
    train_teacher.py                 Stage 2
    train_student.py                  Stage 3
  inference/
    predict.py                        Loads student checkpoint(s) -> submission.csv
  utils/                                config loader, seeding, metrics, logging
scripts/                                thin CLI wrappers around the above
tests/                                  offline unit tests (no data/network needed)
```

## Setup

### On Kaggle (where training and the final submission actually run)

1. Push this repo to GitHub (see **Getting this onto GitHub** below).
2. In a Kaggle Notebook, attach the competition data (Add Data →
   competition), then clone and install the repo. During interactive
   development, internet access is on, so a plain git clone works:

   ```python
   !git clone https://github.com/<your-org>/<your-repo>.git
   %cd <your-repo>
   !pip install -q -e .
   ```
3. Set the notebook's Accelerator to GPU (Settings → Accelerator → GPU
   T4 x2 or P100).
4. **For the actual scored submission notebook**, internet must be off —
   see "Submitting to Kaggle" below for how to attach this repo as a Kaggle
   Dataset instead of git-cloning it.

### Locally (for editing code, running the offline test suite, or a CPU
smoke test with a tiny sample of data you've downloaded by hand)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -v          # should pass with no GPU and no competition data
```

## Running the pipeline, in order

All commands assume you're in the repo root with the package installed
(`pip install -e .`). Every script accepts `--set key.path=value` to
override any config value from the CLI without editing the YAML, e.g.
`--set train.batch_size=16`.

### Step 0 — fix the data assumptions (Track C, do this first)

Open the real `train.csv`, the real image folder, and the real reports
file/column on Kaggle, then update `configs/default.yaml`'s `data:` section
to match. See the caveat at the top of this README.

### Step 1 — build the preprocessed cache

```bash
# ALWAYS smoke-test on a handful of studies first — this is the step most
# likely to break against real data layout quirks, and a 20-study run takes
# seconds instead of hours.
python scripts/build_cache.py --split train --limit 20

# once that looks right:
python scripts/build_cache.py --split train
python scripts/build_cache.py --split test
```

Inspect the warnings it prints. A handful of failed studies is often fine
(corrupt/missing series); if most or all studies fail, it's almost always a
data-layout assumption mismatch — see `iter_series_dirs()` in
`src/rsna_knee/data/preprocess.py`.

### Step 2 — build the CV split (do this once; everyone uses the same file)

```bash
python scripts/make_cv_splits.py
```

Writes `cv_splits.csv` next to the cache. Commit this file's *presence* to
your process (not to git — it's in `.gitignore` since it depends on the
cache path — instead, everyone runs this command once locally/on Kaggle; the
fixed seed in the config means everyone gets the identical split).

### Step 3 — baseline (Track A/C, week 2 — get a submission out fast)

```bash
python -m rsna_knee.training.train_baseline --fold 0
# once fold 0 looks reasonable, train every fold:
python -m rsna_knee.training.train_baseline --all-folds
```

This prints a per-epoch val macro-AUC and, with `--all-folds`, a final
out-of-fold (OOF) macro-AUC — your first real, leak-free estimate of
leaderboard performance. Generate a submission from it immediately (see
Step 6) to prove the full loop works before doing anything fancier.

### Step 4 — teacher (Track A + B, week 4)

```bash
python -m rsna_knee.training.train_teacher --all-folds
```

Watch both `bce` (hard-label loss) and `contrastive` in the printed output —
if contrastive loss isn't decreasing at all, the image and text encoders
aren't learning to align, which is worth flagging before moving on to
distillation.

### Step 5 — distill into the student (Track A, week 5)

Requires a teacher checkpoint for each fold you want to distill (from Step 4).

```bash
python -m rsna_knee.training.train_student --all-folds
```

Compare the student's OOF macro-AUC against the baseline's from Step 3. That
comparison is your actual evidence the report-guided approach is worth its
complexity — per the team playbook, "complexity is earned, not assumed." If
the student doesn't beat the baseline, that's a real, useful result too:
it tells you to debug the teacher/distillation before trusting it further,
not to ship it anyway.

### Step 6 — generate submission.csv

```bash
python scripts/build_submission.py \
    --checkpoints checkpoints/student_fold0.pt checkpoints/student_fold1.pt \
    --out /kaggle/working/submission.csv
```

Passing multiple fold checkpoints averages their predictions (a cheap
ensemble). Every additional checkpoint adds inference time, which counts
against the Efficiency Prize — see the note in `inference/predict.py`.

## Submitting to Kaggle (the actual scored notebook)

The scored submission notebook has **no internet access**, so it cannot
`git clone`. Instead:

1. On Kaggle, create a new Dataset from this GitHub repo: **Add Data → GitHub**,
   paste your repo URL. This clones a snapshot into `/kaggle/input/<repo-name>`.
   Whenever you push new commits, come back and click **Update** on that
   Dataset — it does not stay in sync automatically.
2. Also attach your trained student checkpoints as a Kaggle Dataset (upload
   the `.pt` files from `checkpoints/`).
3. In the submission notebook (internet OFF, GPU ON), your final cell is
   essentially:

   ```python
   import sys
   sys.path.insert(0, "/kaggle/input/<repo-name>/src")
   from rsna_knee.inference.predict import main
   import sys as _sys
   _sys.argv = [
       "predict.py",
       "--config", "/kaggle/input/<repo-name>/configs/default.yaml",
       "--checkpoints",
       "/kaggle/input/<your-checkpoints-dataset>/student_fold0.pt",
       "/kaggle/input/<your-checkpoints-dataset>/student_fold1.pt",
       "--out", "/kaggle/working/submission.csv",
   ]
   main()
   ```
4. Confirm `submission.csv` is named exactly that and appears in
   `/kaggle/working/`, then hit Submit.
5. **Profile the runtime** of this notebook well before the deadline (Track
   C/D) — it must finish in under 9 hours, and its wall-clock time directly
   feeds the Efficiency Score.

## Getting this onto GitHub

```bash
cd rsna-knee-detection
git init
git add .
git commit -m "Initial pipeline: data, models, training stages, inference"
git branch -M main
git remote add origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main
```

From here, follow the branch-per-task workflow from the kickoff discussion —
branch off `main` for each piece of work (e.g. `co/fix-data-paths`,
`ab/multiview-pooling`), open a PR, merge back into `main`, delete the
branch.

## Speeding up the teacher (if it's too slow on Kaggle's free GPU quota)

The text encoder currently embeds report text on-the-fly inside
`KneeMultimodalDataset.__getitem__`. Since report text never changes between
epochs, if this becomes a bottleneck, precompute every report's embedding
once (a short offline script using `models/text_encoder.py`) and cache it
the same way images are cached, then have the dataset load the cached vector
instead of re-embedding text every epoch.

## Troubleshooting

- **`FileNotFoundError` in `build_cache.py`**: your `data.raw_root` /
  `train_images_dir` in the config doesn't match the real Kaggle path —
  print `Path(cfg.data.raw_root).glob("*")` in a notebook cell to see what's
  actually there.
- **`KeyError` on label columns in `cv_split.py`**: `configs/default.yaml`'s
  `labels:` list doesn't match `train.csv`'s real column names — open
  `train.csv` and fix the list (order matters — it must match
  `sample_submission.csv`).
- **All/most studies fail in `build_cache.py`**: almost always
  `iter_series_dirs()` in `src/rsna_knee/data/preprocess.py` not matching the
  real folder nesting — inspect one study folder by hand
  (`!ls /kaggle/input/.../train_images/<one_study_uid>`) and adjust.
- **CUDA out of memory**: lower `train.batch_size` or
  `data.slices_per_series` in the config (`--set train.batch_size=4`).
- **Teacher's contrastive loss stuck / not decreasing**: check that
  `report_text` isn't empty for most studies (a common symptom of the
  `reports_csv` / `report_text_column` config assumption being wrong).

## Status / what's implemented vs. what's still a TODO for the team

Implemented and tested (`pytest tests/ -v` passes offline, no data needed):
data pipeline, patient-level CV split, image encoder with multi-view
attention, text encoder wrapper, cross-attention fusion, all three loss
functions, all three training stages, submission generation.

Still needs the team, once real data is available:
- Verify/fix every `ASSUMPTION:` against the real Kaggle data (Track C, day 1).
- Run the Step 1–6 pipeline end-to-end on a small sample to catch anything
  this README's guesses got wrong.
- Tune hyperparameters in `configs/default.yaml` against real CV scores.
- Decide, from real OOF numbers, whether the teacher/distillation step is
  actually earning its complexity over the plain baseline.
- Runtime-profile the final submission notebook against the 9-hour cap.
