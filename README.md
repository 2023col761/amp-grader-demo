# AMP Challenge — grading demo

This repository is a runnable copy of the real grading pipeline, so you can test your submission
locally — clone, install, train, generate, verify, score — before you submit anything for real.
For the full assignment description (problem statement, submission phases, sequence constraints,
rubric, academic integrity policy), see the assignment document provided separately.

**What's real here:** every script (`scripts/grade_submission.py`, `scorer/scorer.py`) is the
actual code that will run against your submission. The sequence-validity checks, the
train/generate split, the Phase-3 checkpoint-deletion enforcement, the reproducibility rerun, and
the exact `seqme` metrics computed are all identical to real grading.

**What's a placeholder here:**
- `scorer/scorer.py`'s `PROFILES["grader"]` is set equal to `PROFILES["student"]`. Real grading
  uses different, undisclosed models.
- `--held-out-fasta` defaults to the public `data/training/training.fasta` here, since this demo
  has no access to the real grader's secret held-out reference set (used for
  `FBD (AMPs)`/`Conformity score`). Everything else about how those metrics are computed is
  identical — only the reference data behind them differs.

Data-curation scripts (how `training.fasta`/`background.fasta` were built from the raw MarLys
export) aren't included here — they're not part of how grading works, just how the data was
prepared once.

## Setup

```bash
uv sync
```

## Starting your own submission

Create your project as a **sibling** of this directory (not inside it) — that keeps your
submission's own dependencies (whatever your model needs) separate from `scorer.py`'s
dependencies (`seqme`, heavy and irrelevant to grading correctness):

```bash
uv init --package my-model
cd my-model
```

In `pyproject.toml`, add two entry points:

```toml
[project.scripts]
train = "my_model.train:main"
generate = "my_model.generate:main"
```

Implement `train.py` so `uv run train` reads `data/training/training.fasta` and writes whatever
your model needs into `checkpoint/`. Implement `generate.py` so `uv run generate` loads
`checkpoint/`, produces exactly 1,000 valid unique AMP candidates, and writes them to
`generate/library.fasta`, filtering against `data/antibacterial.fasta` as you go.

Whether you commit `checkpoint/` depends on which phase you're submitting (see the assignment
document): pretrained weights for phases 1-2, nothing (add it to `.gitignore`) for phase 3, where
the grader trains it from scratch itself.

Copy `../amp-grader-demo/data/antibacterial.fasta` and
`../amp-grader-demo/data/training/training.fasta` into your project at
`data/antibacterial.fasta` and `data/training/training.fasta` for local development, and add
`data/` to `.gitignore` — the grader overwrites both files with its own copies at grading time, so
your submission should never commit them.

## Two ways to grade — matching the two ways *you'll* submit

In phases 1-2 you submit **pretrained weights** (`checkpoint/` committed to your repo) and the
grader never retrains your model. In phase 3 you submit **training code only** — the grader
deletes any `checkpoint/` you committed and trains from scratch on its own copy of
`training.fasta`, so nobody can smuggle in extra data or a shortcut checkpoint. Both modes are
available here; swap `https://github.com/2023col761/dummy-submission.git` below for your own repo
once you're ready to test it for real.

**Phase 3 style — train from scratch:**

```bash
uv run python scripts/grade_submission.py all \
  https://github.com/2023col761/dummy-submission.git \
  --scorer-profile grader \
  --report-out reports/mine.json
```

**Phase 1-2 style — use the submission's committed `checkpoint/` as-is:**

```bash
uv run python scripts/grade_submission.py all \
  https://github.com/2023col761/dummy-submission.git \
  --skip-train \
  --scorer-profile grader \
  --report-out reports/mine.json
```

(`--training-fasta`, `--antibacterial-fasta`, `--generic-fasta`, and `--held-out-fasta` all have
sensible defaults pointing at `data/` in this repo — you only need to override them if you've
moved files around.)

## Running stages independently

You don't have to redo the clone or retrain every time you want to check one thing — `setup`,
`train`, `generate`, and `score` are independent subcommands that all operate on the same `--dir`
(defaults to `submission/`):

```bash
uv run python scripts/grade_submission.py setup https://github.com/<you>/<your-submission>.git
uv run python scripts/grade_submission.py train        # deletes checkpoint/, retrains (skip for phases 1-2)
uv run python scripts/grade_submission.py generate      # verify + reproducibility check
uv run python scripts/grade_submission.py score --scorer-profile grader --report-out reports/mine.json
```

`setup` clones your repo and stages this demo's copies of `training.fasta` and
`antibacterial.fasta` over whatever you committed (if anything) — same mechanism the real grader
uses. If this passes here, the mechanical part of real grading will pass too — only the specific
classifier/embedder values and the held-out reference set differ (see above).
