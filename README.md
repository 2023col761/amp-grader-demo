# AMP Challenge — grading demo

This repository is a runnable copy of the grading pipeline for you to test your submission locally — clone, install, train, generate, verify, score — before submission. For the full assignment description, see the assignment document provided on Moodle. This README walks you through the grading process of a dummy submission available at `https://github.com/2023col761/dummy-submission.git`

## Setup

```bash
uv sync
```

## Phase-wise grading

In phases 1-2 you submit pretrained weights (`checkpoint/` committed to your repo); the grader doesn't retrain your model. In phase 3 you submit training code only; the grader deletes any `checkpoint/` you committed and trains from scratch on its own copy of `training.fasta`.

Swap `https://github.com/2023col761/dummy-submission.git` below for your own repo once you're ready to test it.

**Phase 3 style — train from scratch:**

```bash
uv run python scripts/grade_submission.py all \
  https://github.com/2023col761/dummy-submission.git \
  --dir submission \
  --scorer-profile grader \
  --report-out reports/submission
```

**Phase 1-2 style — use the submission's committed `checkpoint/` as-is:**

```bash
uv run python scripts/grade_submission.py all \
  https://github.com/2023col761/dummy-submission.git \
  --dir submission \
  --skip-train \
  --scorer-profile grader \
  --report-out reports/submission
```

`--training-fasta`, `--antibacterial-fasta`, `--generic-fasta`, and `--held-out-fasta` all point at this repo's own `data/` folder (`amp-grader-demo/data/`) — that's the source. 

Your submission repo doesn't need a `data/` folder of its own committed to it: `setup` (part of `all`) creates one inside your submission's clone (`--dir submission` above, so `submission/data/...`), staging `training.fasta`/`antibacterial.fasta` from this repo's copies into it, overwriting anything you committed there. Your submission's own `train`/`generate` code should just read from that path — see "Starting your own submission" below.

## Running stages independently

You can skip cloning and retraining by using independent subcommands — `setup`, `train`, `generate`, and `score` — while operating on the same `--dir`:

```bash
uv run python scripts/grade_submission.py setup https://github.com/2023col761/dummy-submission.git --dir submission
uv run python scripts/grade_submission.py train --dir submission        # deletes checkpoint/, retrains (skip for phases 1-2)
uv run python scripts/grade_submission.py generate --dir submission      # verify + reproducibility check
uv run python scripts/grade_submission.py score --dir submission --scorer-profile grader --report-out reports/submission
```

## Grading
`scripts/grade_submission.py` generates the raw metrics for each submission; `scripts/aggregate_scores.py` aggregates the metrics into a cumulative score used for grading. Some of the metrics used for grading are unbounded. `scripts/aggregate_scores.py` normalizes those metrics against a cohort of submissions before taking their geometric mean (explaination in the assignment document). It needs multiple submissions' reports to normalize against, so running it against a single submission won't produce a meaningful score. If you want to see the mechanics run anyway, point `--reports-dir` at a folder containing multiple `scorer.py --out ...` runs (e.g. from a few candidate models or checkpoints you're comparing). It'll normalize and combine whatever's in there, which is enough to see exactly how your metrics turn into a score:

```bash
uv run python scripts/aggregate_scores.py --reports-dir reports --out reports/leaderboard.json
```
**Placeholders.** The scripts in this repo will be run as is against your submission. However, there are certain placeholders in the code that will be replaced at the time of graindg:
- `scorer/scorer.py`'s `PROFILES["grader"]` is set equal to `PROFILES["student"]`. Grading uses different, undisclosed models.
- `--held-out-fasta` defaults to the public `data/training/training.fasta` here, since this demo does not have access to the grader's held-out reference set (used for `FBD (AMPs)`/`Conformity score`). Everything else about how those metrics are computed is identical — only the reference data behind them differs.

## Starting your own submission

Emulate the dummy submission's structure. Do NOT create your project inside this grader. This grader shouldn't be part of your submission.

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

Implement `train.py` so `uv run train` reads `data/training/training.fasta` and writes whatever your model needs into `checkpoint/`. Implement `generate.py` so `uv run generate` loads `checkpoint/`, produces exactly 1,000 valid unique AMP candidates, and writes them to `generate/library.fasta`, filtering against `data/antibacterial.fasta` as you go. Both `data/...` paths here are relative to *your own submission repo* — as explained above, that folder doesn't exist in your repo yet; the grader creates and populates it for you at grading time (see the note above and `dummy-submission`'s `train.py`/`generate.py`, which read from these exact same paths).

Whether you commit `checkpoint/` depends on which phase you're submitting (see the assignment document): pretrained weights for phases 1-2, nothing (add it to `.gitignore`) for phase 3, where the grader trains it from scratch itself.

For local development, copy `antibacterial.fasta` and `training/training.fasta` into your project at `data/antibacterial.fasta` and `data/training/training.fasta`. Remember, you won't commit this as the grader does this step for you. So add `data/` to `.gitignore`.