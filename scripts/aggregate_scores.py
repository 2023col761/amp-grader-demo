"""Compute the aggregate leaderboard score for a cohort of already-graded submissions.

This is code that grading uses to turn per-submission `seqme` metrics into
the leaderboard score described in the assignment document.

Metrics with no fixed upper bound (FKEA, Activity) need putting on a comparable scale against
the cohort. These are standardized using the cohort's median and MAD (median absolute deviation)
and squashed through a fixed-scale sigmoid into (0,1).

Other metrics are already bounded to (0,1]/[0,1] after direction-fixing and is used as-is (just
floored at `epsilon`, so a component landing at exactly 0 can't zero out the whole geometric mean);
rescaling an already-bounded metric to the cohort would make it cohort-relative for no reason.

This requires seeing every submission in a phase at once, so it can only run after every submission
in that phase has been individually graded by grade_submission.py -- a separate, later step, not
part of grading a single submission. You won't have access to the cohort, so this won't
produce a meaningful score for you locally -- but if you want to see the mechanics run, you can
point --reports-dir at a folder of several of your own `scorer.py --out ...` runs (e.g. across a
few candidate models/checkpoints) to watch it standardize and combine them.

Usage:
  uv run python scripts/aggregate_scores.py --reports-dir reports/phase3 \
      --out reports/phase3/leaderboard.json
"""

import argparse
import json
import math
from pathlib import Path

CONSTRAINT_METRICS = {"Count", "Uniqueness"}
FBD_GENERIC = "FBD (generic)"
FBD_AMPS = "FBD (AMPs)"
FBD_MARGIN = "FBD margin"
UNBOUNDED_METRICS = {"FKEA", "Activity"}
ROBUST_SCALE = 1.0


def load_submissions(reports_dir: Path) -> dict[str, dict]:
    submissions = {}
    for path in sorted(reports_dir.glob("*.metrics.json")):
        report = json.loads(path.read_text())
        submissions[path.name.removesuffix(".metrics.json")] = report
    if not submissions:
        raise ValueError(f"No '*.metrics.json' files found in {reports_dir}.")
    return submissions


def check_consistent_objectives(submissions: dict[str, dict]) -> dict[str, str]:
    objectives = None
    for submission_id, report in submissions.items():
        if objectives is None:
            objectives = report["objective"]
        elif report["objective"] != objectives:
            raise ValueError(f"Objective map for '{submission_id}' differs from the rest of the cohort.")
    return objectives


def extract_raw_values(submissions: dict[str, dict]) -> dict[str, dict[str, float]]:
    raw = {}
    for submission_id, report in submissions.items():
        raw[submission_id] = {metric: cell["value"] for metric, cell in report["values"].items()}
    return raw


def add_fbd_margin(raw: dict[str, dict[str, float]], objectives: dict[str, str]) -> None:
    for values in raw.values():
        generic, amps = values[FBD_GENERIC], values[FBD_AMPS]
        values[FBD_MARGIN] = generic / (generic + amps)
    objectives[FBD_MARGIN] = "maximize"


def direction_fix(raw: dict[str, dict[str, float]], objectives: dict[str, str]) -> dict[str, dict[str, float]]:
    fixed = {}
    for submission_id, values in raw.items():
        fixed[submission_id] = {
            metric: (1.0 / (1.0 + value) if objectives[metric] == "minimize" else value)
            for metric, value in values.items()
        }
    return fixed


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def normalize_components(
    fixed: dict[str, dict[str, float]], components: list[str], epsilon: float
) -> dict[str, dict[str, float]]:
    """Bring every component onto a common, geometric-mean-safe scale.

    Unbounded components (FKEA, Activity) are standardized against the cohort's median/MAD and
    squashed through a sigmoid into (0,1) -- robust to a single unusually good/bad/degenerate
    submission in a small cohort, unlike min-max. Everything else is already bounded after
    direction-fixing (Diversity, Novelty, AuthPct, Conformity score, FBD margin -- all natively
    [0,1]-ish; FBD (generic)/FBD (AMPs) via the 1/(1+x) transform) and is left as raw, just floored
    at epsilon.
    """
    normalized = {submission_id: {} for submission_id in fixed}
    for metric in components:
        if metric in UNBOUNDED_METRICS:
            values = [fixed[submission_id][metric] for submission_id in fixed]
            med = median(values)
            mad = median([abs(v - med) for v in values])
            for submission_id in fixed:
                if mad == 0:
                    normalized[submission_id][metric] = 0.5
                else:
                    z = (fixed[submission_id][metric] - med) / (ROBUST_SCALE * mad)
                    normalized[submission_id][metric] = max(sigmoid(z), epsilon)
        else:
            for submission_id in fixed:
                normalized[submission_id][metric] = max(fixed[submission_id][metric], epsilon)
    return normalized


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()

    submissions = load_submissions(args.reports_dir)
    objectives = check_consistent_objectives(submissions)
    raw = extract_raw_values(submissions)

    add_fbd_margin(raw, objectives)

    components = [metric for metric in objectives if metric not in CONSTRAINT_METRICS]

    fixed = direction_fix(raw, objectives)
    normalized = normalize_components(fixed, components, args.epsilon)

    leaderboard = {}
    for submission_id in submissions:
        leaderboard[submission_id] = {
            "aggregate_score": geometric_mean([normalized[submission_id][metric] for metric in components]),
            "components": normalized[submission_id],
            "raw": raw[submission_id],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(leaderboard, indent=2))
    print(f"Wrote {args.out} ({len(submissions)} submissions, {len(components)} components)")


if __name__ == "__main__":
    main()
