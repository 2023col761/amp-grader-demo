"""Grade an AMP Challenge classroom submission.

DEMO VERSION. This is the exact orchestration code the real (private) grader runs — same
subcommands, same checks, same checkpoint-deletion enforcement. The one difference: `--held-out-
fasta` defaults to the public training set here (this demo has no access to the real grader's
secret held-out reference set). See scorer/scorer.py's module docstring for the matching note
about --profile grader. Everything else about how your submission gets graded is identical.

Adapted from amp-challenge-2027/scripts/verify_submission.py. Differences from the canonical
validator, all deliberate:

  - Two entry points instead of one: `train` (produces checkpoint/) then `generate` (loads
    checkpoint/, writes generate/library.fasta).
  - LIBRARY_SIZE = 1,000, not 50,000. No top.fasta/top-k/80%-Levenshtein logic at all — this
    assignment is generation only, ranking is out of scope.
  - The reproducibility rerun-and-byte-compare applies only to `generate` (cheap, must be
    deterministic under the fixed seed) — not to `train` (expensive, may be legitimately
    stochastic across runs; only the final decoding step needs to be reproducible).
  - Ends with a scoring stage (scorer.py) instead of stopping at validity checks.

Split into subcommands so a clone doesn't have to be redone (or a model retrained) on every
iteration, and so different assignment phases can skip `train` entirely:

  setup     clone + uv sync + stage training.fasta/antibacterial.fasta into --dir
  train     delete any existing checkpoint/ in --dir, then `uv run train` (from-scratch phases)
  generate  `uv run generate`, verify output, check reproducibility
  score     score an already-generated library.fasta
  all       setup -> train (unless --skip-train) -> generate -> score, in one call

Phases 1-2 of this assignment let students submit pretrained weights (checkpoint/ committed to
their repo) — for those, run `all --skip-train` (or just `setup` then `generate` then `score`),
which uses whatever checkpoint/ the submission already has instead of retraining it. Phase 3
requires training from scratch, so run `all` (or `setup` then `train` then `generate` then
`score`) without --skip-train — `train` unconditionally deletes checkpoint/ first, so a submission
can't get credit for a checkpoint it didn't actually produce from the grader-supplied data.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRAIN_ENTRY_POINT = "train"
GENERATE_ENTRY_POINT = "generate"

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
LIBRARY_SIZE = 1_000

TRAINING_DATA_REL_PATH = Path("data/training/training.fasta")
ANTIBACTERIAL_DATA_REL_PATH = Path("data/antibacterial.fasta")
CHECKPOINT_REL_PATH = Path("checkpoint")


def _read_fasta(path: Path) -> tuple[list[str], list[str]]:
    headers, sequences = [], []
    header, seq_parts = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(seq_parts))
            header, seq_parts = line[1:], []
        else:
            seq_parts.append(line.upper())
    if header is not None:
        headers.append(header)
        sequences.append("".join(seq_parts))
    return headers, sequences


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' is not installed or not found on PATH.")


def _clone_git_repository(
    repo_dir: Path,
    repo_url: str,
    branch: str | None = None,
    shallow: bool = True,
    git: str = "git",
):
    if repo_dir.exists():
        raise FileExistsError(f"'{repo_dir}' already exists.")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [git, "clone"]
    if branch:
        cmd += ["-b", branch, "--single-branch"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [repo_url.removeprefix("git+"), str(repo_dir)]

    try:
        subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed:\n{e.stderr}") from e


def _sync_uv(repo_dir: Path, extras: list[str], uv: str = "uv") -> None:
    extra_flags = [flag for extra in extras for flag in ("--extra", extra)]
    try:
        subprocess.run(
            [uv, "sync", *extra_flags],
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=repo_dir,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"uv sync failed:\n{e.stderr}") from e


def _stage_file(repo_dir: Path, src: Path, rel_path: Path) -> None:
    """Overwrite whatever the submission committed at rel_path with our own copy.

    Used for both training.fasta and antibacterial.fasta: a submission should never need to
    commit its own copy of either (see dummy-submission for the reference example) — the grader
    always supplies them fresh, both so "level the playing field" actually holds and so there's
    one less thing for a submission to get stale or wrong.
    """
    dst = repo_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _uv_run(repo_dir: Path, entry_point: str, uv: str = "uv", timeout: float | None = None) -> None:
    cmd = [uv, "run", "--no-sync", entry_point]
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    print(f"Running: {' '.join(cmd)}" + (f" (timeout={timeout}s)" if timeout else ""))
    try:
        subprocess.run(cmd, check=True, cwd=repo_dir, env=env, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"'{entry_point}' failed with exit code {e.returncode}.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"'{entry_point}' exceeded the {timeout}s timeout.") from e


def _verify_sequences(fasta_path: Path) -> set[str]:
    headers, sequences = _read_fasta(fasta_path)
    errors: list[str] = []

    if not sequences:
        errors.append("FASTA file is empty — no sequences found.")
    elif len(sequences) != LIBRARY_SIZE:
        errors.append(f"Expected {LIBRARY_SIZE} sequences, got {len(sequences)}.")

    seen: set[str] = set()
    for i, (header, seq) in enumerate(zip(headers, sequences), start=1):
        if not header.strip():
            errors.append(f"Record {i}: missing header.")
        if not seq:
            errors.append(f"Record {i} ('{header}'): empty sequence.")
            continue
        invalid = set(seq) - STANDARD_AMINO_ACIDS
        if invalid:
            errors.append(f"Record {i} ('{header}'): invalid characters {sorted(invalid)}.")
        if len(seq) < MIN_LENGTH:
            errors.append(f"Record {i} ('{header}'): sequence too short ({len(seq)} < {MIN_LENGTH}).")
        if len(seq) > MAX_LENGTH:
            errors.append(f"Record {i} ('{header}'): sequence too long ({len(seq)} > {MAX_LENGTH}).")
        if seq in seen:
            errors.append(f"Record {i} ('{header}'): duplicate sequence.")
        seen.add(seq)

    if errors:
        raise ValueError("Sequence verification failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return seen


def _verify_no_overlap(full_sequences: set[str], antibacterial_sequences: set[str]) -> None:
    overlap = full_sequences & antibacterial_sequences
    if overlap:
        raise ValueError(f"Overlap check failed: {len(overlap)} sequence(s) found in antibacterial reference.")


def _get_commit_sha(repo_dir: Path, git: str = "git") -> str:
    result = subprocess.run([git, "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _invoke_scorer(
    *,
    scorer_path: Path,
    library_fasta: Path,
    profile: str,
    training_fasta: Path,
    reference_amps_fasta: Path,
    reference_generic_fasta: Path,
    report_path: Path,
) -> dict:
    cmd = [
        "uv",
        "run",
        "--project",
        str(scorer_path.parent.parent),
        "python",
        str(scorer_path),
        "--library-fasta",
        str(library_fasta),
        "--profile",
        profile,
        "--training-fasta",
        str(training_fasta),
        "--reference-amps-fasta",
        str(reference_amps_fasta),
        "--reference-generic-fasta",
        str(reference_generic_fasta),
        "--out",
        str(report_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads(report_path.read_text())


# --- subcommand implementations -------------------------------------------------------------


def do_setup(dir: Path, url: str, *, branch, extras, training_fasta: Path, antibacterial_fasta: Path) -> None:
    _check_tool("git")
    print(f"[setup] Cloning {url} -> {dir}")
    _clone_git_repository(dir, url, branch=branch)

    _check_tool("uv")
    print("[setup] Installing dependencies")
    _sync_uv(dir, extras or [])

    print(f"[setup] Staging {TRAINING_DATA_REL_PATH}")
    _stage_file(dir, training_fasta, TRAINING_DATA_REL_PATH)

    print(f"[setup] Staging {ANTIBACTERIAL_DATA_REL_PATH}")
    _stage_file(dir, antibacterial_fasta, ANTIBACTERIAL_DATA_REL_PATH)

    print("\nSetup complete.")


def do_train(dir: Path, *, timeout: float | None) -> float:
    checkpoint_dir = dir / CHECKPOINT_REL_PATH
    if checkpoint_dir.exists():
        print(f"[train] Deleting existing {checkpoint_dir} to force a from-scratch run")
        shutil.rmtree(checkpoint_dir)

    print("[train] Running train")
    t0 = time.monotonic()
    _uv_run(dir, TRAIN_ENTRY_POINT, timeout=timeout)
    train_seconds = time.monotonic() - t0
    print(f"[train] took {train_seconds:.1f}s")
    return train_seconds


def do_generate(dir: Path, *, antibacterial_fasta: Path, timeout: float | None, check_reproducibility: bool = True) -> set[str]:
    library_fasta = dir / GENERATE_ENTRY_POINT / "library.fasta"

    print("[generate] Running generate")
    _uv_run(dir, GENERATE_ENTRY_POINT, timeout=timeout)

    print("[generate] Verifying library.fasta")
    library_sequences = _verify_sequences(library_fasta)

    print(f"[generate] Checking overlap with {antibacterial_fasta}")
    _, antibacterial_seqs = _read_fasta(antibacterial_fasta)
    _verify_no_overlap(library_sequences, set(antibacterial_seqs))

    if check_reproducibility:
        print("[generate] Checking reproducibility (rerunning generate once more)")
        before = library_fasta.read_bytes()
        _uv_run(dir, GENERATE_ENTRY_POINT, timeout=timeout)
        if library_fasta.read_bytes() != before:
            raise ValueError(
                "Reproducibility check failed: two `generate` runs with identical inputs produced different output."
            )

    print(f"\n{len(library_sequences)} sequences verified at {library_fasta}")
    return library_sequences


def do_score(
    dir: Path,
    *,
    scorer_path: Path,
    profile: str,
    training_fasta: Path,
    held_out_fasta: Path,
    generic_fasta: Path,
    report_out: Path | None,
) -> dict:
    library_fasta = dir / GENERATE_ENTRY_POINT / "library.fasta"
    # Always normalize to a '.metrics.json' suffix -- aggregate_scores.py globs for exactly this,
    # and this is the one place that normalization needs to happen, whether do_score is reached via
    # the standalone `score` subcommand or through `all`.
    report_path = (report_out or Path("reports/latest.json")).with_suffix(".metrics.json")

    print("[score] Scoring")
    metrics = _invoke_scorer(
        scorer_path=scorer_path,
        library_fasta=library_fasta,
        profile=profile,
        training_fasta=training_fasta,
        reference_amps_fasta=held_out_fasta,
        reference_generic_fasta=generic_fasta,
        report_path=report_path,
    )
    print(f"\nReport written to {report_path}")
    return metrics


def do_all(
    dir: Path,
    url: str,
    *,
    branch,
    extras,
    training_fasta: Path,
    antibacterial_fasta: Path,
    held_out_fasta: Path,
    generic_fasta: Path,
    scorer_path: Path,
    scorer_profile: str,
    report_out: Path | None,
    train_timeout: float | None,
    generate_timeout: float | None,
    skip_train: bool,
    skip_scoring: bool,
) -> dict:
    do_setup(dir, url, branch=branch, extras=extras, training_fasta=training_fasta, antibacterial_fasta=antibacterial_fasta)

    train_seconds = None
    if skip_train:
        print("[all] --skip-train: using the submission's committed checkpoint/ as-is")
    else:
        train_seconds = do_train(dir, timeout=train_timeout)

    library_sequences = do_generate(dir, antibacterial_fasta=antibacterial_fasta, timeout=generate_timeout)

    report: dict = {
        "url": url,
        "branch": branch,
        "commit": _get_commit_sha(dir),
        "trained_from_scratch": not skip_train,
        "train_seconds": train_seconds,
        "checks_passed": True,
        "n_sequences": len(library_sequences),
    }

    if not skip_scoring:
        report["metrics"] = do_score(
            dir,
            scorer_path=scorer_path,
            profile=scorer_profile,
            training_fasta=training_fasta,
            held_out_fasta=held_out_fasta,
            generic_fasta=generic_fasta,
            report_out=report_out,
        )

    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nCombined report written to {report_out}")

    print("\nAll checks passed. Submission graded.")
    return report


# --- CLI -------------------------------------------------------------------------------------


def _add_dir_arg(p):
    p.add_argument("--dir", type=Path, default=Path("submission"), help="Submission directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_setup = subparsers.add_parser("setup", help="Clone, uv sync, and stage data into --dir.")
    p_setup.add_argument("url", help="GitHub repository URL (or local path).")
    p_setup.add_argument("--branch", default=None, help="Git branch to clone (default: repo default).")
    _add_dir_arg(p_setup)
    p_setup.add_argument("--extra", dest="extras", action="append", default=[], metavar="EXTRA")
    p_setup.add_argument("--training-fasta", type=Path, default=Path("data/training/training.fasta"))
    p_setup.add_argument("--antibacterial-fasta", type=Path, default=Path("data/antibacterial.fasta"))

    p_train = subparsers.add_parser("train", help="Delete checkpoint/ in --dir and run `uv run train`.")
    _add_dir_arg(p_train)
    p_train.add_argument("--timeout", type=float, default=5400, help="Seconds (default 90min).")

    p_generate = subparsers.add_parser("generate", help="Run `uv run generate`, verify, check reproducibility.")
    _add_dir_arg(p_generate)
    p_generate.add_argument("--antibacterial-fasta", type=Path, default=Path("data/antibacterial.fasta"))
    p_generate.add_argument("--timeout", type=float, default=900, help="Seconds (default 15min).")
    p_generate.add_argument("--skip-reproducibility-check", action="store_true")

    p_score = subparsers.add_parser("score", help="Score --dir/generate/library.fasta.")
    _add_dir_arg(p_score)
    p_score.add_argument("--training-fasta", type=Path, default=Path("data/training/training.fasta"))
    p_score.add_argument("--held-out-fasta", type=Path, default=Path("data/training/training.fasta"))
    p_score.add_argument("--generic-fasta", type=Path, default=Path("data/generic/background.fasta"))
    p_score.add_argument("--scorer-path", type=Path, default=Path("scorer/scorer.py"))
    p_score.add_argument("--scorer-profile", default="grader", choices=["student", "grader"])
    p_score.add_argument("--report-out", type=Path, default=None)

    p_all = subparsers.add_parser("all", help="setup -> train (unless --skip-train) -> generate -> score.")
    p_all.add_argument("url", help="GitHub repository URL (or local path).")
    p_all.add_argument("--branch", default=None)
    _add_dir_arg(p_all)
    p_all.add_argument("--extra", dest="extras", action="append", default=[], metavar="EXTRA")
    p_all.add_argument("--training-fasta", type=Path, default=Path("data/training/training.fasta"))
    p_all.add_argument("--antibacterial-fasta", type=Path, default=Path("data/antibacterial.fasta"))
    p_all.add_argument("--held-out-fasta", type=Path, default=Path("data/training/training.fasta"))
    p_all.add_argument("--generic-fasta", type=Path, default=Path("data/generic/background.fasta"))
    p_all.add_argument("--scorer-path", type=Path, default=Path("scorer/scorer.py"))
    p_all.add_argument("--scorer-profile", default="grader", choices=["student", "grader"])
    p_all.add_argument("--report-out", type=Path, default=None)
    p_all.add_argument("--train-timeout", type=float, default=5400, help="Seconds (default 90min).")
    p_all.add_argument("--generate-timeout", type=float, default=900, help="Seconds (default 15min).")
    p_all.add_argument(
        "--skip-train",
        action="store_true",
        help="Use the submission's committed checkpoint/ instead of training from scratch "
        "(phases 1-2, where pretrained weights are allowed).",
    )
    p_all.add_argument("--skip-scoring", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "setup":
            do_setup(
                args.dir,
                args.url,
                branch=args.branch,
                extras=args.extras,
                training_fasta=args.training_fasta,
                antibacterial_fasta=args.antibacterial_fasta,
            )
        elif args.command == "train":
            do_train(args.dir, timeout=args.timeout)
        elif args.command == "generate":
            do_generate(
                args.dir,
                antibacterial_fasta=args.antibacterial_fasta,
                timeout=args.timeout,
                check_reproducibility=not args.skip_reproducibility_check,
            )
        elif args.command == "score":
            do_score(
                args.dir,
                scorer_path=args.scorer_path,
                profile=args.scorer_profile,
                training_fasta=args.training_fasta,
                held_out_fasta=args.held_out_fasta,
                generic_fasta=args.generic_fasta,
                report_out=args.report_out,
            )
        elif args.command == "all":
            do_all(
                args.dir,
                args.url,
                branch=args.branch,
                extras=args.extras,
                training_fasta=args.training_fasta,
                antibacterial_fasta=args.antibacterial_fasta,
                held_out_fasta=args.held_out_fasta,
                generic_fasta=args.generic_fasta,
                scorer_path=args.scorer_path,
                scorer_profile=args.scorer_profile,
                report_out=args.report_out,
                train_timeout=args.train_timeout,
                generate_timeout=args.generate_timeout,
                skip_train=args.skip_train,
                skip_scoring=args.skip_scoring,
            )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
