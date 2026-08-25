"""Score a generated AMP library using the seqme framework.

DEMO VERSION. This is a copy of the real grading pipeline's scorer.py, with one deliberate
change: the "grader" profile below is configured identically to the "student" profile, so you
can see and run the exact code path grading will use, without access to the real grader's
private configuration (deliberately not disclosed — see the assignment document). The real grader also
passes a secret held-out reference set (never --reference-amps-fasta pointing at your own
training data, as it does here) for FBD(AMPs)/ConformityScore. Everything else about how scoring
works is identical to what will actually happen to your submission.

Adapted from benchmark_peptides.ipynb (a seqme tutorial notebook by the szczurek-lab, not an
official AMP Challenge 2027 artifact — just useful guidance on how to wire up seqme metrics).

Usage:
  uv run python scorer.py --library-fasta generate/library.fasta --profile student \
      --training-fasta data/training/training.fasta \
      --reference-generic-fasta data/generic/background.fasta
"""

import argparse
import json
from pathlib import Path

import seqme as sm

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

PROFILES = {
    "student": {"esm2_checkpoint": "t6_8M"},
    "grader": {
        # DEMO NOTE: identical to "student" here on purpose — see module docstring. Real
        # grading uses a different, undisclosed configuration.
        "esm2_checkpoint": "t6_8M",
    },
}

AMPEPPY_URL = "https://github.com/szczurek-lab/seqme-amPEPpy"


def verify_alphabet(sequences: list[str]) -> None:
    for seq in sequences:
        invalid = set(seq) - AMINO_ACID_SET
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in '{seq}': {', '.join(sorted(invalid))}")


def read_fasta(path: Path) -> list[str]:
    sequences = sm.read_fasta(str(path))
    verify_alphabet(sequences)
    return sequences


def load_reference_sets(args) -> tuple[list[str], list[str], list[str]]:
    seqs_train = read_fasta(args.training_fasta)

    amps_path = args.reference_amps_fasta or args.training_fasta
    seqs_amps = read_fasta(amps_path)

    seqs_generic = read_fasta(args.reference_generic_fasta)

    def maybe_subsample(name, seqs):
        if len(seqs) > args.n_samples:
            seqs = sm.utils.subsample(seqs, n_samples=args.n_samples, seed=args.seed)
        print(f"  {name}: {len(seqs)} sequences")
        return seqs

    print("[2] Reference sets")
    seqs_train = maybe_subsample("training", seqs_train)
    seqs_amps = maybe_subsample("amps (curated AMP reference)", seqs_amps)
    seqs_generic = maybe_subsample("generic (background peptides)", seqs_generic)
    return seqs_train, seqs_amps, seqs_generic


def build_cache(profile_cfg: dict, device: str, thirdparty_dir: Path):
    esm2_checkpoint = getattr(sm.models.ESM2Checkpoint, profile_cfg["esm2_checkpoint"])
    esm2 = sm.models.ESM2(model_name=esm2_checkpoint, batch_size=512, device=device, verbose=False)

    # DEMO NOTE: only amPEPpy is wired up here. Real grading may combine this with additional,
    # undisclosed activity predictors — see module docstring.
    ampeppy = sm.models.ThirdPartyModel(
        entry_point="ampeppy.predict:predict",
        path=thirdparty_dir / "ampeppy",
        url=AMPEPPY_URL,
    )

    models = {
        "esm2-embed": esm2.embed,
        "gravy": sm.models.Gravy(),
        "charge": sm.models.Charge(),
        "amphiphilicity": sm.models.HydrophobicMoment(),
        "amPEPpy": ampeppy,
    }

    return sm.Cache(models=models), "amPEPpy"


def build_metrics(cache, activity_model_name, seqs_train, seqs_amps, seqs_generic, device):
    embedder = cache.model("esm2-embed")
    return [
        sm.metrics.Count(),
        sm.metrics.Uniqueness(),
        sm.metrics.Diversity(k=5, name="Diversity (5)"),
        sm.metrics.Novelty(reference=seqs_train),
        # FKEA defaults to device="cpu" regardless of the embedder's device (unlike FBD/AuthPct,
        # which have no device param at all). Its ~4096x4096 CPU eigendecomposition triggers a
        # PyTorch+MKL bug on some machines ("Intel oneMKL ERROR: Parameter 8 was incorrect on
        # entry to SSYEVD") — pointing it at the GPU sidesteps that entirely.
        sm.metrics.FKEA(embedder=embedder, bandwidth=1.0, strict=False, device=device),
        sm.metrics.FBD(reference=seqs_generic, embedder=embedder, name="FBD (generic)"),
        sm.metrics.FBD(reference=seqs_amps, embedder=embedder, name="FBD (AMPs)"),
        sm.metrics.ID(predictor=cache.model(activity_model_name), name="Activity", objective="maximize"),
        sm.metrics.AuthPct(train_set=seqs_train, embedder=embedder),
        sm.metrics.ConformityScore(
            reference=seqs_amps,
            predictors=[cache.model("amphiphilicity"), cache.model("charge")],
            kde_bandwidth="silverman",
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library-fasta", type=Path, required=True)
    parser.add_argument("--profile", choices=list(PROFILES), required=True)
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument(
        "--reference-amps-fasta",
        type=Path,
        default=None,
        help="Curated AMP reference for FBD(AMPs)/ConformityScore. Defaults to --training-fasta "
        "(this demo has no secret held-out set — the real grader passes one here instead).",
    )
    parser.add_argument("--reference-generic-fasta", type=Path, required=True)
    parser.add_argument("--thirdparty-dir", type=Path, default=Path(".seqme-thirdparty"))
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    profile_cfg = PROFILES[args.profile]
    device = resolve_device(args.device)

    print(f"[1] Loading library from {args.library_fasta}")
    library_sequences = read_fasta(args.library_fasta)
    print(f"    {len(library_sequences)} sequences")

    seqs_train, seqs_amps, seqs_generic = load_reference_sets(args)

    print(f"[3] Building model cache (profile={args.profile}, device={device})")
    cache, activity_model_name = build_cache(profile_cfg, device, args.thirdparty_dir)
    metrics = build_metrics(cache, activity_model_name, seqs_train, seqs_amps, seqs_generic, device)

    print("[4] Scoring")
    df = sm.evaluate({"submission": library_sequences}, metrics)

    print("\n" + df.to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(json.loads(df.to_json(orient="index")), indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
