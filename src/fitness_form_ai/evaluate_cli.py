"""CLI entry point for offline evaluation and benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fitness_form_ai.app.config import SUPPORTED_EXERCISES, SUPPORTED_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate & benchmark pose models on pre-recorded videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m fitness_form_ai.evaluate_cli --videos examples/*.mp4 --exercise squat\n"
            "  fitness-form-evaluate --videos v1.mp4 v2.mp4 --models mediapipe-lite yolov8\n"
        ),
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        required=True,
        help="One or more video file paths to evaluate.",
    )
    parser.add_argument(
        "--exercise",
        choices=SUPPORTED_EXERCISES,
        default="squat",
        help="Exercise to track (default: squat).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="Models to benchmark (default: all).",
    )
    parser.add_argument(
        "--reference",
        choices=SUPPORTED_MODELS,
        default="mediapipe-heavy",
        help="Reference model for confusion matrix (default: mediapipe-heavy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to write results (default: ./results).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("W", "H"),
        help="Resize frames to WxH before inference (default: 640 480).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validate video paths
    for vp in args.videos:
        if not vp.exists():
            print(f"Error: video not found: {vp}", file=sys.stderr)
            return 1

    # Ensure reference model is in the model list
    if args.reference not in args.models:
        args.models.append(args.reference)

    print("=" * 60)
    print("  Fitness Form AI — Evaluation & Benchmarking")
    print("=" * 60)
    print(f"  Videos:    {', '.join(v.name for v in args.videos)}")
    print(f"  Exercise:  {args.exercise}")
    print(f"  Models:    {', '.join(args.models)}")
    print(f"  Reference: {args.reference}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Resize:    {args.resize[0]}×{args.resize[1]}")
    print("=" * 60)

    from fitness_form_ai.evaluation.benchmark import run_all_benchmarks
    from fitness_form_ai.evaluation.compare import compare_results
    from fitness_form_ai.evaluation.report import generate_full_report

    # Run benchmarks
    results = run_all_benchmarks(
        model_names=args.models,
        video_paths=args.videos,
        exercise_name=args.exercise,
        resize=tuple(args.resize),
    )

    # Compare against reference
    comparison = compare_results(results, reference_model=args.reference)

    # Generate reports
    generate_full_report(results, comparison, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
