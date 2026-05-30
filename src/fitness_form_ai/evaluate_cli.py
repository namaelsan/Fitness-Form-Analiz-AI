"""CLI entry point for offline evaluation and benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fitness_form_ai.app.config import SUPPORTED_EXERCISES, SUPPORTED_MODELS

# Root of the examples folder, relative to wherever the CLI is invoked.
_DEFAULT_EXAMPLES_DIR = Path("examples")

VIDEO_TYPES = ("casual", "proper")


def resolve_example_videos(
    examples_dir: Path,
    filter_exercise: str | None,
    filter_type: str | None,
) -> list[Path]:
    """Return video paths from *examples_dir* matching the given filters.

    *filter_exercise* – exercise subfolder name (e.g. ``"biceps"``).
                        ``None`` means all exercises.
    *filter_type*     – type subfolder name (``"casual"`` or ``"proper"``).
                        ``None`` means both types.
    """
    if not examples_dir.is_dir():
        raise FileNotFoundError(
            f"Examples directory not found: {examples_dir}\n"
            "Run the CLI from the project root or pass --examples-dir."
        )

    exercise_glob = filter_exercise if filter_exercise else "*"
    type_glob = filter_type if filter_type else "*"

    videos: list[Path] = sorted(
        p
        for p in examples_dir.glob(f"{exercise_glob}/{type_glob}/*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    return videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate & benchmark pose models on pre-recorded videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Test only biceps casual videos\n"
            "  fitness-form-evaluate --video-set biceps/casual --exercise curl\n"
            "\n"
            "  # Test all biceps videos (casual + proper)\n"
            "  fitness-form-evaluate --video-set biceps --exercise curl\n"
            "\n"
            "  # Test every video in the examples folder\n"
            "  fitness-form-evaluate --video-set all --exercise curl\n"
            "\n"
            "  # Explicit paths (original behaviour)\n"
            "  fitness-form-evaluate --videos v1.mp4 v2.mp4 --exercise squat\n"
        ),
    )

    # ---- video source (mutually exclusive) ----
    video_src = parser.add_mutually_exclusive_group(required=True)
    video_src.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        help="One or more explicit video file paths to evaluate.",
    )
    video_src.add_argument(
        "--video-set",
        metavar="EXERCISE[/TYPE]|all",
        help=(
            "Auto-discover videos from the examples folder. "
            "Use  'biceps/casual'  for a specific exercise+type, "
            "'biceps'  for all types of that exercise, "
            "or  'all'  for every available video."
        ),
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=_DEFAULT_EXAMPLES_DIR,
        help=f"Root of the examples folder (default: {_DEFAULT_EXAMPLES_DIR}).",
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
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Open a GUI window and watch the benchmark run in real time.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        metavar="LABELS_CSV",
        help=(
            "Path to ground-truth labels CSV (e.g. examples/labels.csv). "
            "When provided, adds classification metrics (precision, recall, F1, "
            "confusion matrix, per-rule sensitivity) to the report output."
        ),
    )
    parser.add_argument(
        "--mocap",
        action="store_true",
        default=False,
        help=(
            "Score each model against FIT3D marker-based ground truth "
            "(PA-MPJPE and joint-angle MAE). Only clips with a matching "
            "joints3d_25 file are evaluated."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ---- resolve video list ----
    if args.video_set:
        spec = args.video_set.strip().lower()
        if spec == "all":
            filter_exercise, filter_type = None, None
        elif "/" in spec:
            parts = spec.split("/", 1)
            filter_exercise, filter_type = parts[0], parts[1]
            if filter_type not in VIDEO_TYPES:
                print(
                    f"Error: unknown video type '{filter_type}'. "
                    f"Valid types: {', '.join(VIDEO_TYPES)}",
                    file=sys.stderr,
                )
                return 1
        else:
            filter_exercise, filter_type = spec, None

        try:
            video_paths = resolve_example_videos(
                args.examples_dir, filter_exercise, filter_type
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if not video_paths:
            label = args.video_set
            print(
                f"Error: no videos found for '{label}' in {args.examples_dir}",
                file=sys.stderr,
            )
            return 1
    else:
        video_paths = args.videos
        for vp in video_paths:
            if not vp.exists():
                print(f"Error: video not found: {vp}", file=sys.stderr)
                return 1

    # Ensure reference model is in the model list
    if args.reference not in args.models:
        args.models.append(args.reference)

    print("=" * 60)
    print("  Fitness Form AI — Evaluation & Benchmarking")
    print("=" * 60)
    print(f"  Videos:    {', '.join(v.name for v in video_paths)}")
    print(f"  Exercise:  {args.exercise}")
    print(f"  Models:    {', '.join(args.models)}")
    print(f"  Reference: {args.reference}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Resize:    {args.resize[0]}×{args.resize[1]}")
    print("=" * 60)

    if args.live:
        from fitness_form_ai.ui.benchmark_live_app import BenchmarkLiveApp

        app = BenchmarkLiveApp(
            model_names=args.models,
            video_paths=video_paths,
            exercise_name=args.exercise,
            output_dir=args.output_dir,
            reference_model=args.reference,
            resize=tuple(args.resize),
        )
        app.run()
    else:
        from fitness_form_ai.evaluation.benchmark import run_all_benchmarks
        from fitness_form_ai.evaluation.compare import compare_results
        from fitness_form_ai.evaluation.report import generate_full_report

        results = run_all_benchmarks(
            model_names=args.models,
            video_paths=video_paths,
            exercise_name=args.exercise,
            resize=tuple(args.resize),
        )
        comparison = compare_results(results, reference_model=args.reference)

        classification_report = None
        labels = None
        if args.labels is not None:
            if not args.labels.exists():
                print(f"Error: labels file not found: {args.labels}", file=sys.stderr)
                return 1
            from fitness_form_ai.evaluation.classification import (
                evaluate_classification,
                load_labels,
            )
            labels = load_labels(args.labels)
            classification_report = evaluate_classification(results, labels)
            _print_classification_summary(classification_report)

        mocap_results = None
        if args.mocap:
            from fitness_form_ai.evaluation.mocap import run_mocap_evaluation

            print("\n" + "=" * 60)
            print("  FIT3D Ground-Truth Accuracy")
            print("=" * 60)
            mocap_results = run_mocap_evaluation(
                model_names=args.models,
                video_paths=video_paths,
                exercise_name=args.exercise,
                resize=tuple(args.resize),
            )

        generate_full_report(
            results,
            comparison,
            args.output_dir,
            classification_report,
            labels=labels,
            mocap_results=mocap_results,
        )

    return 0


def _print_classification_summary(report: object) -> None:
    """Print a short classification summary to stdout."""
    print("\n" + "=" * 60)
    print("  Classification Summary")
    print("=" * 60)
    for em in report.exercise_metrics.values():  # type: ignore[attr-defined]
        print(
            f"  {em.exercise:<20} "
            f"Acc={em.accuracy:.2f}  P={em.precision:.2f}  "
            f"R={em.recall:.2f}  F1={em.f1:.2f}  "
            f"(TP={em.tp} TN={em.tn} FP={em.fp} FN={em.fn}"
            + (f" no_reps={em.no_reps}" if em.no_reps else "")
            + ")"
        )
    if report.rule_metrics:  # type: ignore[attr-defined]
        print("\n  Per-rule sensitivity (labelled casual videos only):")
        for rm in report.rule_metrics.values():  # type: ignore[attr-defined]
            print(f"    [{rm.exercise}] {rm.rule_name:<35} {rm.sensitivity:.2f}  ({rm.tp}/{rm.tp+rm.fn})")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
