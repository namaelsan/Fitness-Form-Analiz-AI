"""
Tablo 4.2 — FPS / Gecikme Benchmarkı
=====================================

Her model için ortalama FPS, medyan FPS, p95 gecikme ve algılama oranını ölçer.
FIT3D videoları veya herhangi bir .mp4 dosyası kullanılabilir.

Kullanım:
    # Bir dizindeki tüm MP4'leri kullan:
    python eval_benchmark.py --video-dir /path/to/videos --exercise squat

    # Tek bir video:
    python eval_benchmark.py --video /path/to/squat.mp4 --exercise squat

    # Birden fazla video (glob):
    python eval_benchmark.py --video-dir data/fit3d/s03/videos/50591643 --exercise squat

Çıktılar:  results/benchmark/benchmark_summary.csv
                             aggregate/fps_comparison.png
                             aggregate/latency_boxplot.png
                             aggregate/landmark_jitter.png
"""

import argparse
import sys
from pathlib import Path

SUPPORTED_MODELS = [
    "mediapipe-full",
    "movenet-lightning",
    "movenet-thunder",
    "yolov8",
]

SUPPORTED_EXERCISES = ["curl", "squat", "deadlift", "shoulder_press", "lateral_raise"]


def main() -> int:
    parser = argparse.ArgumentParser(description="FPS / latency benchmark")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory to search for .mp4 files (recursive).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Single video file to benchmark.",
    )
    parser.add_argument(
        "--exercise",
        choices=SUPPORTED_EXERCISES,
        default="squat",
        help="Exercise type (affects which primary joints are tracked).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help="Which models to benchmark (default: all four).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Where to write CSV files and charts.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("W", "H"),
        help="Resize frames before inference (default: 640 480).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap number of videos (useful for a quick test).",
    )
    args = parser.parse_args()

    # Collect video paths
    if args.video is not None:
        if not args.video.exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1
        videos = [args.video]
    elif args.video_dir is not None:
        if not args.video_dir.exists():
            print(f"ERROR: --video-dir not found: {args.video_dir}", file=sys.stderr)
            return 1
        all_mp4 = sorted(args.video_dir.rglob("*.mp4"))
        # Filter to clips whose filename contains the exercise keyword so that
        # e.g. deadlift videos are not benchmarked with squat joint settings.
        videos = [v for v in all_mp4 if args.exercise in v.stem.lower()]
        filtered_out = len(all_mp4) - len(videos)
        if filtered_out:
            print(f"  (skipped {filtered_out} clips whose name does not contain '{args.exercise}')")
    else:
        print("ERROR: provide either --video or --video-dir", file=sys.stderr)
        return 1

    if not videos:
        print(f"ERROR: no .mp4 files found matching exercise='{args.exercise}'.", file=sys.stderr)
        print("Check that filenames contain the exercise keyword (e.g. 'squat_01.mp4').", file=sys.stderr)
        return 1

    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    resize = tuple(args.resize)

    print(f"\n{'=' * 60}")
    print(f"  Videos       : {len(videos)}")
    print(f"  Exercise     : {args.exercise}")
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Resize       : {resize[0]}×{resize[1]}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"{'=' * 60}\n")

    from fitness_form_ai.evaluation.benchmark import run_all_benchmarks
    from fitness_form_ai.evaluation.compare import compare_results
    from fitness_form_ai.evaluation.report import (
        write_benchmark_csv,
        write_comparison_csv,
        plot_fps_comparison,
        plot_latency_boxplot,
        plot_jitter_comparison,
    )

    results = run_all_benchmarks(
        model_names=args.models,
        video_paths=videos,
        exercise_name=args.exercise,
        resize=resize,
        verbose=True,
    )

    if not results:
        print("ERROR: no results produced.", file=sys.stderr)
        return 1

    # Cross-model angle comparison (uses mediapipe-full as reference if available)
    reference = "mediapipe-full" if "mediapipe-full" in args.models else args.models[0]
    comparison = compare_results(results, reference_model=reference)

    # Print table to stdout
    print(f"\n{'=' * 60}")
    print(f"  RESULTS (paste into tez.md Tablo 4.2)")
    print(f"{'=' * 60}")
    header = f"  {'Model':<22} {'Mean FPS':>10} {'Median FPS':>12} {'p95 lat (ms)':>14} {'Detection':>11}"
    print(header)
    print(f"  {'-' * 73}")

    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name in args.models:
        clips = by_model.get(model_name, [])
        if not clips:
            continue
        # Average metrics across videos for this model
        avg_fps = sum(r.mean_fps for r in clips) / len(clips)
        avg_med = sum(r.median_fps for r in clips) / len(clips)
        avg_p95 = sum(r.p95_latency_ms for r in clips) / len(clips)
        avg_det = sum(r.detection_rate for r in clips) / len(clips)
        print(f"  {model_name:<22} {avg_fps:>10.1f} {avg_med:>12.1f} {avg_p95:>14.1f} {avg_det:>10.1%}")

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_csv(results, args.output_dir)
    write_comparison_csv(comparison, args.output_dir)

    try:
        plot_fps_comparison(results, args.output_dir)
        plot_latency_boxplot(results, args.output_dir)
        plot_jitter_comparison(results, args.output_dir)
    except ImportError:
        print("  (matplotlib not installed — skipping charts)")

    print(f"\nDone. Files written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
