"""
Tablo 4.1 — PA-MPJPE (FIT3D mocap doğruluğu)
=============================================

Kullanım:
    python eval_mocap.py --fit3d-root /path/to/fit3d --exercise squat

FIT3D klasör yapısı (beklenen):
    <fit3d-root>/
        s03/
            videos/
                <camera_id>/
                    squat_01.mp4
                    squat_02.mp4
            joints3d_25/
                squat_01.json
                squat_02.json

Çıktılar:  results/mocap/mocap_accuracy.csv
                        mocap_summary.csv   (model başına ortalama ± 95% CI)
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


def collect_videos(fit3d_root: Path, exercise: str) -> list[Path]:
    """Find all .mp4 clips whose name contains the exercise keyword."""
    videos = []
    for mp4 in sorted(fit3d_root.rglob("*.mp4")):
        # Only include clips that live inside a "videos/" subfolder
        if "videos" in mp4.parts and exercise in mp4.stem.lower():
            videos.append(mp4)
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-MPJPE evaluation against FIT3D")
    parser.add_argument(
        "--fit3d-root",
        type=Path,
        required=True,
        help="Root of the FIT3D dataset (contains s03/, s04/, …)",
    )
    parser.add_argument(
        "--exercise",
        choices=SUPPORTED_EXERCISES,
        default="squat",
        help="Exercise filter — only clips whose filename contains this string are used.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help="Which models to evaluate (default: all four).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mocap"),
        help="Where to write CSV files.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Restrict to a single camera folder (e.g. 50591643). Default: all cameras.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Cap the number of clips per model (useful for a quick sanity-check run).",
    )
    args = parser.parse_args()

    if not args.fit3d_root.exists():
        print(f"ERROR: --fit3d-root not found: {args.fit3d_root}", file=sys.stderr)
        return 1

    # Collect video paths
    videos = collect_videos(args.fit3d_root, args.exercise)

    if args.camera:
        videos = [v for v in videos if args.camera in v.parts]

    if not videos:
        print(
            f"ERROR: No clips found under {args.fit3d_root} for exercise='{args.exercise}'.",
            file=sys.stderr,
        )
        print("Check that --fit3d-root points to the correct directory and the", file=sys.stderr)
        print("exercise keyword matches clip filenames (e.g. 'squat_01.mp4').", file=sys.stderr)
        return 1

    # Filter to clips that actually have a paired joints3d_25 JSON
    from fitness_form_ai.evaluation.mocap import resolve_gt_path

    paired = [(v, resolve_gt_path(v)) for v in videos]
    missing = [v for v, gt in paired if gt is None]
    if missing:
        print(f"  ! {len(missing)} clip(s) have no joints3d_25 JSON — skipping:")
        for m in missing[:5]:
            print(f"      {m}")
        if len(missing) > 5:
            print(f"      … and {len(missing) - 5} more")

    paired = [(v, gt) for v, gt in paired if gt is not None]
    if not paired:
        print("ERROR: No clip has a matching ground-truth JSON.", file=sys.stderr)
        return 1

    if args.max_clips is not None:
        paired = paired[: args.max_clips]

    video_paths = [v for v, _ in paired]

    print(f"\n{'=' * 60}")
    print(f"  FIT3D root   : {args.fit3d_root}")
    print(f"  Exercise     : {args.exercise}")
    print(f"  Clips        : {len(video_paths)}")
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"{'=' * 60}\n")

    from fitness_form_ai.evaluation.mocap import run_mocap_evaluation
    from fitness_form_ai.evaluation.report import write_mocap_csv, write_mocap_summary_csv

    results = run_mocap_evaluation(
        model_names=args.models,
        video_paths=video_paths,
        exercise_name=args.exercise,
        verbose=True,
    )

    if not results:
        print("ERROR: No results produced.", file=sys.stderr)
        return 1

    # Print summary table to stdout
    print(f"\n{'=' * 60}")
    print(f"  RESULTS (paste into tez.md Tablo 4.1)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<22} {'PA-MPJPE (mm)':>15} {'p95 (mm)':>10} {'Angle MAE (°)':>14} {'Coverage':>10}")
    print(f"  {'-' * 75}")

    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name in args.models:
        clips = by_model.get(model_name, [])
        if not clips:
            continue
        avg_mpjpe = sum(r.pa_mpjpe_mm for r in clips) / len(clips)
        avg_p95 = sum(r.pa_mpjpe_p95_mm for r in clips) / len(clips)
        avg_mae = sum(r.angle_mae_deg for r in clips) / len(clips)
        avg_cov = sum(r.coverage for r in clips) / len(clips)
        print(f"  {model_name:<22} {avg_mpjpe:>15.1f} {avg_p95:>10.1f} {avg_mae:>14.1f} {avg_cov:>9.1%}")

    # Write CSVs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_mocap_csv(results, args.output_dir)
    write_mocap_summary_csv(results, args.output_dir)

    print(f"\nDone. CSV files written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
