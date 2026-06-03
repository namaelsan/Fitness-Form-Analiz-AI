"""
Tablo 4.3 & 4.4 — Sınıflandırma Metrikleri ve Kural Duyarlılığı
=================================================================

Her video için "proper" / "casual" tahmini yapar ve etiket CSV'siyle karşılaştırır.

Adım 1 — labels.csv hazırla (bir kez, elle):
    Proje kökündeki labels_template.csv'yi kopyala ve her satırı doldur.
    video_path    : video dosyasının göreceli ya da mutlak yolu
    exercise      : curl | squat | deadlift | shoulder_press | lateral_raise
    expected_class: proper  (form doğru)  veya  casual  (form hatalı)
    expected_violated_rules: ihlal edilen kural adları virgülle ayrılmış
                             (bilmiyorsan boş bırak; TP/FN hesabı için gerekli)

Adım 2 — scripti çalıştır:
    python eval_classification.py --labels labels.csv --exercise squat

Çıktılar:  results/classification/classification_results.csv
                                  exercise_metrics.csv
                                  rule_metrics.csv
                                  classification_stats.csv   (bootstrap CI)
                                  classification/confusion_matrices.png
                                  classification/f1_per_exercise.png
                                  classification/rule_sensitivity.png
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
    parser = argparse.ArgumentParser(description="Form classification evaluation")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels CSV (see labels_template.csv).",
    )
    parser.add_argument(
        "--exercise",
        choices=SUPPORTED_EXERCISES,
        default=None,
        help="Filter to a single exercise (default: all exercises in the CSV).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=["mediapipe-full"],
        help="Models to evaluate (default: mediapipe-full).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classification"),
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
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"ERROR: labels file not found: {args.labels}", file=sys.stderr)
        print("Run:  cp labels_template.csv labels.csv  and fill it in.", file=sys.stderr)
        return 1

    from fitness_form_ai.evaluation.classification import load_labels

    all_labels = load_labels(args.labels)
    if not all_labels:
        print("ERROR: labels CSV is empty.", file=sys.stderr)
        return 1

    # Optional exercise filter
    if args.exercise:
        labels = [lb for lb in all_labels if lb.exercise == args.exercise]
        if not labels:
            print(f"ERROR: no labels for exercise='{args.exercise}'.", file=sys.stderr)
            return 1
    else:
        labels = all_labels

    # Collect unique video paths from the labels
    video_paths = list(dict.fromkeys(Path(lb.video_path) for lb in labels))
    missing = [v for v in video_paths if not v.exists()]
    if missing:
        print(f"WARNING: {len(missing)} video(s) listed in labels not found on disk:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    … and {len(missing) - 5} more")
        video_paths = [v for v in video_paths if v.exists()]

    if not video_paths:
        print("ERROR: none of the labelled videos exist on disk.", file=sys.stderr)
        return 1

    resize = tuple(args.resize)

    print(f"\n{'=' * 60}")
    print(f"  Labels       : {len(labels)} videos  "
          f"({sum(1 for l in labels if l.expected_class=='proper')} proper / "
          f"{sum(1 for l in labels if l.expected_class=='casual')} casual)")
    print(f"  Videos found : {len(video_paths)}")
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Resize       : {resize[0]}×{resize[1]}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"{'=' * 60}\n")

    from fitness_form_ai.evaluation.benchmark import run_all_benchmarks
    from fitness_form_ai.evaluation.classification import evaluate_classification
    from fitness_form_ai.evaluation.report import (
        write_classification_csv,
        write_exercise_metrics_csv,
        write_rule_metrics_csv,
        write_classification_stats_csv,
        plot_confusion_matrix,
        plot_f1_bar,
        plot_rule_sensitivity,
    )

    # We need one benchmark run per model. Each exercise in the labels may differ,
    # so we run exercise by exercise and pool the results.
    exercises_in_labels = list(dict.fromkeys(lb.exercise for lb in labels))
    all_bench_results = []

    for exercise in exercises_in_labels:
        ex_videos = [
            Path(lb.video_path)
            for lb in labels
            if lb.exercise == exercise and Path(lb.video_path).exists()
        ]
        if not ex_videos:
            continue
        print(f"\n--- Benchmarking exercise: {exercise} ({len(ex_videos)} videos) ---")
        bench = run_all_benchmarks(
            model_names=args.models,
            video_paths=ex_videos,
            exercise_name=exercise,
            resize=resize,
            verbose=True,
        )
        all_bench_results.extend(bench)

    if not all_bench_results:
        print("ERROR: benchmark produced no results.", file=sys.stderr)
        return 1

    report = evaluate_classification(all_bench_results, labels)

    # Print metrics table
    print(f"\n{'=' * 60}")
    print("  RESULTS — Tablo 4.3 (sınıflandırma)")
    print(f"{'=' * 60}")
    print(f"  {'Egzersiz':<20} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'Spec':>6} | TP TN FP FN")
    print(f"  {'-' * 70}")
    for em in report.exercise_metrics.values():
        print(
            f"  {em.exercise:<20} {em.precision:>6.2f} {em.recall:>6.2f} "
            f"{em.f1:>6.2f} {em.accuracy:>6.2f} {em.specificity:>6.2f} "
            f"| {em.tp:2} {em.tn:2} {em.fp:2} {em.fn:2}"
        )

    if report.rule_metrics:
        print(f"\n  RESULTS — Tablo 4.4 (kural duyarlılığı)")
        print(f"  {'Egzersiz::Kural':<40} {'Sens.':>7} | TP FN")
        print(f"  {'-' * 55}")
        for key, rm in report.rule_metrics.items():
            print(f"  {key:<40} {rm.sensitivity:>7.2f} | {rm.tp:2} {rm.fn:2}")

    # Write all outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_classification_csv(report, args.output_dir)
    write_exercise_metrics_csv(report, args.output_dir)
    write_rule_metrics_csv(report, args.output_dir)
    write_classification_stats_csv(report, labels, args.output_dir)

    try:
        plot_confusion_matrix(report, args.output_dir)
        plot_f1_bar(report, args.output_dir)
        plot_rule_sensitivity(report, args.output_dir)
    except ImportError:
        print("  (matplotlib not installed — skipping charts)")

    print(f"\nDone. Files written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
