"""Generate CSV tables and matplotlib/seaborn charts from benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

from fitness_form_ai.evaluation.benchmark import BenchmarkResult
from fitness_form_ai.evaluation.compare import ComparisonReport


def write_benchmark_csv(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Write a summary CSV with one row per (model, video, exercise) combo."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"

    fieldnames = [
        "model",
        "video",
        "exercise",
        "total_frames",
        "detected_frames",
        "detection_rate",
        "mean_fps",
        "median_fps",
        "p95_latency_ms",
        "total_reps",
        "valid_reps",
        "invalid_reps",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "model": r.model_name,
                    "video": Path(r.video_path).name,
                    "exercise": r.exercise_name,
                    "total_frames": r.total_frames,
                    "detected_frames": r.detected_frames,
                    "detection_rate": f"{r.detection_rate:.4f}",
                    "mean_fps": f"{r.mean_fps:.2f}",
                    "median_fps": f"{r.median_fps:.2f}",
                    "p95_latency_ms": f"{r.p95_latency_ms:.2f}",
                    "total_reps": r.total_reps,
                    "valid_reps": r.valid_reps,
                    "invalid_reps": r.invalid_reps,
                }
            )

    print(f"  ✓ {csv_path}")
    return csv_path


def write_comparison_csv(report: ComparisonReport, output_dir: Path) -> Path:
    """Write continuous angle metrics (RMSE, MAE, R2) between each model and the reference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison.csv"

    fieldnames = [
        "video",
        "reference_model",
        "model",
        "angle_rmse",
        "angle_mae",
        "r_squared",
        "overlap_frames",
        "ref_rep_count",
        "model_rep_count",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for comp in report.comparisons:
            writer.writerow(
                {
                    "video": Path(comp.video_path).name,
                    "reference_model": comp.reference_model,
                    "model": comp.model_name,
                    "angle_rmse": f"{comp.angle_rmse:.2f}",
                    "angle_mae": f"{comp.angle_mae:.2f}",
                    "r_squared": f"{comp.r_squared:.4f}",
                    "overlap_frames": comp.overlap_frames,
                    "ref_rep_count": comp.ref_rep_count,
                    "model_rep_count": comp.model_rep_count,
                }
            )

    print(f"  ✓ {csv_path}")
    return csv_path


def plot_fps_comparison(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Grouped bar chart: mean FPS per model, grouped by video."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts_dir = output_dir / "aggregate"
    charts_dir.mkdir(parents=True, exist_ok=True)

    models: list[str] = []
    videos: list[str] = []
    for r in results:
        if r.model_name not in models:
            models.append(r.model_name)
        vname = Path(r.video_path).name
        if vname not in videos:
            videos.append(vname)

    fps_lookup: dict[tuple[str, str], float] = {}
    for r in results:
        fps_lookup[(r.model_name, Path(r.video_path).name)] = r.mean_fps

    x = np.arange(len(videos))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        fps_values = [fps_lookup.get((model, v), 0) for v in videos]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, fps_values, width, label=model, zorder=3)
        for bar, val in zip(bars, fps_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Video")
    ax.set_ylabel("Mean FPS")
    ax.set_title("Inference Speed Comparison (FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = charts_dir / "fps_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def plot_latency_boxplot(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Box plot showing per-frame latency distribution for each model."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = output_dir / "aggregate"
    charts_dir.mkdir(parents=True, exist_ok=True)

    model_latencies: dict[str, list[float]] = {}
    for r in results:
        model_latencies.setdefault(r.model_name, []).extend(r.latencies_ms)

    models = list(model_latencies.keys())
    data = [model_latencies[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=models, patch_artist=True, showfliers=False)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Inference Latency Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = charts_dir / "latency_boxplot.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def plot_angle_correlation(results: list[BenchmarkResult], report: ComparisonReport, output_dir: Path) -> list[Path]:
    """Scatter plots comparing model angles to the reference model angle."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    paths: list[Path] = []

    groups: dict[tuple[str, str], dict[str, BenchmarkResult]] = {}
    for r in results:
        key = (r.video_path, r.exercise_name)
        groups.setdefault(key, {})[r.model_name] = r

    for comp in report.comparisons:
        video_key = (comp.video_path, comp.exercise_name)
        ref_run = groups[video_key].get(comp.reference_model)
        cand_run = groups[video_key].get(comp.model_name)
        
        if not ref_run or not cand_run:
            continue
            
        # Align by frame_index (see compare.compare_results for rationale).
        ref_by_idx = {
            f.frame_index: f.primary_angle
            for f in ref_run.frames
            if f.primary_angle is not None
        }
        cand_by_idx = {
            f.frame_index: f.primary_angle
            for f in cand_run.frames
            if f.primary_angle is not None
        }
        shared = sorted(ref_by_idx.keys() & cand_by_idx.keys())
        ref_angles = [ref_by_idx[i] for i in shared]
        cand_angles = [cand_by_idx[i] for i in shared]

        if not ref_angles:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Identity line
        min_val = min(min(ref_angles), min(cand_angles)) - 10
        max_val = max(max(ref_angles), max(cand_angles)) + 10
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect Agreement")
        
        # Scatter
        # Downsample if too many points for clarity (e.g., > 1000)
        step = max(1, len(ref_angles) // 1000)
        ax.scatter(ref_angles[::step], cand_angles[::step], alpha=0.4, color="#4C72B0", s=10)
        
        video_name = Path(comp.video_path).stem
        ax.set_title(
            f"Angle Correlation: {comp.model_name} vs {comp.reference_model}\n"
            f"({video_name} · {comp.exercise_name})\n"
            f"R²={comp.r_squared:.3f} | RMSE={comp.angle_rmse:.1f}° | MAE={comp.angle_mae:.1f}°",
            fontsize=10,
        )
        ax.set_xlabel(f"{comp.reference_model} Primary Joint Angle (°)")
        ax.set_ylabel(f"{comp.model_name} Primary Joint Angle (°)")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Force square aspect ratio
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        fig.tight_layout()

        video_dir = output_dir / "correlation" / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        safe_model = comp.model_name.replace("-", "_")
        chart_path = video_dir / f"{safe_model}.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {chart_path}")
        paths.append(chart_path)

    return paths


def plot_jitter_comparison(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Bar chart showing frame-to-frame angle jitter (average absolute derivative)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts_dir = output_dir / "aggregate"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    jitter_by_model: dict[str, list[float]] = {}
    
    for r in results:
        angles = [f.primary_angle for f in r.frames if f.primary_angle is not None]
        if len(angles) > 1:
            diffs = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
            avg_jitter = sum(diffs) / len(diffs)
            jitter_by_model.setdefault(r.model_name, []).append(avg_jitter)
            
    if not jitter_by_model:
        return Path()
        
    models = list(jitter_by_model.keys())
    # Average the jitter across the videos for a macro-level bar chart
    mean_jitters = [sum(jitter_by_model[m]) / len(jitter_by_model[m]) for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52", "#CCB974"]
    bars = ax.bar(models, mean_jitters, color=colors[: len(models)], zorder=3)

    for bar, val in zip(bars, mean_jitters):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}°",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Avg Frame-to-Frame Delta (°)")
    ax.set_title("Landmark Stability (Jitter)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = charts_dir / "landmark_jitter.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def write_classification_csv(report: "ClassificationReport", output_dir: Path) -> Path:
    """Write per-video classification results to CSV."""
    from fitness_form_ai.evaluation.classification import ClassificationReport  # noqa: F401

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classification_results.csv"

    fieldnames = [
        "video",
        "exercise",
        "expected_class",
        "predicted_class",
        "correct",
        "outcome",
        "expected_violated_rules",
        "actually_failed_rules",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for vcr in report.video_results:
            writer.writerow({
                "video": Path(vcr.video_path).name,
                "exercise": vcr.exercise,
                "expected_class": vcr.expected_class,
                "predicted_class": vcr.predicted_class,
                "correct": vcr.is_correct,
                "outcome": vcr.outcome,
                "expected_violated_rules": "|".join(vcr.expected_violated_rules),
                "actually_failed_rules": "|".join(vcr.actually_failed_rules),
            })

    print(f"  ✓ {csv_path}")
    return csv_path


def write_exercise_metrics_csv(report: "ClassificationReport", output_dir: Path) -> Path:
    """Write per-exercise precision / recall / F1 / accuracy to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "exercise_metrics.csv"

    fieldnames = [
        "exercise", "tp", "tn", "fp", "fn", "no_reps",
        "precision", "recall", "f1", "accuracy", "specificity",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for em in report.exercise_metrics.values():
            writer.writerow({
                "exercise": em.exercise,
                "tp": em.tp,
                "tn": em.tn,
                "fp": em.fp,
                "fn": em.fn,
                "no_reps": em.no_reps,
                "precision": f"{em.precision:.4f}",
                "recall": f"{em.recall:.4f}",
                "f1": f"{em.f1:.4f}",
                "accuracy": f"{em.accuracy:.4f}",
                "specificity": f"{em.specificity:.4f}",
            })

    print(f"  ✓ {csv_path}")
    return csv_path


def write_rule_metrics_csv(report: "ClassificationReport", output_dir: Path) -> Path:
    """Write per-rule sensitivity to CSV (only rules with labelled casual videos)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rule_metrics.csv"

    fieldnames = ["exercise", "rule_name", "tp", "fn", "sensitivity"]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rm in report.rule_metrics.values():
            writer.writerow({
                "exercise": rm.exercise,
                "rule_name": rm.rule_name,
                "tp": rm.tp,
                "fn": rm.fn,
                "sensitivity": f"{rm.sensitivity:.4f}",
            })

    print(f"  ✓ {csv_path}")
    return csv_path


def write_classification_stats_csv(
    report: "ClassificationReport",
    labels: "list[VideoLabel]",
    output_dir: Path,
) -> Path:
    """Write bootstrap CIs (clustered by subject) and the majority baseline."""
    from fitness_form_ai.evaluation.stats import (
        classification_cis,
        majority_class_baseline,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classification_stats.csv"

    # Only videos that were actually scored (exclude no_reps) count toward CIs.
    scored = [vr for vr in report.video_results if vr.outcome in {"TP", "TN", "FP", "FN"}]
    cis = classification_cis(scored)
    baseline = majority_class_baseline(labels)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "ci_low", "ci_high", "note"])
        writer.writerow([
            "accuracy", f"{cis.accuracy.point:.4f}",
            f"{cis.accuracy.ci_low:.4f}", f"{cis.accuracy.ci_high:.4f}",
            f"{cis.n_videos} videos / {cis.n_subjects} subjects, 95% cluster bootstrap",
        ])
        writer.writerow([
            "f1", f"{cis.f1.point:.4f}",
            f"{cis.f1.ci_low:.4f}", f"{cis.f1.ci_high:.4f}", "",
        ])
        writer.writerow([
            f"baseline_accuracy ({baseline.strategy})",
            f"{baseline.accuracy:.4f}", "", "", "majority-class floor",
        ])
        writer.writerow([
            f"baseline_f1 ({baseline.strategy})",
            f"{baseline.f1:.4f}", "", "", "",
        ])
    print(f"  ✓ {csv_path}")
    return csv_path


def plot_confusion_matrix(report: "ClassificationReport", output_dir: Path) -> Path:
    """Heatmap grid: one confusion matrix per exercise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts_dir = output_dir / "classification"
    charts_dir.mkdir(parents=True, exist_ok=True)

    exercises = list(report.exercise_metrics.keys())
    n = len(exercises)
    if n == 0:
        return Path()

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, ex in zip(axes_flat, exercises):
        em = report.exercise_metrics[ex]
        matrix = np.array([[em.tn, em.fp], [em.fn, em.tp]])
        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: proper", "Pred: casual"])
        ax.set_yticklabels(["Actual: proper", "Actual: casual"])
        ax.set_title(f"{ex}\nF1={em.f1:.2f}  Acc={em.accuracy:.2f}", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=14, color="black")
        fig.colorbar(im, ax=ax, shrink=0.7)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices by Exercise", fontsize=13, y=1.02)
    fig.tight_layout()
    chart_path = charts_dir / "confusion_matrices.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def plot_f1_bar(report: "ClassificationReport", output_dir: Path) -> Path:
    """Bar chart of F1 score per exercise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = output_dir / "classification"
    charts_dir.mkdir(parents=True, exist_ok=True)

    exercises = list(report.exercise_metrics.keys())
    f1_scores = [report.exercise_metrics[ex].f1 for ex in exercises]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax.bar(exercises, f1_scores, color=colors[: len(exercises)], zorder=3)

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Form Classification F1 Score per Exercise")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = charts_dir / "f1_per_exercise.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def plot_rule_sensitivity(report: "ClassificationReport", output_dir: Path) -> Path:
    """Horizontal bar chart of per-rule detection sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = output_dir / "classification"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not report.rule_metrics:
        return Path()

    labels_list = [f"{rm.exercise}\n{rm.rule_name}" for rm in report.rule_metrics.values()]
    sensitivities = [rm.sensitivity for rm in report.rule_metrics.values()]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels_list) * 0.45)))
    y_pos = range(len(labels_list))
    ax.barh(list(y_pos), sensitivities, color="#4C72B0", zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_list, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Sensitivity (recall per rule)")
    ax.set_title("Per-Rule Detection Sensitivity\n(only rules with labelled casual videos)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    chart_path = charts_dir / "rule_sensitivity.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def write_mocap_csv(results: "list[MocapEvalResult]", output_dir: Path) -> Path:
    """Write per-clip ground-truth accuracy (PA-MPJPE, angle MAE) to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mocap_accuracy.csv"

    fieldnames = [
        "model", "video", "exercise", "subject",
        "pa_mpjpe_mm", "pa_mpjpe_p95_mm", "angle_mae_deg",
        "evaluated_frames", "coverage",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r.model_name,
                "video": Path(r.video_path).name,
                "exercise": r.exercise_name,
                "subject": r.subject,
                "pa_mpjpe_mm": f"{r.pa_mpjpe_mm:.2f}",
                "pa_mpjpe_p95_mm": f"{r.pa_mpjpe_p95_mm:.2f}",
                "angle_mae_deg": f"{r.angle_mae_deg:.2f}",
                "evaluated_frames": r.evaluated_frames,
                "coverage": f"{r.coverage:.4f}",
            })
    print(f"  ✓ {csv_path}")
    return csv_path


def write_mocap_summary_csv(results: "list[MocapEvalResult]", output_dir: Path) -> Path:
    """Aggregate ground-truth accuracy per model, with subject-held-out stats.

    Each clip is one observation.  We report the mean PA-MPJPE / angle MAE
    across clips together with a 95% bootstrap CI *clustered by subject* so the
    interval reflects between-subject variation, not just between-frame noise.
    """
    from fitness_form_ai.evaluation.stats import bootstrap_ci_clustered

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mocap_summary.csv"

    by_model: dict[str, list["MocapEvalResult"]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    fieldnames = [
        "model", "n_clips", "n_subjects",
        "pa_mpjpe_mm", "pa_mpjpe_ci_low", "pa_mpjpe_ci_high",
        "angle_mae_deg", "angle_mae_ci_low", "angle_mae_ci_high",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model, rs in by_model.items():
            mpjpe = [r.pa_mpjpe_mm for r in rs]
            angle = [r.angle_mae_deg for r in rs]
            subjects = [r.subject for r in rs]
            m_lo, m_hi = bootstrap_ci_clustered(mpjpe, subjects)
            a_lo, a_hi = bootstrap_ci_clustered(angle, subjects)
            writer.writerow({
                "model": model,
                "n_clips": len(rs),
                "n_subjects": len(set(subjects)),
                "pa_mpjpe_mm": f"{sum(mpjpe) / len(mpjpe):.2f}",
                "pa_mpjpe_ci_low": f"{m_lo:.2f}",
                "pa_mpjpe_ci_high": f"{m_hi:.2f}",
                "angle_mae_deg": f"{sum(angle) / len(angle):.2f}",
                "angle_mae_ci_low": f"{a_lo:.2f}",
                "angle_mae_ci_high": f"{a_hi:.2f}",
            })
    print(f"  ✓ {csv_path}")
    return csv_path


def generate_full_report(
    results: list[BenchmarkResult],
    comparison: ComparisonReport,
    output_dir: Path,
    classification_report: "ClassificationReport | None" = None,
    labels: "list[VideoLabel] | None" = None,
    mocap_results: "list[MocapEvalResult] | None" = None,
) -> None:
    """Generate all CSVs and charts."""

    print(f"\nGenerating reports in {output_dir} ...")

    write_benchmark_csv(results, output_dir)
    write_comparison_csv(comparison, output_dir)
    plot_fps_comparison(results, output_dir)
    plot_latency_boxplot(results, output_dir)
    plot_jitter_comparison(results, output_dir)
    plot_angle_correlation(results, comparison, output_dir)

    if classification_report is not None:
        write_classification_csv(classification_report, output_dir)
        write_exercise_metrics_csv(classification_report, output_dir)
        write_rule_metrics_csv(classification_report, output_dir)
        plot_confusion_matrix(classification_report, output_dir)
        plot_f1_bar(classification_report, output_dir)
        plot_rule_sensitivity(classification_report, output_dir)
        if labels is not None:
            write_classification_stats_csv(classification_report, labels, output_dir)

    if mocap_results:
        write_mocap_csv(mocap_results, output_dir)
        write_mocap_summary_csv(mocap_results, output_dir)

    print(f"\nAll reports saved to {output_dir}")
