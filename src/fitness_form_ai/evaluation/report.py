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

    output_dir.mkdir(parents=True, exist_ok=True)

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

    chart_path = output_dir / "fps_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def plot_latency_boxplot(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Box plot showing per-frame latency distribution for each model."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

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

    chart_path = output_dir / "latency_boxplot.png"
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

    output_dir.mkdir(parents=True, exist_ok=True)
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
            
        ref_angles = []
        cand_angles = []
        for rf, cf in zip(ref_run.frames, cand_run.frames):
            if rf.primary_angle is not None and cf.primary_angle is not None:
                ref_angles.append(rf.primary_angle)
                cand_angles.append(cf.primary_angle)
                
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

        safe_name = f"corr_{comp.model_name}_{video_name}".replace("-", "_")
        chart_path = output_dir / f"{safe_name}.png"
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

    output_dir.mkdir(parents=True, exist_ok=True)
    
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

    chart_path = output_dir / "landmark_jitter.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {chart_path}")
    return chart_path


def generate_full_report(
    results: list[BenchmarkResult],
    comparison: ComparisonReport,
    output_dir: Path,
) -> None:
    """Generate all CSVs and charts."""

    print(f"\nGenerating reports in {output_dir} ...")

    write_benchmark_csv(results, output_dir)
    write_comparison_csv(comparison, output_dir)
    plot_fps_comparison(results, output_dir)
    plot_latency_boxplot(results, output_dir)
    plot_jitter_comparison(results, output_dir)
    plot_angle_correlation(results, comparison, output_dir)

    print(f"\nAll reports saved to {output_dir}")
