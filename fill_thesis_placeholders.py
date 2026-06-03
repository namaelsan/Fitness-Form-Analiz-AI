#!/usr/bin/env python3
"""Run all evaluations needed to fill thesis placeholders.

Usage (two steps):

    # Step 1 — generate form-classification results from your labelled clips
    #   (only needed once, or when labels.csv changes)
    python eval_classification.py --labels labels.csv

    # Step 2 — run PA-MPJPE + FPS protocols against FIT3D and combine
    #   everything into one ready-to-paste file
    cd <project-root>
    python fill_thesis_placeholders.py

Outputs a single file: thesis_results.txt with all values ready to paste.

NOTE: The PA-MPJPE + FPS section takes a while (FIT3D clips × 4 models).
      The form-classification section is instant — it reads the CSVs that
      eval_classification.py already wrote to results/classification/.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
FIT3D_VIDEOS = PROJECT / "fit3d" / "train" / "s03" / "videos"
FIT3D_GT = PROJECT / "fit3d" / "train" / "s03" / "joints3d_25"

# CSVs written by eval_classification.py
CLASS_DIR = PROJECT / "results" / "classification"
EXERCISE_METRICS_CSV = CLASS_DIR / "exercise_metrics.csv"
RULE_METRICS_CSV = CLASS_DIR / "rule_metrics.csv"
CLASS_STATS_CSV = CLASS_DIR / "classification_stats.csv"

OUT = PROJECT / "thesis_results.txt"

# FIT3D clip name → our exercise key
CLIP_TO_EXERCISE: dict[str, str] = {
    "squat": "squat",
    "deadlift": "deadlift",
    "dumbbell_biceps_curls": "curl",
    "dumbbell_overhead_shoulder_press": "shoulder_press",
    "side_lateral_raise": "lateral_raise",
}

# model display names (thesis convention)
MODEL_DISPLAY = {
    "mediapipe-full": "MediaPipe",
    "movenet-lightning": "MoveNet-L",
    "movenet-thunder": "MoveNet-T",
    "yolov8": "YOLOv8n",
}

EXERCISE_DISPLAY = {
    "squat": "Squat",
    "deadlift": "Deadlift",
    "curl": "Kol Kıvrımı",
    "shoulder_press": "Omuz Presi",
    "lateral_raise": "Yana Kaldırma",
}

MODELS = list(MODEL_DISPLAY.keys())
EXERCISES = list(CLIP_TO_EXERCISE.values())


# ── FIT3D clip discovery ──────────────────────────────────────────────────────

def discover_clips() -> list[tuple[Path, str]]:
    """Return (video_path, exercise_key) for all relevant FIT3D clips."""
    clips = []
    if not FIT3D_VIDEOS.exists():
        print(f"ERROR: {FIT3D_VIDEOS} not found", file=sys.stderr)
        sys.exit(1)
    for cam_dir in sorted(FIT3D_VIDEOS.iterdir()):
        if not cam_dir.is_dir():
            continue
        for clip_name, exercise_key in CLIP_TO_EXERCISE.items():
            video = cam_dir / f"{clip_name}.mp4"
            if video.exists():
                clips.append((video, exercise_key))
    return clips


# ── Protocol 1: PA-MPJPE against FIT3D mocap ─────────────────────────────────

def run_mocap(clips: list[tuple[Path, str]]):
    """Protocol 1: PA-MPJPE against FIT3D ground truth.

    One model instance is created per model_name and reused across all clips to
    avoid the cost (and potential fragility) of 80 separate initialisations.
    Exceptions during model creation skip all clips for that model and print a
    full traceback so failures are never silently lost.
    """
    from fitness_form_ai.evaluation.mocap import evaluate_clip_against_mocap
    from fitness_form_ai.inference.factory import create_pose_model

    results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    total = len(clips) * len(MODELS)
    done = 0

    for model_name in MODELS:
        display = MODEL_DISPLAY[model_name]
        print(f"\n  Initialising {display}...")
        model = None
        try:
            model = create_pose_model(model_name)
        except Exception:
            print(f"  ERROR: failed to create {display} — skipping all clips for this model.")
            traceback.print_exc(file=sys.stdout)
            done += len(clips)
            continue

        try:
            for video_path, exercise_key in clips:
                done += 1
                print(f"  [{done}/{total}] mocap: {display} × {video_path.parent.name}/{video_path.stem}")
                try:
                    r = evaluate_clip_against_mocap(
                        model_name, video_path, exercise_key,
                        resize=(640, 480),
                        model=model,        # reuse — no re-init per clip
                    )
                    results[model_name][exercise_key].append(r)
                except Exception:
                    print(f"    SKIP — exception during evaluation:")
                    traceback.print_exc(file=sys.stdout)
        finally:
            if model is not None:
                model.release()

    return results


# ── Protocol 2: FPS / latency benchmark on FIT3D clips ───────────────────────

def run_benchmarks(clips: list[tuple[Path, str]]):
    """Protocol 2: FPS/latency (no form-classification — that uses labels.csv)."""
    from fitness_form_ai.evaluation.benchmark import run_benchmark

    results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    total = len(clips) * len(MODELS)
    done = 0
    for video_path, exercise_key in clips:
        for model_name in MODELS:
            done += 1
            print(f"  [{done}/{total}] bench: {MODEL_DISPLAY[model_name]} × {video_path.parent.name}/{video_path.stem}")
            try:
                r = run_benchmark(model_name, video_path, exercise_key, resize=(640, 480))
                results[model_name][exercise_key].append(r)
            except Exception as e:
                print(f"    SKIP: {e}")
    return results


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_mocap(mocap_results) -> list[tuple]:
    """Build Table 4.1 data (PA-MPJPE)."""
    rows = []
    for model in MODELS:
        for ex in EXERCISES:
            rs = mocap_results[model][ex]
            if not rs:
                rows.append((MODEL_DISPLAY[model], EXERCISE_DISPLAY[ex], "—", "—", "—", "0.0"))
                continue
            all_mpjpe = [f for r in rs for f in r.per_frame_mpjpe_mm]
            all_angle = [f for r in rs for f in r.per_frame_angle_abs_err]
            ev_frames = sum(r.evaluated_frames for r in rs)
            tot_frames = sum(r.total_video_frames for r in rs)
            coverage = ev_frames / tot_frames * 100 if tot_frames else 0
            mean_mpjpe = np.mean(all_mpjpe) if all_mpjpe else 0
            p95_mpjpe = np.percentile(all_mpjpe, 95) if all_mpjpe else 0
            angle_mae = np.mean(all_angle) if all_angle else 0
            rows.append((
                MODEL_DISPLAY[model],
                EXERCISE_DISPLAY[ex],
                f"{mean_mpjpe:.1f}",
                f"{p95_mpjpe:.1f}",
                f"{angle_mae:.1f}",
                f"{coverage:.1f}",
            ))
    return rows


def aggregate_fps(bench_results) -> list[tuple]:
    """Build Table 4.2 data (real-time performance)."""
    rows = []
    for model in MODELS:
        all_lats = []
        total_detected = 0
        total_frames = 0
        for ex in EXERCISES:
            for r in bench_results[model][ex]:
                all_lats.extend(r.latencies_ms)
                total_detected += r.detected_frames
                total_frames += r.total_frames

        if not all_lats:
            rows.append((MODEL_DISPLAY[model], "—", "—", "—"))
            continue
        mean_lat = np.mean(all_lats)
        mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0
        p95_lat = np.percentile(all_lats, 95)
        det_rate = total_detected / total_frames * 100 if total_frames else 0
        rows.append((
            MODEL_DISPLAY[model],
            f"{mean_fps:.1f}",
            f"{p95_lat:.1f}",
            f"{det_rate:.1f}",
        ))
    return rows


# ── Classification: read pre-built CSVs from eval_classification.py ───────────

def _read_csv(path: Path) -> list[dict]:
    """Read a CSV (with possible \r\n artifacts) into a list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # strip any stray carriage-returns from keys and values
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def load_classification_results() -> tuple[list[tuple], list[tuple], dict]:
    """Read exercise_metrics.csv and rule_metrics.csv.

    Returns:
        class_table  — rows for Table 4.3
        rule_table   — rows for Table 4.4
        overall      — overall aggregate dict for summary section
    """
    missing = [p for p in (EXERCISE_METRICS_CSV, RULE_METRICS_CSV) if not p.exists()]
    if missing:
        print("\nERROR: Classification CSVs not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nRun this first:\n"
            "  python eval_classification.py --labels labels.csv\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Table 4.3 ──
    ex_rows = _read_csv(EXERCISE_METRICS_CSV)
    class_table: list[tuple] = []
    overall = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for row in ex_rows:
        ex_key = row.get("exercise", "")
        ex_label = EXERCISE_DISPLAY.get(ex_key, ex_key)
        tp  = int(row.get("tp", 0))
        tn  = int(row.get("tn", 0))
        fp  = int(row.get("fp", 0))
        fn  = int(row.get("fn", 0))
        prec = float(row.get("precision", 0))
        rec  = float(row.get("recall", 0))
        f1   = float(row.get("f1", 0))
        acc  = float(row.get("accuracy", 0))
        spec = float(row.get("specificity", 0))
        spec_str = "—" if (tn + fp) == 0 else f"{spec:.2f}"
        class_table.append((ex_label, f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}", spec_str, f"{acc:.2f}"))
        overall["tp"] += tp
        overall["tn"] += tn
        overall["fp"] += fp
        overall["fn"] += fn

    # Overall row
    tp, tn, fp, fn = overall["tp"], overall["tn"], overall["fp"], overall["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    sp = tn / (tn + fp) if (tn + fp) else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    sp_str = "—" if (tn + fp) == 0 else f"{sp:.2f}"
    class_table.append(("Genel Ort.", f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}", sp_str, f"{acc:.2f}"))

    # ── Table 4.4 ──
    rule_rows = _read_csv(RULE_METRICS_CSV)
    rule_table: list[tuple] = []
    for row in rule_rows:
        ex_key   = row.get("exercise", "")
        ex_label = EXERCISE_DISPLAY.get(ex_key, ex_key)
        rule_name = row.get("rule_name", "")
        tp_r  = int(row.get("tp", 0))
        fn_r  = int(row.get("fn", 0))
        sens  = float(row.get("sensitivity", 0))
        rule_table.append((ex_label, rule_name, f"{tp_r}", f"{fn_r}", f"{sens:.2f}"))

    return class_table, rule_table, overall


def load_classification_stats() -> dict[str, str]:
    """Read the bootstrap CI row from classification_stats.csv if present."""
    if not CLASS_STATS_CSV.exists():
        return {}
    stats = {}
    for row in _read_csv(CLASS_STATS_CSV):
        metric = row.get("metric", "")
        stats[metric] = row
    return stats


# ── Output writer ─────────────────────────────────────────────────────────────

def write_output(
    mocap_table, fps_table, class_table, rule_table, class_overall,
    class_stats, clips,
):
    lines = []
    w = lines.append

    w("=" * 80)
    w("THESIS PLACEHOLDER VALUES")
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    w("=" * 80)

    # ── Dataset stats ──
    cameras = {vp.parent.name for vp, _ in clips}
    w(f"\n## Dataset Stats (Section 3.2 placeholder)")
    w(f"FIT3D cameras  : {len(cameras)} ({', '.join(sorted(cameras))})")
    w(f"Exercises      : {len(CLIP_TO_EXERCISE)}")
    w(f"Total FIT3D clips (camera × exercise): {len(clips)}")
    for ex in EXERCISES:
        count = sum(1 for _, e in clips if e == ex)
        w(f"  {EXERCISE_DISPLAY[ex]}: {count} clips")

    # ── Table 4.1: PA-MPJPE ──
    w(f"\n## Table 4.1: Pose Estimation Accuracy (PA-MPJPE)")
    w("NOTE: MoveNet and YOLOv8 are 2D models (z=0 for all joints). Their PA-MPJPE")
    w("      reflects 2D-projected Procrustes alignment against 3D GT and is not")
    w("      directly comparable to MediaPipe which provides true 3D world landmarks.")
    w(f"{'Model':<12} {'Exercise':<16} {'PA-MPJPE Ort.':<15} {'PA-MPJPE p95':<15} {'Açı MAE (°)':<14} {'Kapsam (%)'}")
    w("-" * 85)
    for row in mocap_table:
        w(f"{row[0]:<12} {row[1]:<16} {row[2]:<15} {row[3]:<15} {row[4]:<14} {row[5]}")

    # ── Table 4.2: FPS ──
    w(f"\n## Table 4.2: Real-Time Performance")
    w(f"{'Model':<25} {'Ort. FPS':<12} {'p95 Gecikme (ms)':<18} {'Algılama (%)'}")
    w("-" * 70)
    for row in fps_table:
        w(f"{row[0]:<25} {row[1]:<12} {row[2]:<18} {row[3]}")

    # ── Table 4.3: Classification ──
    w(f"\n## Table 4.3: Form Classification (MediaPipe, labelled clips)")
    w(f"Source: {EXERCISE_METRICS_CSV}")
    w(f"{'Egzersiz':<16} {'Hassasiyet':<12} {'Duyarlılık':<12} {'F1':<8} {'Özgüllük':<12} {'Doğruluk'}")
    w("-" * 70)
    for row in class_table:
        w(f"{row[0]:<16} {row[1]:<12} {row[2]:<12} {row[3]:<8} {row[4]:<12} {row[5]}")

    # Bootstrap CI (if available)
    if class_stats:
        f1_row  = class_stats.get("f1", {})
        acc_row = class_stats.get("accuracy", {})
        if f1_row:
            w(f"\n  Overall F1  : {f1_row.get('value','?')}  "
              f"95% CI [{f1_row.get('ci_low','?')} – {f1_row.get('ci_high','?')}]")
        if acc_row:
            w(f"  Overall Acc : {acc_row.get('value','?')}  "
              f"95% CI [{acc_row.get('ci_low','?')} – {acc_row.get('ci_high','?')}]")

    # ── Table 4.4: Rule sensitivity ──
    w(f"\n## Table 4.4: Rule Sensitivity (MediaPipe)")
    w(f"Source: {RULE_METRICS_CSV}")
    w(f"{'Egzersiz':<16} {'Kural':<35} {'TP':<5} {'FN':<5} {'Duyarlılık'}")
    w("-" * 75)
    for row in rule_table:
        w(f"{row[0]:<16} {row[1]:<35} {row[2]:<5} {row[3]:<5} {row[4]}")

    # ── Summary for ÖZET / ABSTRACT ──
    w(f"\n## Summary values for ÖZET and ABSTRACT placeholders")
    mp_mpjpes = [float(r[2]) for r in mocap_table if r[0] == "MediaPipe" and r[2] != "—"]
    if mp_mpjpes:
        w(f"MediaPipe overall PA-MPJPE (mean across exercises): {np.mean(mp_mpjpes):.1f} mm")
    overall_f1_row = class_stats.get("f1", {})
    if overall_f1_row:
        w(f"MediaPipe form classification overall F1: {overall_f1_row.get('value','?')}  "
          f"(95% CI {overall_f1_row.get('ci_low','?')}–{overall_f1_row.get('ci_high','?')})")
    else:
        # fall back to last row of class_table (Genel Ort.)
        w(f"MediaPipe form classification overall F1: {class_table[-1][3]}")

    # ── FIT3D reference ──
    w(f"\n## FIT3D Citation (KAYNAKÇA placeholder)")
    w("Fieraru, M., Khoreva, A., Pishchulin, L., & Schiele, B. (2021).")
    w("Three-dimensional human pose estimation in fitness exercises.")
    w("In Proceedings of the IEEE/CVF Winter Conference on Applications")
    w("of Computer Vision (WACV) (pp. 2140–2150).")

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Results written to: {OUT}")
    print(f"{'=' * 60}")
    print(text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Load classification results first so we fail fast if the CSVs are missing
    print("Loading form-classification results from eval_classification.py outputs...")
    class_table, rule_table, class_overall = load_classification_results()
    class_stats = load_classification_stats()
    print(f"  ✓ {len(class_table) - 1} exercises, {len(rule_table)} rules loaded")

    print("\nDiscovering FIT3D clips...")
    clips = discover_clips()
    print(f"Found {len(clips)} clips across {len(set(c[0].parent.name for c in clips))} cameras")

    print("\n[1/2] Running mocap evaluation (PA-MPJPE)...")
    mocap_results = run_mocap(clips)

    print("\n[2/2] Running FPS benchmark...")
    bench_results = run_benchmarks(clips)

    print("\nAggregating results...")
    mocap_table = aggregate_mocap(mocap_results)
    fps_table = aggregate_fps(bench_results)

    write_output(
        mocap_table, fps_table, class_table, rule_table, class_overall,
        class_stats, clips,
    )


if __name__ == "__main__":
    main()
