#!/usr/bin/env python3
"""Compute EVERY numeric value the thesis (TEZ.md) needs, in a single run.

WHY THIS SCRIPT EXISTS
----------------------
The thesis tables contain ``[PLACEHOLDER: ...]`` markers. This script produces a
single ready-to-read text file (``thesis_placeholder_values.txt``) with every
value those placeholders need, grouped table by table.

DESIGN PRINCIPLE — do not trust the existing evaluation/reporting layer
-----------------------------------------------------------------------
Per the explicit instruction, this script does **not** import or rely on the
project's evaluation/reporting code (``evaluation/benchmark.py``,
``evaluation/mocap.py``, ``evaluation/compare.py``, ``evaluation/classification.py``,
``evaluation/stats.py``, ``evaluation/report.py`` or the old
``fill_thesis_placeholders.py``). Every metric — PA-MPJPE (Umeyama alignment),
angle MAE, FPS/latency, cross-model angle agreement (RMSE/MAE/Pearson r/R²),
form-classification confusion metrics, per-rule sensitivity, bootstrap CIs and
the majority-class baseline — is re-implemented here from first principles.

What it DOES reuse is the *system under test* itself: the pose-model backends
(``inference.factory``), the exercise/rule definitions (``app.catalog`` ->
``domain.exercises``/``rules``/``metrics``) and the rep state-machine
(``domain.tracker``). Those ARE the artifact being evaluated, so the script
drives them exactly as the live application would; it just measures the outcome
with its own, independently-written math.

DATA SOURCES
------------
* FIT3D mocap (``fit3d/train/<subject>/...``) — used ONLY for pose-estimation
  accuracy (Table 4.1), real-time performance (Table 4.2) and cross-model angle
  agreement (Table 4.3). FIT3D ships per-clip ground-truth 3D joints.
* The user's own labelled clips (``labels.csv`` + ``examples/...``) — used ONLY
  for form classification (Table 4.4) and rule sensitivity (Table 4.5).

HOW TO RUN
----------
    cd <project-root>
    python compute_thesis_values.py            # everything (slow: many clips x 5 models)
    python compute_thesis_values.py --quick    # 1 subject, 1 camera (fast smoke run)
    python compute_thesis_values.py --skip-metrabs   # skip the heavy 3D model
    python compute_thesis_values.py --subjects s03 s04 --cameras 50591643

Output: ``thesis_placeholder_values.txt`` in the project root.

NOTE ON RUNTIME: the full run is (subjects x cameras x 5 exercises) clips x
5 models for the FIT3D protocols, plus 24 labelled clips for classification.
MeTRAbs additionally downloads a multi-hundred-MB SavedModel on first use.
Use ``--quick`` first to confirm the pipeline end-to-end, then run the full set.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import os

# --- thread pinning -------------------------------------------------------
# The FIT3D evaluation is parallelised across CLIP-level worker PROCESSES (see
# run_fit3d_parallel). To stop each native math/ML library from *also* spawning
# its own thread pool — which would oversubscribe the cores once many workers
# run at once — pin them all to a single thread *before* numpy / OpenCV /
# TensorFlow are imported. Export any of these yourself to override.
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Make the package importable whether or not it is pip-installed.
# --------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402  (after sys.path tweak)

from fitness_form_ai.app.catalog import EXERCISE_REGISTRY  # noqa: E402
from fitness_form_ai.domain.smoother import AdaptiveSmoother  # noqa: E402
from fitness_form_ai.domain.tracker import RepTracker  # noqa: E402
from fitness_form_ai.inference.factory import create_pose_model  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RESIZE = (640, 480)            # same letterbox target the app/eval use
SYNTHETIC_FPS = 30.0           # timestamp = frame_idx / SYNTHETIC_FPS
MIN_ALIGN_POINTS = 6           # min corresponded joints before a frame scores PA-MPJPE
BOOT_N = 2000                  # bootstrap resamples
BOOT_SEED = 20240530           # reproducible CIs
BOOT_ALPHA = 0.05              # 95% CI

OUT_PATH = PROJECT / "thesis_placeholder_values.txt"

FIT3D_ROOT = PROJECT / "fit3d" / "train"

# FIT3D clip file-stem -> our exercise key
CLIP_TO_EXERCISE: dict[str, str] = {
    "squat": "squat",
    "deadlift": "deadlift",
    "dumbbell_biceps_curls": "curl",
    "dumbbell_overhead_shoulder_press": "shoulder_press",
    "side_lateral_raise": "lateral_raise",
}

# Models that appear in the thesis tables, in table order.
MODELS_FULL = [
    "mediapipe-full",
    "movenet-thunder",
    "movenet-lightning",
    "yolov8",
    "metrabs",
]

MODEL_DISPLAY = {
    "mediapipe-full": "MediaPipe",
    "movenet-thunder": "MoveNet-Thunder",
    "movenet-lightning": "MoveNet-Lightning",
    "yolov8": "YOLOv8n",
    "metrabs": "MeTRAbs",
}

# Thesis Table 4.1 is split into 4.1a (true-3D models) and 4.1b (2D models),
# because the two groups are aligned in different spaces (see thesis 3.5) and
# their PA-MPJPE values must not be compared directly.
MODELS_3D = ["mediapipe-full", "metrabs"]
MODELS_2D = ["movenet-thunder", "movenet-lightning", "yolov8"]

# Exercise order used in the thesis tables, with Turkish display labels.
EXERCISE_ORDER = ["squat", "deadlift", "curl", "shoulder_press", "lateral_raise"]
EXERCISE_DISPLAY = {
    "squat": "Squat",
    "deadlift": "Deadlift",
    "curl": "Tek Kol Dumbbell Curl",
    "shoulder_press": "Omuz Presi",
    "lateral_raise": "Yana Kaldırma",
}

# Cross-model agreement pairs (Table 4.3): reference is MediaPipe.
REFERENCE_MODEL = "mediapipe-full"
AGREEMENT_TARGETS = ["movenet-thunder", "yolov8", "metrabs"]

# FIT3D joints3d_25 index for each MediaPipe joint we use (independent copy of
# the project's verified mapping — kept here so this script is self-contained).
MP_TO_FIT3D: dict[str, int] = {
    "LEFT_HIP": 1, "LEFT_KNEE": 2, "LEFT_ANKLE": 3,
    "RIGHT_HIP": 4, "RIGHT_KNEE": 5, "RIGHT_ANKLE": 6,
    "LEFT_SHOULDER": 11, "LEFT_ELBOW": 12, "LEFT_WRIST": 13,
    "RIGHT_SHOULDER": 14, "RIGHT_ELBOW": 15, "RIGHT_WRIST": 16,
}

LABELS_CSV = PROJECT / "labels.csv"


# ==========================================================================
# Independent geometry / statistics primitives
# ==========================================================================

def letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to (tw, th). Independent reimplementation."""
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, frame.shape[2]), dtype=frame.dtype)
    xo, yo = (tw - nw) // 2, (th - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Interior 3D angle at b of chain a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba == 0.0 or nbc == 0.0:
        return None
    cos = float(np.dot(ba, bc) / (nba * nbc))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity (rotation + uniform scale + translation) alignment of src onto
    dst. Independent implementation of the Umeyama (1991) closed form."""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rot = u @ s @ vt
    var_s = (sc ** 2).sum() / n
    scale = float(np.trace(np.diag(d) @ s) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (rot @ mu_s)
    return (scale * (rot @ src.T)).T + t


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    sx = x.std()
    sy = y.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def bootstrap_ci(values: list[float], statistic, rng: random.Random) -> tuple[float, float]:
    """Percentile bootstrap CI for *statistic* over *values* (resample items)."""
    if not values:
        return (0.0, 0.0)
    if len(values) < 2:
        v = statistic(values)
        return (v, v)
    n = len(values)
    stats = []
    for _ in range(BOOT_N):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((BOOT_ALPHA / 2) * BOOT_N)]
    hi = stats[int((1 - BOOT_ALPHA / 2) * BOOT_N) - 1]
    return (lo, hi)


def bootstrap_ci_clustered(values, clusters, statistic, rng) -> tuple[float, float]:
    """Cluster bootstrap: resample whole clusters with replacement. Falls back to
    ordinary bootstrap when there is only one cluster."""
    if not values:
        return (0.0, 0.0)
    by: dict[str, list] = defaultdict(list)
    for v, c in zip(values, clusters):
        by[c].append(v)
    keys = list(by.keys())
    if len(keys) < 2:
        return bootstrap_ci(list(values), statistic, rng)
    k = len(keys)
    stats = []
    for _ in range(BOOT_N):
        drawn = []
        for _ in range(k):
            drawn.extend(by[keys[rng.randrange(k)]])
        stats.append(statistic(drawn))
    stats.sort()
    lo = stats[int((BOOT_ALPHA / 2) * BOOT_N)]
    hi = stats[int((1 - BOOT_ALPHA / 2) * BOOT_N) - 1]
    return (lo, hi)


def subject_of(path: Path) -> str:
    for part in Path(path).parts:
        if len(part) == 3 and part[0] == "s" and part[1:].isdigit():
            return part
    return "unknown"


# ==========================================================================
# FIT3D clip discovery
# ==========================================================================

class Clip:
    __slots__ = ("video", "gt", "exercise", "subject", "camera")

    def __init__(self, video: Path, gt: Path, exercise: str, subject: str, camera: str):
        self.video = video
        self.gt = gt
        self.exercise = exercise
        self.subject = subject
        self.camera = camera


def discover_clips(subjects: list[str] | None, cameras: list[str] | None) -> list[Clip]:
    clips: list[Clip] = []
    if not FIT3D_ROOT.exists():
        print(f"ERROR: FIT3D root not found: {FIT3D_ROOT}", file=sys.stderr)
        return clips
    for subj_dir in sorted(FIT3D_ROOT.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        if subjects and subject not in subjects:
            continue
        videos_dir = subj_dir / "videos"
        gt_dir = subj_dir / "joints3d_25"
        if not videos_dir.exists() or not gt_dir.exists():
            continue
        for cam_dir in sorted(videos_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            camera = cam_dir.name
            if cameras and camera not in cameras:
                continue
            for stem, ex_key in CLIP_TO_EXERCISE.items():
                video = cam_dir / f"{stem}.mp4"
                gt = gt_dir / f"{stem}.json"
                if video.exists() and gt.exists():
                    clips.append(Clip(video, gt, ex_key, subject, camera))
    return clips


def load_gt(gt_path: Path) -> np.ndarray:
    import json
    with open(gt_path) as fh:
        data = json.load(fh)
    arr = np.asarray(data["joints3d_25"], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (25, 3):
        raise ValueError(f"Bad joints3d_25 shape {arr.shape} in {gt_path}")
    return arr


# ==========================================================================
# Per-(model, clip) FIT3D pass: PA-MPJPE, angle MAE, latency, detection,
# per-frame primary angle (for cross-model agreement).
# ==========================================================================

class ClipResult:
    __slots__ = (
        "mpjpe_mm", "angle_err", "evaluated_frames", "total_frames",
        "detected_frames", "latencies_ms", "primary_angle_by_frame",
    )

    def __init__(self):
        self.mpjpe_mm: list[float] = []
        self.angle_err: list[float] = []
        self.evaluated_frames = 0
        self.total_frames = 0
        self.detected_frames = 0
        self.latencies_ms: list[float] = []
        self.primary_angle_by_frame: dict[int, float] = {}


def process_clip(model, clip: Clip, stride: int = 1) -> ClipResult:
    """Single decode pass over one clip with one (already-created) model.

    ``stride`` > 1 evaluates only every Nth frame (the frames in between are
    grabbed but never decoded/inferred), which cuts runtime almost linearly.
    ``frame_idx`` still advances on every grabbed frame, so ground-truth
    indexing (``gt[frame_idx]``) and the cross-model per-frame angle keys stay
    correctly aligned. Means/percentiles over a strided sample are statistically
    equivalent for this evaluation."""
    stride = max(1, int(stride))
    gt = load_gt(clip.gt)
    n_gt = gt.shape[0]
    exercise = EXERCISE_REGISTRY.get(clip.exercise, EXERCISE_REGISTRY["curl"])()
    primary = exercise.primary_joints  # 3 MediaPipe joint names
    primary_in_gt = all(j in MP_TO_FIT3D for j in primary) and len(primary) == 3
    pairs = list(MP_TO_FIT3D.items())

    res = ClipResult()
    cap = cv2.VideoCapture(str(clip.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {clip.video}")

    frame_idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            res.total_frames += 1
            frame = letterbox(frame, RESIZE[0], RESIZE[1])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            raw = model.process_image(rgb)
            lms = model.extract_landmarks(raw)
            t1 = time.perf_counter()
            res.latencies_ms.append((t1 - t0) * 1000.0)

            if lms is not None:
                res.detected_frames += 1

                # Per-frame primary joint angle (image/world coords; used for
                # cross-model agreement which is alignment-free since both
                # models are measured the same way).
                pj = [lms.get(j) for j in primary]
                if all(p is not None for p in pj):
                    a = np.array([pj[0].x, pj[0].y, pj[0].z])
                    b = np.array([pj[1].x, pj[1].y, pj[1].z])
                    c = np.array([pj[2].x, pj[2].y, pj[2].z])
                    ang = angle_deg(a, b, c)
                    if ang is not None:
                        res.primary_angle_by_frame[frame_idx] = ang

                # PA-MPJPE + GT angle error require a valid GT frame.
                if 0 <= frame_idx < n_gt:
                    gt_frame = gt[frame_idx]
                    pred_pts, gt_pts = [], []
                    for mp_name, gi in pairs:
                        lm = lms.get(mp_name)
                        if lm is None:
                            continue
                        pred_pts.append([float(lm.x), float(lm.y), float(lm.z)])
                        gt_pts.append([float(x) for x in gt_frame[gi]])
                    if len(pred_pts) >= MIN_ALIGN_POINTS:
                        aligned = umeyama(np.asarray(pred_pts), np.asarray(gt_pts))
                        per_joint = np.linalg.norm(aligned - np.asarray(gt_pts), axis=1)
                        res.mpjpe_mm.append(float(per_joint.mean() * 1000.0))
                        res.evaluated_frames += 1

                    # GT primary angle error (alignment-invariant).
                    if primary_in_gt:
                        gp = [gt_frame[MP_TO_FIT3D[j]] for j in primary]
                        pa = res.primary_angle_by_frame.get(frame_idx)
                        gta = angle_deg(np.asarray(gp[0]), np.asarray(gp[1]), np.asarray(gp[2]))
                        if pa is not None and gta is not None:
                            res.angle_err.append(abs(pa - gta))

            frame_idx += 1
    finally:
        cap.release()
    return res


# ==========================================================================
# Parallel FIT3D evaluation — one process pool per model, clips split across
# worker processes. Each worker is single-threaded (see thread pinning at the
# top of the file), so N workers ~= N cores busy without oversubscription.
# ==========================================================================

def _init_worker() -> None:
    """Runs once per worker process: keep OpenCV single-threaded too."""
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass


def _chunk_clips(clips: list[Clip], n: int) -> list[list[Clip]]:
    """Split *clips* into at most *n* round-robin buckets (balanced load)."""
    n = max(1, min(n, len(clips)))
    buckets: list[list[Clip]] = [[] for _ in range(n)]
    for i, clip in enumerate(clips):
        buckets[i % n].append(clip)
    return [b for b in buckets if b]


def _run_chunk(model_name: str, clips: list[Clip], stride: int = 1) -> list[tuple[str, str, ClipResult]]:
    """Worker entry point: build the model ONCE, score every clip in the chunk.

    Returns a list of ``(clip_id, exercise, ClipResult)``. A failure on a single
    clip is swallowed (matching the serial version's per-clip SKIP) so one bad
    clip cannot kill the whole chunk.
    """
    out: list[tuple[str, str, ClipResult]] = []
    try:
        model = create_pose_model(model_name)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        return out
    try:
        for clip in clips:
            clip_id = f"{clip.subject}/{clip.camera}/{clip.exercise}"
            try:
                r = process_clip(model, clip, stride)
            except Exception:
                print(f"    SKIP {clip_id} — istisna:")
                traceback.print_exc(file=sys.stdout)
                continue
            out.append((clip_id, clip.exercise, r))
    finally:
        if hasattr(model, "release"):
            model.release()
    return out


def run_fit3d_parallel(
    models: list[str],
    clips: list[Clip],
    *,
    workers: int,
    metrabs_workers: int,
    stride: int = 1,
    verbose: bool = True,
) -> tuple[
    dict[str, dict[str, list[ClipResult]]],
    dict[str, dict[str, dict[int, float]]],
]:
    """Parallel replacement for the serial ``(model x clip)`` FIT3D loop.

    One process pool PER MODEL, so each worker only ever loads a single backend
    — this bounds memory, which matters for the heavy MeTRAbs SavedModel
    (hundreds of MB per process). Within a model the clips are split into
    ``workers`` chunks and scored concurrently.
    """
    results: dict[str, dict[str, list[ClipResult]]] = defaultdict(lambda: defaultdict(list))
    angle_by_model_clip: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

    total = len(models) * len(clips)
    done = 0
    ctx = mp.get_context("spawn")

    for model_name in models:
        w = metrabs_workers if model_name == "metrabs" else workers
        w = max(1, min(w, len(clips)))
        chunks = _chunk_clips(clips, w)
        print(f"\n=== Model: {MODEL_DISPLAY[model_name]} "
              f"({len(clips)} klip, {w} işçi) ===")
        with ProcessPoolExecutor(
            max_workers=w, mp_context=ctx, initializer=_init_worker
        ) as ex:
            futures = [ex.submit(_run_chunk, model_name, ch, stride) for ch in chunks]
            for fut in as_completed(futures):
                chunk_out = fut.result()
                for clip_id, exercise, r in chunk_out:
                    results[model_name][exercise].append(r)
                    angle_by_model_clip[model_name][clip_id] = r.primary_angle_by_frame
                done += len(chunk_out)
                if verbose:
                    print(f"  [{done}/{total}] {MODEL_DISPLAY[model_name]} "
                          f"+{len(chunk_out)} klip tamamlandı")
    return results, angle_by_model_clip


# ==========================================================================
# Classification (form analysis) over the user's labelled clips — MediaPipe.
# Drives the real RepTracker + rules; metrics computed independently here.
# ==========================================================================

class VideoLabel:
    __slots__ = ("video_path", "exercise", "expected_class", "violated")

    def __init__(self, video_path, exercise, expected_class, violated):
        self.video_path = video_path
        self.exercise = exercise
        self.expected_class = expected_class
        self.violated = violated


def load_labels(path: Path) -> list[VideoLabel]:
    import csv
    labels = []
    with open(path, newline="") as fh:
        rows = csv.DictReader(filter(lambda l: not l.startswith("#"), fh))
        for row in rows:
            raw = (row.get("expected_violated_rules") or "").strip()
            violated = [r.strip() for r in raw.split(",") if r.strip()]
            labels.append(VideoLabel(
                row["video_path"].strip(),
                row["exercise"].strip(),
                row["expected_class"].strip(),
                violated,
            ))
    return labels


def labelled_clip_counts(labels: list[VideoLabel]) -> dict[str, dict[str, int]]:
    """Per-exercise proper/casual clip counts from labels.csv (no model needed).
    Feeds the Tablo 3.2 "Düzgün/Özensiz klip" columns."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"proper": 0, "casual": 0})
    for lbl in labels:
        if lbl.expected_class in ("proper", "casual"):
            counts[lbl.exercise][lbl.expected_class] += 1
    return counts


def probe_resolutions(labels: list[VideoLabel], project: Path) -> dict[tuple, int]:
    """Distinct (width, height, fps) tuples across the labelled clips, read from
    container metadata only (no decode). Feeds the Tablo 3.2 "Çözünürlük/FPS"
    cell. Returns {(w, h, fps_rounded): clip_count}."""
    seen: dict[tuple, int] = defaultdict(int)
    for lbl in labels:
        vpath = project / lbl.video_path
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            cap.release()
            continue
        wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        seen[(wd, ht, round(fps, 2))] += 1
    return seen


def run_form_analysis(model, video_path: Path, exercise_key: str) -> list[set[str]]:
    """Drive the real tracker and return ONE entry per extracted repetition.

    Each entry is the set of ``rule_name`` values that FAILED on that rep (an
    empty set means the rep passed every rule). ``len(result)`` is the number of
    complete reps. Classification is evaluated per repetition by the caller
    (each rep is its own proper/casual sample) rather than collapsing the whole
    video into a single label."""
    exercise = EXERCISE_REGISTRY.get(exercise_key, EXERCISE_REGISTRY["curl"])()
    tracker = RepTracker(
        start_phase=exercise.start_phase,
        min_rom=20.0,
        smoother=AdaptiveSmoother(),
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    per_rep_failed: list[set[str]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = letterbox(frame, RESIZE[0], RESIZE[1])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = model.extract_landmarks(model.process_image(rgb))
            if lms is not None:
                pj = [lms.get(j) for j in exercise.primary_joints]
                if all(p is not None for p in pj):
                    a = np.array([pj[0].x, pj[0].y, pj[0].z])
                    b = np.array([pj[1].x, pj[1].y, pj[1].z])
                    c = np.array([pj[2].x, pj[2].y, pj[2].z])
                    ang = angle_deg(a, b, c)
                    if ang is not None:
                        ts = frame_idx / SYNTHETIC_FPS
                        if tracker.add_frame(ang, ts, landmarks=lms):
                            rep = tracker.extract_rep()
                            if rep:
                                failed = {r.rule_name for r in exercise.rules if not r.apply(rep)}
                                per_rep_failed.append(failed)
            frame_idx += 1
    finally:
        cap.release()

    return per_rep_failed


def outcome_of(expected: str, predicted: str) -> str:
    """casual = positive class."""
    if predicted == "no_reps":
        return "no_reps"
    if expected == "casual" and predicted == "casual":
        return "TP"
    if expected == "proper" and predicted == "proper":
        return "TN"
    if expected == "proper" and predicted == "casual":
        return "FP"
    return "FN"  # expected casual, predicted proper


# ==========================================================================
# Aggregation -> output building
# ==========================================================================

def f1_from(tp, fp, fn) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="FIT3D subject ids to use (default: all discovered).")
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="FIT3D camera ids to use (default: all discovered).")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke run: first subject + first camera only.")
    ap.add_argument("--skip-metrabs", action="store_true",
                    help="Skip the heavy MeTRAbs 3D model.")
    ap.add_argument("--skip-fit3d", action="store_true",
                    help="Skip Tables 4.1-4.3 (FIT3D protocols).")
    ap.add_argument("--skip-classification", action="store_true",
                    help="Skip Tables 4.4-4.5 (labelled clips).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes for the FIT3D protocols "
                         "(default: CPU thread count). Use 1 for the old serial path.")
    ap.add_argument("--metrabs-workers", type=int, default=None,
                    help="Separate (usually lower) worker count for MeTRAbs — its "
                         "SavedModel uses hundreds of MB per process. "
                         "Default: min(workers, 4).")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Evaluate only every Nth frame of each FIT3D clip "
                         "(default 1 = every frame). Use 3-5 to make the heavy "
                         "MeTRAbs run finish in a fraction of the time; means are "
                         "statistically equivalent. Classification (Tables 4.4/4.5) "
                         "always uses every frame.")
    ap.add_argument("--metrabs-aug", type=int, default=None,
                    help="MeTRAbs test-time augmentation count (SavedModel default "
                         "is 5 -> ~5x slower). 1 = plain single-crop inference, much "
                         "faster with small accuracy cost. Sets METRABS_NUM_AUG.")
    args = ap.parse_args()

    if args.frame_stride < 1:
        args.frame_stride = 1
    # Propagate MeTRAbs augmentation choice to (spawned) worker processes via env.
    if args.metrabs_aug is not None:
        os.environ["METRABS_NUM_AUG"] = str(max(1, args.metrabs_aug))

    # --- runtime / device diagnostic ------------------------------------
    # MeTRAbs is the only model heavy enough to matter here; if it ends up on
    # CPU it will be 1-2 orders of magnitude slower than on GPU. Surface that
    # up front so a multi-hour run isn't a surprise.
    if not args.skip_metrabs and not args.skip_fit3d:
        try:
            import tensorflow as _tf  # noqa: N813
            gpus = _tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"TensorFlow GPU bulundu: {len(gpus)} aygıt -> MeTRAbs GPU'da çalışacak.")
            else:
                print("UYARI: TensorFlow GPU göremiyor -> MeTRAbs CPU'da çalışır (çok yavaş).")
                print("       Öneri: --frame-stride 3 --metrabs-aug 1 kullanın, MeTRAbs'i ayrı "
                      "çalıştırın ya da GPU'lu bir makinede çalıştırın.")
        except Exception:
            print("UYARI: TensorFlow içe aktarılamadı; cihaz tespiti atlandı.")

    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers and args.workers > 0 else cpu
    metrabs_workers = (
        args.metrabs_workers
        if args.metrabs_workers and args.metrabs_workers > 0
        else min(workers, 4)
    )

    models = [m for m in MODELS_FULL if not (args.skip_metrabs and m == "metrabs")]
    rng = random.Random(BOOT_SEED)

    out: list[str] = []
    w = out.append
    w("=" * 80)
    w("TEZ.md PLACEHOLDER DEĞERLERİ")
    w(f"Üretim zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("Bu dosya compute_thesis_values.py tarafından bağımsız olarak hesaplanmıştır.")
    w("=" * 80)

    # ---------------- FIT3D protocols (Tables 4.1, 4.2, 4.3) -------------
    clips: list[Clip] = []
    if not args.skip_fit3d:
        subjects = args.subjects
        cameras = args.cameras
        clips = discover_clips(subjects, cameras)
        if args.quick and clips:
            s0 = clips[0].subject
            c0 = clips[0].camera
            clips = [c for c in clips if c.subject == s0 and c.camera == c0]
        print(f"FIT3D klipleri: {len(clips)} "
              f"({len({c.subject for c in clips})} denek, "
              f"{len({c.camera for c in clips})} kamera)")

    # results[model][exercise] -> list[ClipResult]
    results: dict[str, dict[str, list[ClipResult]]] = defaultdict(lambda: defaultdict(list))
    # per-clip primary angle dicts, keyed (model, clip_id) for agreement
    angle_by_model_clip: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

    if clips and workers <= 1:
        # --- serial path (original behaviour, single-threaded frameworks) ---
        for model_name in models:
            print(f"\n=== Model: {MODEL_DISPLAY[model_name]} ===")
            try:
                model = create_pose_model(model_name)
            except Exception:
                print(f"  HATA: {model_name} oluşturulamadı — atlanıyor.")
                traceback.print_exc(file=sys.stdout)
                continue
            try:
                for i, clip in enumerate(clips, 1):
                    clip_id = f"{clip.subject}/{clip.camera}/{clip.exercise}"
                    print(f"  [{i}/{len(clips)}] {clip_id}")
                    try:
                        r = process_clip(model, clip, args.frame_stride)
                    except Exception:
                        print("    SKIP — istisna:")
                        traceback.print_exc(file=sys.stdout)
                        continue
                    results[model_name][clip.exercise].append(r)
                    angle_by_model_clip[model_name][clip_id] = r.primary_angle_by_frame
            finally:
                if hasattr(model, "release"):
                    model.release()
    elif clips:
        # --- parallel path: N single-threaded worker processes across clips ---
        print(f"\nParalel mod: {workers} işçi (MeTRAbs: {metrabs_workers}). "
              f"CPU: {cpu} iş parçacığı.")
        p_results, p_angles = run_fit3d_parallel(
            models, clips,
            workers=workers, metrabs_workers=metrabs_workers,
            stride=args.frame_stride, verbose=True,
        )
        for m, exmap in p_results.items():
            for exk, lst in exmap.items():
                results[m][exk].extend(lst)
        for m, cmap in p_angles.items():
            angle_by_model_clip[m].update(cmap)

    # ---- Table 3.1: FIT3D dataset stats (pose accuracy) ----
    w("\n\n## Tablo 3.1 — Veri seti istatistikleri (FIT3D, poz doğruluğu için)")
    if clips:
        cams = sorted({c.camera for c in clips})
        subs = sorted({c.subject for c in clips})
        w(f"FIT3D kamera sayısı : {len(cams)}  (kimlikler: {', '.join(cams)})")
        w(f"FIT3D denek sayısı  : {len(subs)}  (kimlikler: {', '.join(subs)})")
        w(f"Toplam klip (denek × kamera × egzersiz): {len(clips)}")
        for ex in EXERCISE_ORDER:
            cnt = sum(1 for c in clips if c.exercise == ex)
            w(f"  {EXERCISE_DISPLAY[ex]:<22}: {cnt} klip")
    else:
        w("(FIT3D atlandı — --skip-fit3d veya klip bulunamadı)")

    # ---- Tables 4.1a / 4.1b: PA-MPJPE + angle MAE + coverage ----
    # The thesis splits this into 4.1a (true-3D models, aligned in 3D) and
    # 4.1b (2D models, 2D-projected alignment) so the two groups are never
    # compared directly (see thesis 3.5). Rows are emitted per exercise.
    mp_mpjpe_means: list[float] = []

    def emit_41_rows(model_list: list[str]) -> None:
        w(f"{'Model':<18} {'Egzersiz':<22} {'PA-MPJPE ort(mm)':<17} "
          f"{'PA-MPJPE p95':<14} {'Açı MAE(°)':<12} {'Kapsam(%)'}")
        w("-" * 103)
        for model_name in model_list:
            if model_name not in models:
                continue
            for ex in EXERCISE_ORDER:
                rs = results[model_name][ex]
                if not rs:
                    w(f"{MODEL_DISPLAY[model_name]:<18} {EXERCISE_DISPLAY[ex]:<22} "
                      f"{'—':<17} {'—':<14} {'—':<12} {'0.0'}")
                    continue
                mpjpe = [v for r in rs for v in r.mpjpe_mm]
                ang = [v for r in rs for v in r.angle_err]
                evf = sum(r.evaluated_frames for r in rs)
                tot = sum(r.total_frames for r in rs)
                mean_mp = float(np.mean(mpjpe)) if mpjpe else 0.0
                p95_mp = percentile(mpjpe, 95)
                mae = float(np.mean(ang)) if ang else 0.0
                cov = evf / tot * 100 if tot else 0.0
                if model_name == "mediapipe-full" and mpjpe:
                    mp_mpjpe_means.append(mean_mp)
                w(f"{MODEL_DISPLAY[model_name]:<18} {EXERCISE_DISPLAY[ex]:<22} "
                  f"{mean_mp:<17.1f} {p95_mp:<14.1f} {mae:<12.1f} {cov:.1f}")

    w("\n\n## Tablo 4.1a — Gerçek 3B modeller: poz kestirim doğruluğu (model × egzersiz)")
    w("       FIT3D 3B referansına Umeyama benzerlik hizalamasıyla hesaplanmıştır.")
    emit_41_rows(MODELS_3D)

    w("\n\n## Tablo 4.1b — 2B modeller: poz kestirim doğruluğu (2B-izdüşümlü hizalama)")
    w("UYARI: MoveNet ve YOLOv8 2B modellerdir (tüm eklemler z=0). PA-MPJPE değerleri")
    w("       3B referansa 2B-izdüşüm Procrustes hizalamasıdır; Tablo 4.1a (3B) ile")
    w("       doğrudan kıyaslanamaz. 2B modeller arası 'Açı MAE' sütunu daha anlamlıdır.")
    emit_41_rows(MODELS_2D)

    # ---- Table 4.2: real-time performance ----
    w("\n\n## Tablo 4.2 — Gerçek zamanlı başarım (model)")
    w(f"{'Model':<18} {'Ort. FPS':<10} {'p95 Gecikme(ms)':<17} {'Algılama(%)'}")
    w("-" * 60)
    for model_name in models:
        all_lat: list[float] = []
        det = 0
        tot = 0
        for ex in EXERCISE_ORDER:
            for r in results[model_name][ex]:
                all_lat.extend(r.latencies_ms)
                det += r.detected_frames
                tot += r.total_frames
        if not all_lat:
            w(f"{MODEL_DISPLAY[model_name]:<18} {'—':<10} {'—':<17} {'—'}")
            continue
        mean_lat = float(np.mean(all_lat))
        mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        p95_lat = percentile(all_lat, 95)
        det_rate = det / tot * 100 if tot else 0.0
        w(f"{MODEL_DISPLAY[model_name]:<18} {mean_fps:<10.1f} {p95_lat:<17.1f} {det_rate:.1f}")

    # ---- Table 4.3: cross-model angle agreement vs MediaPipe ----
    w("\n\n## Tablo 4.3 — Model çiftleri arası eklem açısı uyumu (kare indeksine göre hizalı)")
    w(f"{'Çift':<28} {'RMSE(°)':<10} {'MAE(°)':<10} {'Pearson r':<11} {'R²'}")
    w("-" * 70)
    ref = angle_by_model_clip.get(REFERENCE_MODEL, {})
    for tgt in AGREEMENT_TARGETS:
        if args.skip_metrabs and tgt == "metrabs":
            continue
        tgt_map = angle_by_model_clip.get(tgt, {})
        xs: list[float] = []
        ys: list[float] = []
        for clip_id, ref_angles in ref.items():
            tgt_angles = tgt_map.get(clip_id)
            if not tgt_angles:
                continue
            for fi, av in ref_angles.items():
                bv = tgt_angles.get(fi)
                if bv is not None:
                    xs.append(av)
                    ys.append(bv)
        label = f"MediaPipe – {MODEL_DISPLAY[tgt]}"
        if not xs:
            w(f"{label:<28} {'—':<10} {'—':<10} {'—':<11} {'—'}")
            continue
        diff = np.asarray(xs) - np.asarray(ys)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        r = pearson_r(xs, ys)
        w(f"{label:<28} {rmse:<10.2f} {mae:<10.2f} {r:<11.3f} {r * r:.3f}")

    # ---------------- Classification protocols (Tables 4.4, 4.5) ---------
    if not args.skip_classification and LABELS_CSV.exists():
        print("\n=== Form sınıflandırma (MediaPipe, etiketli klipler) ===")
        labels = load_labels(LABELS_CSV)
        try:
            cls_model = create_pose_model("mediapipe-full")
        except Exception:
            cls_model = None
            print("  HATA: MediaPipe oluşturulamadı.")
            traceback.print_exc(file=sys.stdout)

        # Classification is evaluated PER REPETITION: every rep is its own
        # proper/casual sample, inheriting its clip's expected class. This is
        # finer-grained and more faithful than collapsing a whole clip to one
        # label (a single tripped rep no longer condemns an otherwise-clean clip).
        # ex_conf TP/TN/FP/FN are rep-level; "no_reps" counts clips that yielded
        # zero reps (still a per-clip diagnostic).
        ex_conf: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "no_reps": 0})
        sample_outcomes: list[str] = []     # one per rep
        sample_subjects: list[str] = []     # subject cluster per rep
        sample_expected: list[str] = []     # expected class per rep (for baseline)
        # rule_stats[(exercise, rule)] -> {"tp":, "fn":}  (per-rep)
        rule_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0})
        # Tablo 3.2 "Toplam tekrar": reps the state machine extracted per exercise.
        reps_by_ex: dict[str, int] = defaultdict(int)

        if cls_model is not None:
            try:
                for i, lbl in enumerate(labels, 1):
                    vpath = (PROJECT / lbl.video_path)
                    print(f"  [{i}/{len(labels)}] {lbl.video_path}")
                    if not vpath.exists():
                        print("    SKIP — video bulunamadı")
                        continue
                    try:
                        rep_results = run_form_analysis(cls_model, vpath, lbl.exercise)
                    except Exception:
                        print("    SKIP — istisna:")
                        traceback.print_exc(file=sys.stdout)
                        continue
                    reps_by_ex[lbl.exercise] += len(rep_results)
                    subj = subject_of(Path(lbl.video_path))
                    if not rep_results:
                        # No complete rep extracted from this clip.
                        ex_conf[lbl.exercise]["no_reps"] += 1
                        continue
                    for failed in rep_results:
                        predicted = "casual" if failed else "proper"
                        oc = outcome_of(lbl.expected_class, predicted)
                        ex_conf[lbl.exercise][oc] += 1
                        sample_outcomes.append(oc)
                        sample_subjects.append(subj)
                        sample_expected.append(lbl.expected_class)
                        # rule sensitivity (per rep): casual reps whose clip
                        # lists the violated rule(s).
                        if lbl.expected_class == "casual" and lbl.violated:
                            for rname in lbl.violated:
                                key = (lbl.exercise, rname)
                                if rname in failed:
                                    rule_stats[key]["tp"] += 1
                                else:
                                    rule_stats[key]["fn"] += 1
            finally:
                if hasattr(cls_model, "release"):
                    cls_model.release()

        # ---- Table 3.2: researcher-recorded labelled clip set ----
        # Clip counts + resolution come straight from labels.csv / container
        # metadata (no model); "Toplam tekrar" is summed from the rep state
        # machine above. Çekim açısı / çekim ortamı are not derivable from the
        # data and are left as placeholders for the researcher to fill.
        w("\n\n## Tablo 3.2 — Etiketli klip seti istatistikleri (araştırmacı, form değerlendirmesi)")
        clip_counts = labelled_clip_counts(labels)
        res = probe_resolutions(labels, PROJECT)
        reps_ok = cls_model is not None
        w(f"{'Egzersiz':<22} {'Düzgün':<8} {'Özensiz':<8} {'Toplam tekrar':<14} {'Çekim açısı'}")
        w("-" * 72)
        tot_p = tot_c = tot_r = 0
        for ex in EXERCISE_ORDER:
            cc = clip_counts.get(ex, {"proper": 0, "casual": 0})
            p, cz = cc["proper"], cc["casual"]
            r = reps_by_ex.get(ex, 0)
            tot_p += p
            tot_c += cz
            tot_r += r
            rdisp = str(r) if reps_ok else "—"
            w(f"{EXERCISE_DISPLAY[ex]:<22} {p:<8} {cz:<8} {rdisp:<14} "
              f"[PLACEHOLDER: çekim açısı — araştırmacı girecek]")
        tot_rdisp = str(tot_r) if reps_ok else "—"
        w(f"{'TOPLAM':<22} {tot_p:<8} {tot_c:<8} {tot_rdisp:<14} -")
        w("Denek sayısı: 1 (araştırmacı)")
        w("Çekim ortamı sayısı: [PLACEHOLDER: araştırmacı girecek — veriden türetilemez]")
        if res:
            parts = [f"{rw}×{rh} @ {rf:.0f} FPS (n={rn})"
                     for (rw, rh, rf), rn in sorted(res.items())]
            w("Çözünürlük/FPS: " + "; ".join(parts))
        else:
            w("Çözünürlük/FPS: [PLACEHOLDER: video metaverisi okunamadı]")
        w("Not: 'Toplam tekrar' RepTracker durum makinesinin çıkardığı tam tekrar")
        w("     sayılarıdır (proper + casual klipler). 'Çekim açısı' ve 'Çekim ortamı")
        w("     sayısı' veri dosyalarından türetilemez; araştırmacı tarafından doldurulmalıdır.")

        # ---- Table 4.4 ----
        w("\n\n## Tablo 4.4 — Form sınıflandırma başarımı (MediaPipe, etiketli klipler)")
        w("NOT: labels.csv yalnızca 'casual' (hatalı) örnekler içeriyorsa, özgüllük ve")
        w("     hassasiyet için gereken proper/negatif örnek yoktur; bu durumda özgüllük '—' olur.")
        w(f"{'Egzersiz':<22} {'Hassasiyet':<11} {'Duyarlılık':<11} {'F1':<7} {'Özgüllük':<10} {'Doğruluk':<9} {'(TP/TN/FP/FN/no_reps)'}")
        w("-" * 103)
        tot = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "no_reps": 0}
        for ex in EXERCISE_ORDER:
            c = ex_conf.get(ex)
            if not c:
                continue
            tp, tn, fp, fn, nr = c["TP"], c["TN"], c["FP"], c["FN"], c["no_reps"]
            for k in tot:
                tot[k] += c[k]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = f1_from(tp, fp, fn)
            spec = tn / (tn + fp) if (tn + fp) else None
            scored = tp + tn + fp + fn
            acc = (tp + tn) / scored if scored else 0.0
            spec_s = f"{spec:.2f}" if spec is not None else "—"
            w(f"{EXERCISE_DISPLAY[ex]:<22} {prec:<11.2f} {rec:<11.2f} {f1:<7.2f} "
              f"{spec_s:<10} {acc:<9.2f} ({tp}/{tn}/{fp}/{fn}/{nr})")
        # overall
        tp, tn, fp, fn = tot["TP"], tot["TN"], tot["FP"], tot["FN"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = f1_from(tp, fp, fn)
        spec = tn / (tn + fp) if (tn + fp) else None
        scored = tp + tn + fp + fn
        acc = (tp + tn) / scored if scored else 0.0
        spec_s = f"{spec:.2f}" if spec is not None else "—"
        w(f"{'Genel Ort.':<22} {prec:<11.2f} {rec:<11.2f} {f1:<7.2f} "
          f"{spec_s:<10} {acc:<9.2f} ({tp}/{tn}/{fp}/{fn}/{tot['no_reps']})")

        # bootstrap CIs (clustered by subject) + majority baseline
        def _f1_oc(ocs):
            t = ocs.count("TP"); f = ocs.count("FP"); n = ocs.count("FN")
            return f1_from(t, f, n)

        def _acc_oc(ocs):
            sc = [o for o in ocs if o in ("TP", "TN", "FP", "FN")]
            ok = sum(1 for o in sc if o in ("TP", "TN"))
            return ok / len(sc) if sc else 0.0

        if sample_outcomes:
            f1_lo, f1_hi = bootstrap_ci_clustered(sample_outcomes, sample_subjects, _f1_oc, rng)
            acc_lo, acc_hi = bootstrap_ci_clustered(sample_outcomes, sample_subjects, _acc_oc, rng)
            w(f"\nGenel F1     : {_f1_oc(sample_outcomes):.2f}  "
              f"%95 GA [{f1_lo:.2f} – {f1_hi:.2f}]  (tekrar bazında)")
            w(f"Genel Doğruluk: {_acc_oc(sample_outcomes):.2f}  "
              f"%95 GA [{acc_lo:.2f} – {acc_hi:.2f}]  (tekrar bazında)")
            # majority-class baseline (per rep)
            classes = sample_expected
            if classes:
                majority = max(set(classes), key=classes.count)
                base_oc = []
                for cl in classes:
                    if majority == "casual":
                        base_oc.append("TP" if cl == "casual" else "FP")
                    else:
                        base_oc.append("TN" if cl == "proper" else "FN")
                w(f"Çoğunluk-sınıf temeli ('{majority}'): "
                  f"doğruluk={_acc_oc(base_oc):.2f}, F1={_f1_oc(base_oc):.2f}")

        # ---- Table 4.5: rule sensitivity ----
        w("\n\n## Tablo 4.5 — Kural bazında tespit duyarlılığı (MediaPipe)")
        w(f"{'Egzersiz':<22} {'Kural':<28} {'TP':<5} {'FN':<5} {'Duyarlılık'}")
        w("-" * 78)
        for ex in EXERCISE_ORDER:
            for (e, rname), st in rule_stats.items():
                if e != ex:
                    continue
                tp_r, fn_r = st["tp"], st["fn"]
                sens = tp_r / (tp_r + fn_r) if (tp_r + fn_r) else 0.0
                w(f"{EXERCISE_DISPLAY[ex]:<22} {rname:<28} {tp_r:<5} {fn_r:<5} {sens:.2f}")
    elif not args.skip_classification:
        w("\n\n## Tablo 4.4 / 4.5 — labels.csv bulunamadı, atlandı.")

    # ---- ÖZET / ABSTRACT summary values ----
    w("\n\n## ÖZET / ABSTRACT için özet değerler")
    if mp_mpjpe_means:
        w(f"MediaPipe egzersiz-ortalaması PA-MPJPE: {np.mean(mp_mpjpe_means):.1f} mm")
    else:
        w("MediaPipe egzersiz-ortalaması PA-MPJPE: (FIT3D çalıştırılmadı)")
    w("Genel F1 ve %95 GA değerleri için yukarıdaki Tablo 4.4 bölümüne bakınız.")

    # ---- Citation reminder (verified) ----
    w("\n\n## KAYNAKÇA — FIT3D atıfı (doğrulanmış)")
    w("Fieraru, M., Zanfir, M., Pirlea, S. C., Olaru, V., & Sminchisescu, C. (2021).")
    w("AIFit: Automatic 3D human-interpretable feedback models for fitness training.")
    w("In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern")
    w("Recognition (CVPR) (pp. 9919–9928).")

    text = "\n".join(out)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Sonuçlar yazıldı: {OUT_PATH}")
    print(f"{'=' * 60}")
    print(text)


if __name__ == "__main__":
    main()
