"""Batch-process videos with each pose model, measuring FPS and form analysis."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fitness_form_ai.app.catalog import EXERCISE_REGISTRY
from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.smoother import AdaptiveSmoother
from fitness_form_ai.domain.tracker import RepTracker
from fitness_form_ai.inference.base import PoseModel
from fitness_form_ai.inference.factory import create_pose_model
from fitness_form_ai.utils.geometry import calculate_angle_3d, letterbox_resize
from fitness_form_ai.utils.landmarks import read_landmark


@dataclass(slots=True)
class LiveFrameUpdate:
    """Snapshot pushed to the live GUI each frame."""

    frame_bgr: np.ndarray          # frame with landmarks drawn
    model_name: str
    video_name: str
    exercise_name: str
    frame_idx: int
    total_frames: int              # 0 if unknown
    mean_fps: float
    detection_rate: float
    total_reps: int
    valid_reps: int
    job_idx: int                   # 1-based index of current (model × video) job
    total_jobs: int


LiveCallback = Callable[[LiveFrameUpdate], None]


@dataclass(slots=True)
class RepRecord:
    """Result of a single repetition."""

    rep_index: int
    is_valid: bool
    failed_rules: list[str]
    passed_rules: list[str]
    rule_values: dict[str, float]  # rule_name -> metric value
    frame_count: int


@dataclass(slots=True)
class FrameRecord:
    """Per-frame timing and detection data."""

    frame_index: int
    latency_ms: float
    landmarks_detected: bool
    primary_angle: float | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Full benchmark output for one (model, video, exercise) combination."""

    model_name: str
    video_path: str
    exercise_name: str
    frames: list[FrameRecord] = field(default_factory=list)
    reps: list[RepRecord] = field(default_factory=list)

    # ------ derived properties ------

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def detected_frames(self) -> int:
        return sum(1 for f in self.frames if f.landmarks_detected)

    @property
    def detection_rate(self) -> float:
        return self.detected_frames / self.total_frames if self.total_frames else 0.0

    @property
    def latencies_ms(self) -> list[float]:
        return [f.latency_ms for f in self.frames]

    @property
    def mean_fps(self) -> float:
        lats = self.latencies_ms
        if not lats:
            return 0.0
        mean_lat = sum(lats) / len(lats)
        return 1000.0 / mean_lat if mean_lat > 0 else 0.0

    @property
    def median_fps(self) -> float:
        lats = sorted(self.latencies_ms)
        if not lats:
            return 0.0
        mid = len(lats) // 2
        median_lat = lats[mid] if len(lats) % 2 else (lats[mid - 1] + lats[mid]) / 2
        return 1000.0 / median_lat if median_lat > 0 else 0.0

    @property
    def p95_latency_ms(self) -> float:
        lats = sorted(self.latencies_ms)
        if not lats:
            return 0.0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def valid_reps(self) -> int:
        return sum(1 for r in self.reps if r.is_valid)

    @property
    def invalid_reps(self) -> int:
        return self.total_reps - self.valid_reps


def run_benchmark(
    model_name: str,
    video_path: Path,
    exercise_name: str,
    *,
    resize: tuple[int, int] = (640, 480),
    progress_callback: object = None,
    live_callback: LiveCallback | None = None,
    job_idx: int = 1,
    total_jobs: int = 1,
    smoother: AdaptiveSmoother | None = None,
) -> BenchmarkResult:
    """Run a single benchmark: one model × one video × one exercise."""

    result = BenchmarkResult(
        model_name=model_name,
        video_path=str(video_path),
        exercise_name=exercise_name,
    )

    # Build exercise and tracker
    exercise_cls = EXERCISE_REGISTRY.get(exercise_name, EXERCISE_REGISTRY["curl"])
    exercise: Exercise = exercise_cls()
    tracker = RepTracker(
        start_phase=exercise.start_phase,
        min_rom=20.0,
        smoother=smoother if smoother is not None else AdaptiveSmoother(),
    )

    # Build model
    model: PoseModel = create_pose_model(model_name)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep_counter = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = letterbox_resize(frame, resize[0], resize[1])
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- timed section ----
            t0 = time.perf_counter()
            raw_result = model.process_image(image_rgb)
            landmarks = model.extract_landmarks(raw_result)
            t1 = time.perf_counter()
            # -----------------------

            latency_ms = (t1 - t0) * 1000.0
            detected = landmarks is not None

            # ---- live preview ----
            if live_callback is not None:
                frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                model.draw_landmarks(frame_bgr, raw_result)
                lats_so_far = [f.latency_ms for f in result.frames]
                mean_lat = sum(lats_so_far) / len(lats_so_far) if lats_so_far else latency_ms
                fps_so_far = 1000.0 / mean_lat if mean_lat > 0 else 0.0
                det_so_far = (
                    (result.detected_frames + int(detected)) / (len(result.frames) + 1)
                )
                live_callback(LiveFrameUpdate(
                    frame_bgr=frame_bgr,
                    model_name=model_name,
                    video_name=video_path.name,
                    exercise_name=exercise_name,
                    frame_idx=frame_idx,
                    total_frames=total_video_frames,
                    mean_fps=fps_so_far,
                    detection_rate=det_so_far,
                    total_reps=rep_counter,
                    valid_reps=sum(1 for r in result.reps if r.is_valid),
                    job_idx=job_idx,
                    total_jobs=total_jobs,
                ))

            result.frames.append(
                FrameRecord(
                    frame_index=frame_idx,
                    latency_ms=latency_ms,
                    landmarks_detected=detected,
                    primary_angle=None,
                )
            )

            # Feed tracker
            if detected:
                joints = [
                    read_landmark(j, landmarks)
                    for j in exercise.primary_joints
                ]
                if all(j is not None for j in joints):
                    angle = calculate_angle_3d(list(joints))
                    result.frames[-1].primary_angle = angle
                    
                    timestamp = frame_idx / 30.0  # synthetic timestamp
                    rep_completed = tracker.add_frame(angle, timestamp, landmarks=landmarks)
                    if rep_completed:
                        rep_data = tracker.extract_rep()
                        if rep_data:
                            is_valid, _reason = exercise.apply_rules(rep_data)
                            failed = [
                                r.rule_name
                                for r in exercise.rules
                                if not r.apply(rep_data)
                            ]
                            failed_set = set(failed)
                            passed = [
                                r.rule_name
                                for r in exercise.rules
                                if r.rule_name not in failed_set
                            ]

                            # Each rule reports the scalar its own threshold is
                            # compared against (rule.reduce); no isinstance soup.
                            rule_values = {}
                            for rule in exercise.rules:
                                value = rule.reduce(rep_data)
                                if value is not None:
                                    rule_values[rule.rule_name] = value

                            result.reps.append(
                                RepRecord(
                                    rep_index=rep_counter,
                                    is_valid=is_valid,
                                    failed_rules=failed,
                                    passed_rules=passed,
                                    rule_values=rule_values,
                                    frame_count=len(rep_data),
                                )
                            )
                            rep_counter += 1

            frame_idx += 1

            if progress_callback and total_video_frames > 0:
                progress_callback(frame_idx, total_video_frames)
    finally:
        cap.release()
        model.release()

    return result


def run_all_benchmarks(
    model_names: list[str],
    video_paths: list[Path],
    exercise_name: str,
    *,
    resize: tuple[int, int] = (640, 480),
    verbose: bool = True,
    live_callback: LiveCallback | None = None,
    smoother_factory: Callable[[], AdaptiveSmoother] | None = None,
) -> list[BenchmarkResult]:
    """Run benchmarks for every (model, video) combination."""

    results: list[BenchmarkResult] = []
    total_jobs = len(model_names) * len(video_paths)
    job_idx = 0

    for video_path in video_paths:
        for model_name in model_names:
            job_idx += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Job {job_idx}/{total_jobs}")
                print(f"  Model: {model_name}")
                print(f"  Video: {video_path.name}")
                print(f"  Exercise: {exercise_name}")
                print(f"{'='*60}")

            def _progress(current: int, total: int) -> None:
                if current % 50 == 0 or current == total:
                    pct = current / total * 100
                    print(f"  [{current}/{total}] {pct:.0f}%", end="\r")

            result = run_benchmark(
                model_name,
                video_path,
                exercise_name,
                resize=resize,
                progress_callback=_progress if verbose else None,
                live_callback=live_callback,
                job_idx=job_idx,
                total_jobs=total_jobs,
                # Each job gets a fresh smoother so one video's FPS estimate
                # does not bleed into the next.
                smoother=smoother_factory() if smoother_factory else None,
            )

            if verbose:
                print(f"\n  → {result.total_frames} frames, "
                      f"{result.mean_fps:.1f} FPS, "
                      f"{result.detection_rate:.1%} detection, "
                      f"{result.total_reps} reps "
                      f"({result.valid_reps} valid)")

            results.append(result)

    return results
