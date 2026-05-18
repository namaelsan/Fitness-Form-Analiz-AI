"""Batch-process videos with each pose model, measuring FPS and form analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fitness_form_ai.app.catalog import EXERCISE_REGISTRY
from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.tracker import RepTracker
from fitness_form_ai.inference.base import PoseModel
from fitness_form_ai.inference.factory import create_pose_model
from fitness_form_ai.utils.geometry import calculate_angle_3d
from fitness_form_ai.utils.landmarks import read_landmark


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
    tracker = RepTracker(start_phase=exercise.start_phase, min_rom=20.0)

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

            frame = cv2.resize(frame, resize)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- timed section ----
            t0 = time.perf_counter()
            raw_result = model.process_image(image_rgb)
            landmarks = model.extract_landmarks(raw_result)
            t1 = time.perf_counter()
            # -----------------------

            latency_ms = (t1 - t0) * 1000.0
            detected = landmarks is not None

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
                            all_rules = [r.rule_name for r in exercise.rules]
                            failed = [
                                r.rule_name
                                for r in exercise.rules
                                if not r.apply(rep_data)
                            ]
                            passed = [n for n in all_rules if n not in failed]
                            
                            # Extract continuous values for each rule over the rep
                            # Assuming that for values, returning max/min/average over the rep might be needed,
                            # but simple approach: get the value from the last frame or let the rule compute it
                            # Actually, `get_current_value(landmarks)` is on Rule. We will just compute the value at the end of the rep or similar, 
                            # or use the metric logic if possible.
                            # For simplicity, we can get metric values from the rep_data.
                            # The easiest way is to use `rule.get_current_value(rep_data[-1].landmarks)` for now, OR for range rules just store min/max?
                            rule_vals = {}
                            for rule in exercise.rules:
                                if hasattr(rule, 'metric'):
                                    vals = rule.metric.series(rep_data)
                                    if vals:
                                        # Depending on rule type, we usually care about the extreme value
                                        from fitness_form_ai.domain.rules import MaxValueRule, MinValueRule, RangeRule, MaxDepthRule, StabilityRule
                                        if isinstance(rule, (MaxValueRule, MinValueRule, RangeRule, MaxDepthRule)):
                                            # store max for MaxValue, min for MaxDepth, etc. Just storing the values range
                                            # Let's just track the minimum and maximum observed in the rep, or a tuple. 
                                            # We will track the extreme value relevant to the rule, or just average for simplicity.
                                            pass
                                # Actually, rule.describe_current takes context. We can just use the value from rep_data.
                                # Let's just calculate a simple metric. For now, since rules implement apply(rep_data), we might need
                                # a way to get the numerical result out of them.
                            
                            rule_values = {}
                            for rule in exercise.rules:
                                if hasattr(rule, 'metric'):
                                    series = rule.metric.series(rep_data)
                                    if series:
                                        # If rule is checking max value, store max of series
                                        rule_values[rule.rule_name] = sum(series)/len(series) # average for now
                                        from fitness_form_ai.domain.rules import MaxValueRule, MinValueRule, RangeRule, MaxDepthRule, StabilityRule
                                        if isinstance(rule, MaxValueRule):
                                            rule_values[rule.rule_name] = max(series)
                                        elif isinstance(rule, MinValueRule) or isinstance(rule, MaxDepthRule):
                                            rule_values[rule.rule_name] = min(series)
                                        elif isinstance(rule, RangeRule):
                                            rule_values[rule.rule_name] = max(series) if max(series) > rule.angle_range[1] else min(series) # Pick the worst bound
                                        elif isinstance(rule, StabilityRule):
                                            rule_values[rule.rule_name] = max(series) - min(series)
                                elif "Knee Valgus" in rule.rule_name or hasattr(rule, "min_ratio"):
                                    # KneeValgusRule
                                    # ratio series
                                    ratios = rule._ratio_series(rep_data)
                                    if ratios:
                                        rule_values[rule.rule_name] = min(ratios)

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
) -> list[BenchmarkResult]:
    """Run benchmarks for every (model, video) combination."""

    results: list[BenchmarkResult] = []

    for video_path in video_paths:
        for model_name in model_names:
            if verbose:
                print(f"\n{'='*60}")
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
            )

            if verbose:
                print(f"\n  → {result.total_frames} frames, "
                      f"{result.mean_fps:.1f} FPS, "
                      f"{result.detection_rate:.1%} detection, "
                      f"{result.total_reps} reps "
                      f"({result.valid_reps} valid)")

            results.append(result)

    return results
