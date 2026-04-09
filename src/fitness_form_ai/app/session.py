from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fitness_form_ai.app.catalog import EXERCISE_REGISTRY, MODEL_FACTORY
from fitness_form_ai.app.config import TrackingConfig
from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.rules import RuleContext
from fitness_form_ai.domain.tracker import RepTracker
from fitness_form_ai.inference.base import PoseModel
from fitness_form_ai.utils.geometry import calculate_angle_3d
from fitness_form_ai.utils.landmarks import read_landmark


@dataclass(slots=True)
class FrameOutcome:
    image_bgr: np.ndarray | None
    angle: float | None
    tracker_state: str
    rule_states: list[tuple[str, str]]
    valid_reps: int
    total_reps: int
    message: str


class TrackingSession:
    def __init__(
        self,
        exercise_name: str,
        model_name: str,
        video_path: Path | None = None,
        tracking_config: TrackingConfig | None = None,
    ) -> None:
        self.exercise_name = exercise_name
        self.model_name = model_name
        self.video_path = video_path
        self.config = tracking_config or TrackingConfig()

        self.video_capture: cv2.VideoCapture | None = None
        self.model: PoseModel | None = None
        self.exercise: Exercise | None = None
        self.tracker: RepTracker | None = None

        self.valid_reps = 0
        self.total_reps = 0
        self.last_message = ""
        self.video_fps = 30.0
        self.video_start_time = 0.0
        self.is_playing = True

    def initialize(self) -> None:
        self._release_video_capture()
        self._release_model()

        self.video_capture = self._open_capture(self.video_path)
        self.exercise = EXERCISE_REGISTRY.get(
            self.exercise_name,
            EXERCISE_REGISTRY["curl"],
        )()
        self.tracker = RepTracker(
            start_phase=self.exercise.start_phase,
            min_rom=self.config.min_rom,
        )
        self.valid_reps = 0
        self.total_reps = 0
        self.last_message = ""
        self.is_playing = True

        try:
            self.model = MODEL_FACTORY(self.model_name)
        except Exception:
            if self.model_name == "mediapipe-full":
                self.model = None
                self.last_message = "CRITICAL: Default model failed"
                return

            self.model_name = "mediapipe-full"
            self.model = MODEL_FACTORY(self.model_name)
            self.last_message = "Model fallback triggered"

    def reset_video(self) -> None:
        if self.exercise is None:
            return

        self.tracker = RepTracker(
            start_phase=self.exercise.start_phase,
            min_rom=self.config.min_rom,
        )
        self.valid_reps = 0
        self.total_reps = 0

        if self.video_path is None:
            self.last_message = "Tracking Reset"
            return

        self._release_video_capture()
        self.video_capture = self._open_capture(self.video_path)
        self.is_playing = True
        self.last_message = "Video Reset"

    def set_video_source(self, video_path: Path | None) -> None:
        self.video_path = video_path
        self.initialize()

    def set_exercise(self, exercise_name: str) -> None:
        self.exercise_name = exercise_name
        self.initialize()

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.initialize()

    def toggle_playback(self) -> bool:
        if self.video_path is None or self.video_capture is None:
            return True

        self.is_playing = not self.is_playing
        if self.is_playing:
            current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.video_start_time = time.time() - (current_frame / self.video_fps)
        return self.is_playing

    def process_next_frame(self) -> FrameOutcome:
        if self.model is None or self.video_capture is None or self.exercise is None or self.tracker is None:
            return self._empty_outcome()

        if not self.video_capture.isOpened():
            return self._empty_outcome()

        if self.video_path is not None:
            self._sync_video_position()
            if not self.is_playing:
                return self._empty_outcome()

        ok, frame = self.video_capture.read()
        if not ok:
            if self.video_path is None:
                return self._empty_outcome()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_start_time = time.time()
            return self._empty_outcome()

        frame = cv2.resize(
            frame,
            (self.config.texture_width, self.config.texture_height),
        )
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.process_image(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        landmarks = self.model.extract_landmarks(result)

        angle: float | None = None
        rule_states: list[tuple[str, str]] = []

        if landmarks:
            joints = [read_landmark(joint_name, landmarks) for joint_name in self.exercise.primary_joints]
            if all(joint is not None for joint in joints):
                angle = calculate_angle_3d(list(joints))
                timestamp = time.time()
                rep_completed = self.tracker.add_frame(angle, timestamp, landmarks=landmarks)
                current_frame = RepFrame(
                    angle=angle,
                    timestamp=timestamp,
                    velocity=self.tracker.current_velocity,
                    landmarks=landmarks,
                )
                if rep_completed:
                    self._finalize_rep()
                rule_states = self.exercise.get_rule_states(
                    RuleContext(
                        landmarks=landmarks,
                        current_frame=current_frame,
                        rep_duration=self.tracker.current_rep_duration,
                    )
                )

        self.model.draw_landmarks(image_bgr, result)
        return FrameOutcome(
            image_bgr=image_bgr,
            angle=angle,
            tracker_state=self.tracker.state,
            rule_states=rule_states,
            valid_reps=self.valid_reps,
            total_reps=self.total_reps,
            message=self.last_message,
        )

    def render_texture_data(self, image_bgr: np.ndarray | None) -> np.ndarray | None:
        if image_bgr is None:
            return None
        image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
        return (image_rgba.ravel() / 255.0).astype(np.float32)

    def cleanup(self) -> None:
        self._release_video_capture()
        self._release_model()

    def _empty_outcome(self) -> FrameOutcome:
        tracker_state = self.tracker.state if self.tracker else "IDLE"
        return FrameOutcome(
            image_bgr=None,
            angle=None,
            tracker_state=tracker_state,
            rule_states=[],
            valid_reps=self.valid_reps,
            total_reps=self.total_reps,
            message=self.last_message,
        )

    def _finalize_rep(self) -> None:
        self.total_reps += 1
        rep_data = self.tracker.extract_rep()
        if not rep_data or self.exercise is None:
            return

        is_valid, reason = self.exercise.apply_rules(rep_data)
        if is_valid:
            self.valid_reps += 1
            self.last_message = "VALID REP!"
        else:
            self.last_message = f"INVALID: {reason}"

    def _sync_video_position(self) -> None:
        if self.video_capture is None or self.video_path is None or not self.is_playing:
            return
        elapsed = time.time() - self.video_start_time
        target_frame = int(elapsed * self.video_fps)
        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if target_frame > current_frame:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def _open_capture(self, video_path: Path | None) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(video_path) if video_path else 0)
        self.video_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_start_time = time.time()
        return capture

    def _release_video_capture(self) -> None:
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

    def _release_model(self) -> None:
        if self.model is not None:
            try:
                self.model.release()
            except Exception:
                pass
            self.model = None
