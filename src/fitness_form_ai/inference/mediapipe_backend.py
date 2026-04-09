from __future__ import annotations

from typing import Any

import numpy as np

from fitness_form_ai.inference.base import LandmarkPoint, PoseModel


class MediaPipeModel(PoseModel):
    def __init__(self, complexity: int = 1) -> None:
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            self.pose.process(dummy_image)
        except Exception as exc:
            self.release()
            raise RuntimeError(
                f"MediaPipe Pose initialization failed for complexity {complexity}: {exc}"
            ) from exc

    def process_image(self, image: np.ndarray) -> Any:
        if self.pose is None:
            raise RuntimeError("MediaPipe Pose model is not initialized.")
        return self.pose.process(image)

    def extract_landmarks(self, results: Any) -> dict[str, LandmarkPoint] | None:
        if results is None or not getattr(results, "pose_world_landmarks", None):
            return None

        landmarks_dict: dict[str, LandmarkPoint] = {}
        for landmark_name in self.mp_pose.PoseLandmark:
            index = landmark_name.value
            landmark = results.pose_world_landmarks.landmark[index]
            landmarks_dict[landmark_name.name] = LandmarkPoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
            )
        return landmarks_dict

    def draw_landmarks(self, image: np.ndarray, results: Any) -> None:
        if results is not None and getattr(results, "pose_landmarks", None):
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

    def release(self) -> None:
        if getattr(self, "pose", None) is not None:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None
