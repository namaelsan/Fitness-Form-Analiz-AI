from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from fitness_form_ai.inference.base import LandmarkPoint, PoseModel


class YOLOv8Model(PoseModel):
    def __init__(self, model_version: str = "yolov8n-pose.pt") -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_version)
        self.keypoint_mapping = {
            0: "NOSE",
            1: "LEFT_EYE",
            2: "RIGHT_EYE",
            3: "LEFT_EAR",
            4: "RIGHT_EAR",
            5: "LEFT_SHOULDER",
            6: "RIGHT_SHOULDER",
            7: "LEFT_ELBOW",
            8: "RIGHT_ELBOW",
            9: "LEFT_WRIST",
            10: "RIGHT_WRIST",
            11: "LEFT_HIP",
            12: "RIGHT_HIP",
            13: "LEFT_KNEE",
            14: "RIGHT_KNEE",
            15: "LEFT_ANKLE",
            16: "RIGHT_ANKLE",
        }

    def process_image(self, image: np.ndarray) -> Any:
        results = self.model(image, conf=0.6, verbose=False)
        return results[0]

    def extract_landmarks(self, results: Any) -> dict[str, LandmarkPoint] | None:
        if results.keypoints is None or len(results.keypoints) == 0:
            return None

        keypoints = results.keypoints.data[0].cpu().numpy()
        landmarks: dict[str, LandmarkPoint] = {}
        for index, name in self.keypoint_mapping.items():
            if index >= len(keypoints):
                continue
            x, y, confidence = keypoints[index]
            if confidence <= 0.6:
                continue
            landmarks[name] = LandmarkPoint(x=float(x), y=float(y), z=0.0)

        return landmarks or None

    def draw_landmarks(self, image: np.ndarray, results: Any) -> None:
        annotated_frame = results.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        np.copyto(image, annotated_frame_rgb)
