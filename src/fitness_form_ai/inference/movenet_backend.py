from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fitness_form_ai.inference.base import LandmarkPoint, PoseModel


class MoveNetModel(PoseModel):
    # MoveNet is a 2D model: extract_landmarks sets z=0.0 for all joints.
    # PA-MPJPE results against 3D mocap ground truth reflect 2D-projected alignment only.
    provides_3d_landmarks = False

    def __init__(self, variant: str = "lightning") -> None:
        import tensorflow as tf

        self.variant = variant
        
        # Determine URLs and shapes
        if variant == "thunder":
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            self.input_size = 256
        else:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
            self.input_size = 192

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"movenet_{variant}.tflite"

        if not model_path.exists():
            print(f"Downloading MoveNet {variant} to {model_path}...")
            urllib.request.urlretrieve(url, str(model_path))

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

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
        # MoveNet requires a [1, input_size, input_size, 3] float32 or int32 depending on model.
        # Float16 model usually takes int32 or float32. We'll use int32.
        import tensorflow as tf

        # Remember original shape so extract_landmarks can de-pad coordinates.
        self._last_image_shape: tuple[int, int, int] = image.shape

        # Create a copy and resize
        image_resized = tf.image.resize_with_pad(image, self.input_size, self.input_size)
        input_dtype = self.input_details[0]['dtype']
        input_image = tf.cast(image_resized, dtype=input_dtype)
        input_image = tf.expand_dims(input_image, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

        return keypoints_with_scores

    def extract_landmarks(self, results: Any) -> dict[str, LandmarkPoint] | None:
        # results shape: [1, 1, 17, 3] -> (y, x, score) normalized to padded square [0, 1]
        keypoints = results[0, 0, :, :]
        landmarks: dict[str, LandmarkPoint] = {}

        # De-pad normalized coordinates so they are relative to the original image,
        # matching the convention used by MediaPipe.
        image_shape = getattr(self, "_last_image_shape", None)
        if image_shape is not None:
            h, w = image_shape[:2]
            S = self.input_size
            if w >= h:
                scale = S / w
                pad_y = (S - h * scale) / 2
                pad_x = 0.0
            else:
                scale = S / h
                pad_x = (S - w * scale) / 2
                pad_y = 0.0
        else:
            h = w = S = scale = 1
            pad_x = pad_y = 0.0

        for index, name in self.keypoint_mapping.items():
            if index >= len(keypoints):
                continue
            y, x, confidence = keypoints[index]
            if confidence <= 0.3:
                continue
            if image_shape is not None:
                # Both axes divided by the same factor (scale * w) so x and y
                # remain on a consistent scale. Dividing y by (scale * h) instead
                # would give different units per axis and distort angle calculations
                # on non-square images.
                x_out = (float(x) * S - pad_x) / (scale * w)
                y_out = (float(y) * S - pad_y) / (scale * w)
            else:
                x_out, y_out = float(x), float(y)
            landmarks[name] = LandmarkPoint(x=x_out, y=y_out, z=0.0)

        return landmarks or None

    def _keypoint_to_image_coords(self, y_n: float, x_n: float, h: int, w: int) -> tuple[int, int]:
        """Inverse of tf.image.resize_with_pad: map normalized padded-square coords to original image pixels."""
        S = self.input_size
        if w >= h:
            scale = S / w
            pad_y = (S - h * scale) / 2
            pad_x = 0.0
        else:
            scale = S / h
            pad_x = (S - w * scale) / 2
            pad_y = 0.0
        px = int((x_n * S - pad_x) / scale)
        py = int((y_n * S - pad_y) / scale)
        return px, py

    def draw_landmarks(self, image: np.ndarray, results: Any) -> None:
        keypoints = results[0, 0, :, :]
        h, w, _ = image.shape

        # Define COCO connections
        connections = [
            (5, 7), (7, 9),      # left arm
            (6, 8), (8, 10),     # right arm
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
            (5, 6), (11, 12),    # torso top/bottom
            (5, 11), (6, 12),    # torso left/right
            (0, 1), (1, 3),      # left face
            (0, 2), (2, 4)       # right face
        ]

        # Draw connections
        for p1, p2 in connections:
            if p1 < len(keypoints) and p2 < len(keypoints):
                y1, x1, conf1 = keypoints[p1]
                y2, x2, conf2 = keypoints[p2]
                if conf1 > 0.3 and conf2 > 0.3:
                    pt1 = self._keypoint_to_image_coords(y1, x1, h, w)
                    pt2 = self._keypoint_to_image_coords(y2, x2, h, w)
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for index in range(len(keypoints)):
            y, x, conf = keypoints[index]
            if conf > 0.3:
                pt = self._keypoint_to_image_coords(y, x, h, w)
                cv2.circle(image, pt, 4, (0, 0, 255), -1)

    def release(self) -> None:
        self.interpreter = None
