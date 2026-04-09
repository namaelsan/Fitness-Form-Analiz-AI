from fitness_form_ai.inference.base import PoseModel
from fitness_form_ai.inference.mediapipe_backend import MediaPipeModel
from fitness_form_ai.inference.yolov8_backend import YOLOv8Model


def create_pose_model(model_name: str) -> PoseModel:
    if model_name == "mediapipe-lite":
        return MediaPipeModel(complexity=0)
    if model_name == "mediapipe-heavy":
        return MediaPipeModel(complexity=2)
    if model_name == "yolov8":
        return YOLOv8Model()
    return MediaPipeModel(complexity=1)
