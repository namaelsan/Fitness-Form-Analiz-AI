from collections.abc import Callable

from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.exercises import (
    Deadlift,
    LateralRaise,
    OneArmDumbbellCurl,
    ShoulderPress,
    Squat,
)
from fitness_form_ai.app.config import SUPPORTED_MODELS
from fitness_form_ai.inference.factory import create_pose_model

ExerciseFactory = Callable[[], Exercise]

# Pose-model registry. "metrabs" is the direct single-image 3D estimator and
# sits alongside the MediaPipe (3D) backends and the MoveNet / YOLOv8 2D
# baselines. The canonical list lives in app.config.SUPPORTED_MODELS; it is
# re-exported here so the catalog is the one place to look up everything the
# system can instantiate.
MODEL_REGISTRY: tuple[str, ...] = SUPPORTED_MODELS

EXERCISE_REGISTRY: dict[str, ExerciseFactory] = {
    "curl": OneArmDumbbellCurl,
    "squat": Squat,
    "deadlift": Deadlift,
    "shoulder_press": ShoulderPress,
    "lateral_raise": LateralRaise,
}

MODEL_FACTORY = create_pose_model
