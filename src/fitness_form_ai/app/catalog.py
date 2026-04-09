from collections.abc import Callable

from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.exercises import (
    Deadlift,
    LateralRaise,
    OneArmDumbbellCurl,
    PushUp,
    ShoulderPress,
    Squat,
)
from fitness_form_ai.inference.factory import create_pose_model

ExerciseFactory = Callable[[], Exercise]

EXERCISE_REGISTRY: dict[str, ExerciseFactory] = {
    "curl": OneArmDumbbellCurl,
    "squat": Squat,
    "deadlift": Deadlift,
    "shoulder_press": ShoulderPress,
    "lateral_raise": LateralRaise,
    "pushup": PushUp,
}

MODEL_FACTORY = create_pose_model
