from fitness_form_ai.domain.exercise import Exercise
from fitness_form_ai.domain.metrics import (
    AngleMetric,
    JointPairAxisDistanceMetric,
    SegmentOrientationMetric,
    SymmetryMetric,
    TrackerAngleMetric,
)
from fitness_form_ai.domain.rules import (
    FloorRule,
    KneeValgusRule,
    MaxDepthRule,
    MaxValueRule,
    MinValueRule,
    RangeRule,
    StabilityRule,
    TempoRule,
)


class OneArmDumbbellCurl(Exercise):
    def __init__(self) -> None:
        elbow_angle = TrackerAngleMetric("Elbow flexion")
        upper_arm_angle = AngleMetric(
            "Upper arm angle",
            ["RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"],
        )
        super().__init__(
            name="One Arm Dumbbell Curl",
            rules=[
                MaxDepthRule("Full contraction", elbow_angle, 75),
                MinValueRule("Full extension", elbow_angle, 155),
                StabilityRule("Upper Arm Still", upper_arm_angle, 40),
                TempoRule("Controlled tempo", max_speed=250, min_duration=1.0),
            ],
            primary_joints=["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
            start_phase="concentric",
        )


class Squat(Exercise):
    def __init__(self) -> None:
        knee_angle = TrackerAngleMetric("Knee flexion")
        torso_angle = AngleMetric(
            "Torso stack",
            ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE"],
        )
        knee_width = JointPairAxisDistanceMetric(
            "Knee width",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "x",
        )
        ankle_width = JointPairAxisDistanceMetric(
            "Ankle width",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "x",
        )
        super().__init__(
            name="Squat",
            rules=[
                MaxDepthRule("Depth reached", knee_angle, 85),
                MinValueRule("Standing lockout", knee_angle, 160),
                RangeRule("Torso stays stacked", torso_angle, (145, 180)),
                KneeValgusRule("Knees track over feet", knee_width, ankle_width, min_ratio=0.55),
                TempoRule("Controlled descent", max_speed=250, min_duration=1.0),
            ],
            primary_joints=["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
            start_phase="eccentric",
        )


class Deadlift(Exercise):
    def __init__(self) -> None:
        hip_angle = TrackerAngleMetric("Hip extension")
        spine_angle = AngleMetric(
            "Spine angle",
            ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
        )
        bar_path = JointPairAxisDistanceMetric(
            "Bar proximity",
            "LEFT_WRIST",
            "LEFT_ANKLE",
            "x",
        )
        shoulder_hip_stack = SegmentOrientationMetric(
            "Torso orientation",
            "LEFT_HIP",
            "LEFT_SHOULDER",
            "vertical",
        )
        super().__init__(
            name="Deadlift",
            rules=[
                MaxDepthRule("Hip hinge depth", hip_angle, 90),
                RangeRule("Neutral spine", spine_angle, (150, 180)),
                MaxValueRule("Bar stays close", bar_path, 0.22),
                RangeRule("Torso angle controlled", shoulder_hip_stack, (0, 65)),
                TempoRule("No jerking", max_speed=150, min_duration=1.0),
            ],
            primary_joints=["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
            start_phase="concentric",
        )


class ShoulderPress(Exercise):
    def __init__(self) -> None:
        elbow_angle = TrackerAngleMetric("Press angle")
        arm_stack = SegmentOrientationMetric(
            "Arm stack",
            "RIGHT_ELBOW",
            "RIGHT_WRIST",
            "vertical",
        )
        trunk_line = AngleMetric(
            "Trunk line",
            ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_ANKLE"],
        )
        left_arm = SegmentOrientationMetric(
            "Left arm",
            "LEFT_ELBOW",
            "LEFT_WRIST",
            "vertical",
        )
        right_arm = SegmentOrientationMetric(
            "Right arm",
            "RIGHT_ELBOW",
            "RIGHT_WRIST",
            "vertical",
        )
        symmetry = SymmetryMetric("Arm symmetry", left_arm, right_arm)
        super().__init__(
            name="Shoulder Press",
            rules=[
                MinValueRule("Full press extension", elbow_angle, 160),
                RangeRule("Wrists stay stacked", arm_stack, (0, 30)),
                RangeRule("No back overextension", trunk_line, (155, 180)),
                MaxValueRule("Press stays symmetric", symmetry, 18),
                TempoRule("Controlled press", max_speed=150, min_duration=1.0),
            ],
            primary_joints=["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
            start_phase="concentric",
        )


class LateralRaise(Exercise):
    def __init__(self) -> None:
        shoulder_angle = TrackerAngleMetric("Shoulder raise")
        torso_angle = AngleMetric(
            "Torso stability",
            ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_ANKLE"],
        )
        shrug_metric = JointPairAxisDistanceMetric(
            "Shoulder shrug",
            "RIGHT_SHOULDER",
            "RIGHT_EAR",
            "y",
        )
        elbow_angle = AngleMetric(
            "Soft elbow",
            ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        )
        super().__init__(
            name="Lateral Raise",
            rules=[
                RangeRule("Top height stays clean", shoulder_angle, (35, 110)),
                StabilityRule("Torso stays still", torso_angle, 18),
                FloorRule("Shoulder stays depressed", shrug_metric, 0.08),
                RangeRule("Elbow bend stays soft", elbow_angle, (150, 178)),
                TempoRule("No swinging", max_speed=120, min_duration=1.0),
            ],
            primary_joints=["RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"],
            start_phase="concentric",
        )
