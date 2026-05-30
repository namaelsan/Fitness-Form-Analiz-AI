from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.metrics import AngleMetric, SymmetryMetric, TrackerAngleMetric
from fitness_form_ai.domain.rules import (
    FloorRule,
    KneeValgusRule,
    MaxValueRule,
    RangeRule,
    StabilityRule,
    TempoRule,
)
from fitness_form_ai.inference.base import LandmarkPoint


def test_angle_rule_uses_frame_angles_without_landmarks() -> None:
    rule = RangeRule("Range", TrackerAngleMetric("Tracker"), (30, 120))
    rep = [
        RepFrame(angle=45, timestamp=0.0),
        RepFrame(angle=90, timestamp=0.1),
    ]
    assert rule.apply(rep) is True


def test_angle_rule_uses_landmarks_when_available() -> None:
    rule = RangeRule("Elbow", AngleMetric("Elbow", ["A", "B", "C"]), (80, 100))
    landmarks = {
        "A": LandmarkPoint(x=0, y=1, z=0),
        "B": LandmarkPoint(x=0, y=0, z=0),
        "C": LandmarkPoint(x=1, y=0, z=0),
    }
    rep = [RepFrame(angle=0, timestamp=0.0, landmarks=landmarks)]
    assert rule.apply(rep) is True


def test_stability_rule_rejects_large_joint_drift() -> None:
    metric = AngleMetric("Upper arm", ["A", "B", "C"])
    rule = StabilityRule("Still", metric, max_delta=5)
    rep = [
        RepFrame(
            angle=0,
            timestamp=0.0,
            landmarks={
                "A": LandmarkPoint(x=0, y=1, z=0),
                "B": LandmarkPoint(x=0, y=0, z=0),
                "C": LandmarkPoint(x=1, y=0, z=0),
            },
        ),
        RepFrame(
            angle=0,
            timestamp=0.1,
            landmarks={
                "A": LandmarkPoint(x=0, y=1, z=0),
                "B": LandmarkPoint(x=0, y=0, z=0),
                "C": LandmarkPoint(x=0.3, y=0.7, z=0),
            },
        ),
    ]
    assert rule.apply(rep) is False


def test_tempo_rule_rejects_short_fast_rep() -> None:
    rule = TempoRule("Tempo", max_speed=50, min_duration=1.0)
    rep = [
        RepFrame(angle=90, timestamp=0.0, velocity=0),
        RepFrame(angle=120, timestamp=0.4, velocity=75),
    ]
    assert rule.apply(rep) is False


def test_symmetry_rule_accepts_balanced_values() -> None:
    left = AngleMetric("Left", ["A", "B", "C"])
    right = AngleMetric("Right", ["D", "E", "F"])
    metric = SymmetryMetric("Symmetry", left, right)
    rule = MaxValueRule("Balanced", metric, maximum=5)
    landmarks = {
        "A": LandmarkPoint(x=0, y=1, z=0),
        "B": LandmarkPoint(x=0, y=0, z=0),
        "C": LandmarkPoint(x=1, y=0, z=0),
        "D": LandmarkPoint(x=0, y=1, z=0),
        "E": LandmarkPoint(x=0, y=0, z=0),
        "F": LandmarkPoint(x=1, y=0, z=0),
    }
    rep = [RepFrame(angle=0, timestamp=0.0, landmarks=landmarks)]
    assert rule.apply(rep) is True


def test_floor_rule_passes_when_minimum_stays_above_threshold() -> None:
    rule = FloorRule("Floor", TrackerAngleMetric("Tracker"), minimum=0.08)
    passing = [RepFrame(angle=0.12, timestamp=0.0), RepFrame(angle=0.10, timestamp=0.1)]
    failing = [RepFrame(angle=0.12, timestamp=0.0), RepFrame(angle=0.05, timestamp=0.1)]
    assert rule.apply(passing) is True
    assert rule.apply(failing) is False


def test_knee_valgus_rule_fails_only_when_knees_collapse_inward() -> None:
    from fitness_form_ai.domain.metrics import JointPairAxisDistanceMetric

    knee_width = JointPairAxisDistanceMetric("Knee width", "LEFT_KNEE", "RIGHT_KNEE", "x")
    stance_width = JointPairAxisDistanceMetric("Stance width", "LEFT_ANKLE", "RIGHT_ANKLE", "x")
    rule = KneeValgusRule("Knees track over feet", knee_width, stance_width, min_ratio=0.75)

    inward_collapse = [
        RepFrame(
            angle=0,
            timestamp=0.0,
            landmarks={
                "LEFT_KNEE": LandmarkPoint(x=0.45, y=0, z=0),
                "RIGHT_KNEE": LandmarkPoint(x=0.55, y=0, z=0),
                "LEFT_ANKLE": LandmarkPoint(x=0.0, y=0, z=0),
                "RIGHT_ANKLE": LandmarkPoint(x=1.0, y=0, z=0),
            },
        ),
    ]
    knees_wider_than_feet = [
        RepFrame(
            angle=0,
            timestamp=0.0,
            landmarks={
                "LEFT_KNEE": LandmarkPoint(x=-0.1, y=0, z=0),
                "RIGHT_KNEE": LandmarkPoint(x=1.1, y=0, z=0),
                "LEFT_ANKLE": LandmarkPoint(x=0.0, y=0, z=0),
                "RIGHT_ANKLE": LandmarkPoint(x=1.0, y=0, z=0),
            },
        ),
    ]

    assert rule.apply(inward_collapse) is False
    assert rule.apply(knees_wider_than_feet) is True


def test_reduce_reports_threshold_scalar_per_rule_type() -> None:
    """Each rule's reduce() returns the scalar its threshold compares against."""
    from fitness_form_ai.domain.metrics import JointPairAxisDistanceMetric
    from fitness_form_ai.domain.rules import MaxDepthRule, MinValueRule

    rep = [
        RepFrame(angle=160, timestamp=0.0, velocity=0),
        RepFrame(angle=80, timestamp=0.5, velocity=120),
        RepFrame(angle=155, timestamp=1.0, velocity=40),
    ]
    metric = TrackerAngleMetric("Angle")

    assert MaxDepthRule("Depth", metric, 85).reduce(rep) == 80      # deepest
    assert MinValueRule("Lockout", metric, 150).reduce(rep) == 160  # peak
    assert StabilityRule("Drift", metric, 40).reduce(rep) == 80     # max-min
    assert TempoRule("Tempo", max_speed=250).reduce(rep) == 120     # peak speed
    # RangeRule with no breach reports the extreme closest to a bound.
    assert RangeRule("Band", metric, (0, 200)).reduce(rep) in (80, 160)
    # Empty rep -> no value.
    assert MinValueRule("Lockout", metric, 150).reduce([]) is None
