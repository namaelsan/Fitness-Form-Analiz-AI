from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.metrics import AngleMetric, SymmetryMetric, TrackerAngleMetric
from fitness_form_ai.domain.rules import AngleRule, MaxValueRule, RangeRule, SpeedRule, StabilityRule, TempoRule
from fitness_form_ai.inference.base import LandmarkPoint


def test_speed_rule_rejects_fast_rep() -> None:
    rule = SpeedRule("Tempo", max_speed=20)
    rep = [
        RepFrame(angle=90, timestamp=0.0, velocity=10),
        RepFrame(angle=100, timestamp=0.1, velocity=25),
    ]
    assert rule.apply(rep) is False


def test_angle_rule_uses_frame_angles_without_landmarks() -> None:
    rule = AngleRule("Range", TrackerAngleMetric("Tracker"), (30, 120))
    rep = [
        RepFrame(angle=45, timestamp=0.0),
        RepFrame(angle=90, timestamp=0.1),
    ]
    assert rule.apply(rep) is True


def test_angle_rule_uses_landmarks_when_available() -> None:
    rule = AngleRule("Elbow", AngleMetric("Elbow", ["A", "B", "C"]), (80, 100))
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
