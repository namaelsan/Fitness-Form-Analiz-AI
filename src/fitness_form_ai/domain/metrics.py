from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.utils.geometry import calculate_angle_3d
from fitness_form_ai.utils.landmarks import read_landmark


class Metric(ABC):
    def __init__(self, label: str) -> None:
        self.label = label

    def series(self, rep_data: list[RepFrame]) -> list[float]:
        values: list[float] = []
        for frame in rep_data:
            value = self.from_frame(frame)
            if value is not None:
                values.append(value)
        return values

    @abstractmethod
    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        raise NotImplementedError

    def from_frame(self, frame: RepFrame) -> float | None:
        if frame.landmarks:
            return self.from_landmarks(frame.landmarks)
        return None


class AngleMetric(Metric):
    def __init__(self, label: str, joints: list[str]) -> None:
        super().__init__(label)
        self.joints = joints

    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        joints = [read_landmark(joint, landmarks) for joint in self.joints]
        if any(joint is None for joint in joints):
            return None
        return calculate_angle_3d(joints)


class TrackerAngleMetric(Metric):
    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        return None

    def from_frame(self, frame: RepFrame) -> float | None:
        return frame.angle


class AxisMetric(Metric):
    def __init__(self, label: str, joint: str, axis: str) -> None:
        super().__init__(label)
        self.joint = joint
        self.axis = axis

    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        landmark = read_landmark(self.joint, landmarks)
        if landmark is None:
            return None
        return float(getattr(landmark, self.axis))


class JointPairAxisDistanceMetric(Metric):
    def __init__(self, label: str, joint_a: str, joint_b: str, axis: str) -> None:
        super().__init__(label)
        self.joint_a = joint_a
        self.joint_b = joint_b
        self.axis = axis

    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        first = read_landmark(self.joint_a, landmarks)
        second = read_landmark(self.joint_b, landmarks)
        if first is None or second is None:
            return None
        return abs(float(getattr(first, self.axis)) - float(getattr(second, self.axis)))


class SegmentOrientationMetric(Metric):
    def __init__(self, label: str, start_joint: str, end_joint: str, reference: str) -> None:
        super().__init__(label)
        self.start_joint = start_joint
        self.end_joint = end_joint
        self.reference = reference

    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        start = read_landmark(self.start_joint, landmarks)
        end = read_landmark(self.end_joint, landmarks)
        if start is None or end is None:
            return None

        if self.reference == "vertical":
            anchor = _Point(x=start.x, y=start.y - 1.0, z=start.z)
        else:
            anchor = _Point(x=start.x + 1.0, y=start.y, z=start.z)
        return calculate_angle_3d([anchor, start, end])


class SymmetryMetric(Metric):
    def __init__(self, label: str, left_metric: Metric, right_metric: Metric) -> None:
        super().__init__(label)
        self.left_metric = left_metric
        self.right_metric = right_metric

    def from_landmarks(self, landmarks: dict[str, object]) -> float | None:
        left = self.left_metric.from_landmarks(landmarks)
        right = self.right_metric.from_landmarks(landmarks)
        if left is None or right is None:
            return None
        return abs(left - right)


@dataclass(slots=True)
class _Point:
    x: float
    y: float
    z: float
