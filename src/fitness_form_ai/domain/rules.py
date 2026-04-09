from __future__ import annotations

from abc import ABC, abstractmethod

from fitness_form_ai.domain.metrics import Metric
from fitness_form_ai.domain.rep_frame import RepFrame


class Rule(ABC):
    def __init__(self, rule_name: str) -> None:
        self.rule_name = rule_name

    @abstractmethod
    def apply(self, rep_data: list[RepFrame]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_current_value(self, landmarks: dict[str, object]) -> float:
        raise NotImplementedError


class MetricRule(Rule):
    def __init__(self, rule_name: str, metric: Metric) -> None:
        super().__init__(rule_name)
        self.metric = metric

    def get_current_value(self, landmarks: dict[str, object]) -> float:
        value = self.metric.from_landmarks(landmarks)
        return 0.0 if value is None else value


class SpeedRule(Rule):
    def __init__(self, rule_name: str, max_speed: float) -> None:
        super().__init__(rule_name)
        self.max_speed = max_speed

    def apply(self, rep_data: list[RepFrame]) -> bool:
        if not rep_data:
            return True
        max_rep_speed = max(abs(frame.velocity) for frame in rep_data)
        return max_rep_speed <= self.max_speed

    def get_current_value(self, landmarks: dict[str, object]) -> float:
        return 0.0


class RangeRule(MetricRule):
    def __init__(
        self,
        rule_name: str,
        metric: Metric,
        angle_range: tuple[float, float],
    ) -> None:
        super().__init__(rule_name, metric)
        self.angle_range = angle_range

    def apply(self, rep_data: list[RepFrame]) -> bool:
        if not rep_data:
            return False
        values = self.metric.series(rep_data)
        if not values:
            return False
        return min(values) >= self.angle_range[0] and max(values) <= self.angle_range[1]


class MinValueRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, minimum: float) -> None:
        super().__init__(rule_name, metric)
        self.minimum = minimum

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return max(values) >= self.minimum


class MaxValueRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, maximum: float) -> None:
        super().__init__(rule_name, metric)
        self.maximum = maximum

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return max(values) <= self.maximum


class StabilityRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, max_delta: float) -> None:
        super().__init__(rule_name, metric)
        self.max_delta = max_delta

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return (max(values) - min(values)) <= self.max_delta


class TempoRule(Rule):
    def __init__(
        self,
        rule_name: str,
        max_speed: float | None = None,
        min_duration: float | None = None,
    ) -> None:
        super().__init__(rule_name)
        self.max_speed = max_speed
        self.min_duration = min_duration

    def apply(self, rep_data: list[RepFrame]) -> bool:
        if not rep_data:
            return False
        if self.max_speed is not None:
            max_speed = max(abs(frame.velocity) for frame in rep_data)
            if max_speed > self.max_speed:
                return False
        if self.min_duration is not None:
            duration = rep_data[-1].timestamp - rep_data[0].timestamp
            if duration < self.min_duration:
                return False
        return True

    def get_current_value(self, landmarks: dict[str, object]) -> float:
        return 0.0


class AngleRule(RangeRule):
    pass
