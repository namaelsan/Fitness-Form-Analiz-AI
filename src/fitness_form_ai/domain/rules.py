from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fitness_form_ai.domain.metrics import Metric
from fitness_form_ai.domain.rep_frame import RepFrame


@dataclass(slots=True)
class RuleContext:
    landmarks: dict[str, object] | None = None
    current_frame: RepFrame | None = None
    rep_duration: float = 0.0


class Rule(ABC):
    def __init__(self, rule_name: str) -> None:
        self.rule_name = rule_name

    @abstractmethod
    def apply(self, rep_data: list[RepFrame]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_current_value(self, landmarks: dict[str, object]) -> float:
        raise NotImplementedError

    @abstractmethod
    def describe_current(self, context: RuleContext) -> str:
        raise NotImplementedError


class MetricRule(Rule):
    def __init__(self, rule_name: str, metric: Metric) -> None:
        super().__init__(rule_name)
        self.metric = metric

    def get_current_value(self, landmarks: dict[str, object]) -> float:
        value = self.metric.from_landmarks(landmarks)
        return 0.0 if value is None else value

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return "waiting for landmarks"
        return f"{self.metric.label}: {self._format_value(value)}"

    def _current_metric_value(self, context: RuleContext) -> float | None:
        if context.current_frame is not None:
            value = self.metric.from_frame(context.current_frame)
            if value is not None:
                return value
        if context.landmarks:
            return self.metric.from_landmarks(context.landmarks)
        return None

    @staticmethod
    def _format_value(value: float) -> str:
        if abs(value) >= 10:
            return f"{value:.0f}"
        if abs(value) >= 1:
            return f"{value:.1f}"
        return f"{value:.3f}"


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

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"target {self._format_value(self.angle_range[0])}-{self._format_value(self.angle_range[1])}"
        return (
            f"{self.metric.label}: {self._format_value(value)} "
            f"(target {self._format_value(self.angle_range[0])}-{self._format_value(self.angle_range[1])})"
        )


class MinValueRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, minimum: float) -> None:
        super().__init__(rule_name, metric)
        self.minimum = minimum

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return max(values) >= self.minimum

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"target >= {self._format_value(self.minimum)}"
        return f"{self.metric.label}: {self._format_value(value)} (target >= {self._format_value(self.minimum)})"


class MaxValueRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, maximum: float) -> None:
        super().__init__(rule_name, metric)
        self.maximum = maximum

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return max(values) <= self.maximum

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"target <= {self._format_value(self.maximum)}"
        return f"{self.metric.label}: {self._format_value(value)} (target <= {self._format_value(self.maximum)})"


class MaxDepthRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, threshold: float) -> None:
        super().__init__(rule_name, metric)
        self.threshold = threshold

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return min(values) <= self.threshold

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"target depth <= {self._format_value(self.threshold)}"
        return f"{self.metric.label}: {self._format_value(value)} (target depth <= {self._format_value(self.threshold)})"


class StabilityRule(MetricRule):
    def __init__(self, rule_name: str, metric: Metric, max_delta: float) -> None:
        super().__init__(rule_name, metric)
        self.max_delta = max_delta

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return (max(values) - min(values)) <= self.max_delta

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"drift limit {self._format_value(self.max_delta)}"
        return f"{self.metric.label}: {self._format_value(value)} live, drift limit {self._format_value(self.max_delta)}"


class KneeValgusRule(Rule):
    def __init__(
        self,
        rule_name: str,
        knee_width_metric: Metric,
        stance_width_metric: Metric,
        min_ratio: float,
    ) -> None:
        super().__init__(rule_name)
        self.knee_width_metric = knee_width_metric
        self.stance_width_metric = stance_width_metric
        self.min_ratio = min_ratio

    def apply(self, rep_data: list[RepFrame]) -> bool:
        ratios = self._ratio_series(rep_data)
        if not ratios:
            return False
        return min(ratios) >= self.min_ratio

    def get_current_value(self, landmarks: dict[str, object]) -> float:
        knee_width = self.knee_width_metric.from_landmarks(landmarks)
        stance_width = self.stance_width_metric.from_landmarks(landmarks)
        if knee_width is None or stance_width is None or stance_width <= 0:
            return 0.0
        return knee_width / stance_width

    def describe_current(self, context: RuleContext) -> str:
        if not context.landmarks:
            return f"knees/feet ratio >= {self.min_ratio:.2f}"

        knee_width = self.knee_width_metric.from_landmarks(context.landmarks)
        stance_width = self.stance_width_metric.from_landmarks(context.landmarks)
        if knee_width is None or stance_width is None or stance_width <= 0:
            return f"knees/feet ratio >= {self.min_ratio:.2f}"

        ratio = knee_width / stance_width
        return (
            f"knees {knee_width:.3f}, feet {stance_width:.3f}, "
            f"ratio {ratio:.2f} (min {self.min_ratio:.2f})"
        )

    def _ratio_series(self, rep_data: list[RepFrame]) -> list[float]:
        ratios: list[float] = []
        for frame in rep_data:
            if not frame.landmarks:
                continue
            knee_width = self.knee_width_metric.from_landmarks(frame.landmarks)
            stance_width = self.stance_width_metric.from_landmarks(frame.landmarks)
            if knee_width is None or stance_width is None or stance_width <= 0:
                continue
            ratios.append(knee_width / stance_width)
        return ratios


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

    def describe_current(self, context: RuleContext) -> str:
        parts: list[str] = []
        if context.current_frame is not None and self.max_speed is not None:
            parts.append(f"speed {abs(context.current_frame.velocity):.0f}/{self.max_speed:.0f} deg/s")
        if self.min_duration is not None:
            parts.append(f"rep {context.rep_duration:.1f}/{self.min_duration:.1f}s")
        if not parts:
            return "evaluated at rep end"
        return ", ".join(parts)

