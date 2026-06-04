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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        """Return the single scalar this rule's threshold is compared against.

        This is the value that determines pass/fail for the rep (e.g. the
        deepest angle for a depth rule, the peak velocity for a tempo rule).
        Returning ``None`` means the rule had no usable data for the rep.

        Subclasses override this so the benchmark never has to ``isinstance``
        its way through the rule hierarchy to recover a reported value.
        """
        return None


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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        if not values:
            return None
        lo, hi = self.angle_range
        mn, mx = min(values), max(values)
        # If the band is breached, report the more-violated extreme; otherwise
        # report the extreme with the smaller margin so the reader sees how
        # close the rep came to failing.
        if mx > hi or mn < lo:
            return mx if (mx - hi) >= (lo - mn) else mn
        return mx if (hi - mx) <= (mn - lo) else mn

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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        return max(values) if values else None

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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        return max(values) if values else None

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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        return min(values) if values else None

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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        return (max(values) - min(values)) if values else None

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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        ratios = self._ratio_series(rep_data)
        return min(ratios) if ratios else None

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


class FloorRule(MetricRule):
    """Passes when the minimum observed value stays at or above *minimum*.

    Use this when a metric must never drop below a threshold — e.g. the
    shoulder-ear distance must not collapse (shoulder shrug).
    """

    def __init__(self, rule_name: str, metric: Metric, minimum: float) -> None:
        super().__init__(rule_name, metric)
        self.minimum = minimum

    def apply(self, rep_data: list[RepFrame]) -> bool:
        values = self.metric.series(rep_data)
        if not values:
            return False
        return min(values) >= self.minimum

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        return min(values) if values else None

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"target >= {self._format_value(self.minimum)}"
        return f"{self.metric.label}: {self._format_value(value)} (floor >= {self._format_value(self.minimum)})"


class RelativeDropRule(MetricRule):
    """Passes when the metric never contracts too far from its per-rep baseline.

    Unlike :class:`FloorRule`, this carries no absolute threshold. The baseline
    is the largest value the metric reaches during the rep (its most open /
    most-depressed position), and the rule fails when the metric later
    collapses by more than ``max_drop`` *fraction* of that baseline. Because the
    comparison is a ratio against the rep's own scale, it is invariant to body
    size, camera distance, and landmark normalization — so it detects a genuine
    "major change" in the gap between two landmarks rather than tripping on an
    arbitrary minimum value.

    Example: for a lateral raise, the shoulder-ear vertical gap is widest when
    the shoulder is depressed; a shrug shrinks that gap. ``max_drop=0.20`` fails
    the rep when the gap contracts by more than 20% off its baseline.
    """

    def __init__(self, rule_name: str, metric: Metric, max_drop: float) -> None:
        super().__init__(rule_name, metric)
        self.max_drop = max_drop

    def _drop_ratio(self, rep_data: list[RepFrame]) -> float | None:
        values = self.metric.series(rep_data)
        if not values:
            return None
        baseline = max(values)
        if baseline <= 0:
            return None
        return (baseline - min(values)) / baseline

    def apply(self, rep_data: list[RepFrame]) -> bool:
        ratio = self._drop_ratio(rep_data)
        if ratio is None:
            return False
        return ratio <= self.max_drop

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        return self._drop_ratio(rep_data)

    def describe_current(self, context: RuleContext) -> str:
        value = self._current_metric_value(context)
        if value is None:
            return f"max drop {self.max_drop * 100:.0f}% off baseline"
        return (
            f"{self.metric.label}: {self._format_value(value)} "
            f"(max drop {self.max_drop * 100:.0f}% off baseline)"
        )


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

    def reduce(self, rep_data: list[RepFrame]) -> float | None:
        """Peak absolute angular velocity over the rep (deg/s)."""
        if not rep_data:
            return None
        return max(abs(frame.velocity) for frame in rep_data)

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

