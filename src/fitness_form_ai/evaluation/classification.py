"""Classification evaluation: precision, recall, F1 per exercise and per rule.

Requires a labels CSV with columns:
    video_path, exercise, expected_class, expected_violated_rules

A video is predicted "casual" if at least one of its reps has at least one
failed rule.  A video with zero detected reps is labelled "no_reps" and
excluded from accuracy metrics (but reported separately so you can see it).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from fitness_form_ai.evaluation.benchmark import BenchmarkResult


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VideoLabel:
    video_path: str          # normalised relative path
    exercise: str
    expected_class: str      # "proper" or "casual"
    expected_violated_rules: list[str]  # may be empty even for casual


def load_labels(labels_csv: Path) -> list[VideoLabel]:
    """Parse the labels CSV.  Comment lines (starting with #) are skipped."""
    labels: list[VideoLabel] = []
    with open(labels_csv, newline="") as fh:
        for row in csv.DictReader(filter(lambda l: not l.startswith("#"), fh)):
            raw_rules = row.get("expected_violated_rules", "").strip()
            rules = [r.strip() for r in raw_rules.split(",") if r.strip()]
            labels.append(VideoLabel(
                video_path=row["video_path"].strip(),
                exercise=row["exercise"].strip(),
                expected_class=row["expected_class"].strip(),
                expected_violated_rules=rules,
            ))
    return labels


# ---------------------------------------------------------------------------
# Per-video prediction
# ---------------------------------------------------------------------------

def _predict_class(result: BenchmarkResult) -> str:
    """Predict "proper", "casual", or "no_reps" from a benchmark result."""
    if not result.reps:
        return "no_reps"
    # "casual" if ANY rep failed ANY rule, else "proper".
    if any(rep.failed_rules for rep in result.reps):
        return "casual"
    return "proper"


def _failed_rules_union(result: BenchmarkResult) -> set[str]:
    """All rule names that failed in at least one rep."""
    failed: set[str] = set()
    for rep in result.reps:
        failed.update(rep.failed_rules)
    return failed


# ---------------------------------------------------------------------------
# Per-video classification result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VideoClassificationResult:
    video_path: str
    exercise: str
    expected_class: str
    predicted_class: str          # "proper", "casual", or "no_reps"
    expected_violated_rules: list[str]
    actually_failed_rules: list[str]  # union across all reps

    @property
    def is_correct(self) -> bool:
        return self.predicted_class == self.expected_class

    @property
    def outcome(self) -> str:
        """TP / TN / FP / FN / no_reps label for the video-level binary task."""
        if self.predicted_class == "no_reps":
            return "no_reps"
        if self.expected_class == "casual" and self.predicted_class == "casual":
            return "TP"
        if self.expected_class == "proper" and self.predicted_class == "proper":
            return "TN"
        if self.expected_class == "proper" and self.predicted_class == "casual":
            return "FP"
        # expected casual, predicted proper
        return "FN"


# ---------------------------------------------------------------------------
# Aggregate metrics per exercise
# ---------------------------------------------------------------------------

@dataclass
class ExerciseMetrics:
    exercise: str
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    no_reps: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Per-rule metrics
# ---------------------------------------------------------------------------

@dataclass
class RuleMetrics:
    rule_name: str
    exercise: str
    # A rule TP = expected to fire AND fired.
    # A rule FN = expected to fire but did NOT fire.
    # We cannot compute FP / TN at rule level without per-video rule labels
    # for proper videos, so we report sensitivity only.
    tp: int = 0
    fn: int = 0

    @property
    def sensitivity(self) -> float:
        """Fraction of labelled casual videos (with this rule listed) that the
        system correctly flagged via this rule."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    video_results: list[VideoClassificationResult] = field(default_factory=list)
    exercise_metrics: dict[str, ExerciseMetrics] = field(default_factory=dict)
    rule_metrics: dict[str, RuleMetrics] = field(default_factory=dict)  # key = "exercise::rule"


def evaluate_classification(
    benchmark_results: list[BenchmarkResult],
    labels: list[VideoLabel],
) -> ClassificationReport:
    """Match benchmark results to labels and compute all classification metrics."""

    # Build lookup: normalised video path → BenchmarkResult
    result_map: dict[str, BenchmarkResult] = {}
    for r in benchmark_results:
        # Normalise to forward-slash relative path
        key = Path(r.video_path).as_posix()
        result_map[key] = r

    report = ClassificationReport()

    for label in labels:
        key = Path(label.video_path).as_posix()
        result = result_map.get(key)
        if result is None:
            # Video was labelled but not benchmarked — skip
            continue

        predicted = _predict_class(result)
        failed = sorted(_failed_rules_union(result))

        vcr = VideoClassificationResult(
            video_path=label.video_path,
            exercise=label.exercise,
            expected_class=label.expected_class,
            predicted_class=predicted,
            expected_violated_rules=label.expected_violated_rules,
            actually_failed_rules=failed,
        )
        report.video_results.append(vcr)

        # ---- exercise-level confusion matrix ----
        ex = label.exercise
        if ex not in report.exercise_metrics:
            report.exercise_metrics[ex] = ExerciseMetrics(exercise=ex)
        em = report.exercise_metrics[ex]
        outcome = vcr.outcome
        if outcome == "TP":
            em.tp += 1
        elif outcome == "TN":
            em.tn += 1
        elif outcome == "FP":
            em.fp += 1
        elif outcome == "FN":
            em.fn += 1
        else:
            em.no_reps += 1

        # ---- per-rule sensitivity (only for casual videos with listed rules) ----
        if label.expected_class == "casual" and label.expected_violated_rules:
            failed_set = set(failed)
            for rule_name in label.expected_violated_rules:
                rkey = f"{ex}::{rule_name}"
                if rkey not in report.rule_metrics:
                    report.rule_metrics[rkey] = RuleMetrics(
                        rule_name=rule_name, exercise=ex
                    )
                rm = report.rule_metrics[rkey]
                if rule_name in failed_set:
                    rm.tp += 1
                else:
                    rm.fn += 1

    return report
