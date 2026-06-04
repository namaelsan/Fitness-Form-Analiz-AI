"""Classification evaluation: precision, recall, F1 per exercise and per rule.

Requires a labels CSV with columns:
    video_path, exercise, expected_class, expected_violated_rules

Classification is evaluated **per repetition**: every rep the tracker extracts
is its own proper/casual sample, inheriting its clip's expected class.  A rep is
predicted "casual" if it failed at least one rule, otherwise "proper".  This is
finer-grained and more faithful than collapsing a whole clip into a single
label — a single tripped rep no longer condemns an otherwise-clean clip, and a
casual clip contributes one sample per rep rather than one per video.

A clip that yields zero detected reps cannot produce any rep sample; it is
counted separately as ``no_reps`` (a per-clip diagnostic) and excluded from the
rep-level accuracy metrics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from fitness_form_ai.evaluation.benchmark import BenchmarkResult, RepRecord


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
# Per-rep prediction
# ---------------------------------------------------------------------------

def _predict_rep_class(rep: RepRecord) -> str:
    """Predict "proper" or "casual" for a single repetition.

    A rep is "casual" if it failed at least one rule, else "proper".
    """
    return "casual" if rep.failed_rules else "proper"


# ---------------------------------------------------------------------------
# Per-rep classification result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RepClassificationResult:
    video_path: str
    exercise: str
    rep_index: int
    expected_class: str           # inherited from the clip label
    predicted_class: str          # "proper" or "casual"
    failed_rules: list[str]       # rules that failed on this rep

    @property
    def is_correct(self) -> bool:
        return self.predicted_class == self.expected_class

    @property
    def outcome(self) -> str:
        """TP / TN / FP / FN for the rep-level binary task (casual = positive)."""
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
    # tp/tn/fp/fn are rep-level counts.
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    # no_reps is a clip-level diagnostic: clips that produced zero reps.
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
    # A rule TP = expected to fire on a casual rep AND fired.
    # A rule FN = expected to fire on a casual rep but did NOT fire.
    # We cannot compute FP / TN at rule level without per-rep rule labels
    # for proper reps, so we report sensitivity only.
    tp: int = 0
    fn: int = 0

    @property
    def sensitivity(self) -> float:
        """Fraction of casual reps (whose clip lists this rule) that the system
        correctly flagged via this rule."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    rep_results: list[RepClassificationResult] = field(default_factory=list)
    exercise_metrics: dict[str, ExerciseMetrics] = field(default_factory=dict)
    rule_metrics: dict[str, RuleMetrics] = field(default_factory=dict)  # key = "exercise::rule"
    # Clips that produced zero reps (path -> exercise), kept for diagnostics.
    no_reps_clips: list[str] = field(default_factory=list)


def evaluate_classification(
    benchmark_results: list[BenchmarkResult],
    labels: list[VideoLabel],
) -> ClassificationReport:
    """Match benchmark results to labels and compute per-rep classification metrics."""

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

        ex = label.exercise
        if ex not in report.exercise_metrics:
            report.exercise_metrics[ex] = ExerciseMetrics(exercise=ex)
        em = report.exercise_metrics[ex]

        if not result.reps:
            # No complete rep extracted — clip-level diagnostic only.
            em.no_reps += 1
            report.no_reps_clips.append(label.video_path)
            continue

        for rep in result.reps:
            predicted = _predict_rep_class(rep)
            rcr = RepClassificationResult(
                video_path=label.video_path,
                exercise=ex,
                rep_index=rep.rep_index,
                expected_class=label.expected_class,
                predicted_class=predicted,
                failed_rules=list(rep.failed_rules),
            )
            report.rep_results.append(rcr)

            # ---- exercise-level (rep) confusion matrix ----
            outcome = rcr.outcome
            if outcome == "TP":
                em.tp += 1
            elif outcome == "TN":
                em.tn += 1
            elif outcome == "FP":
                em.fp += 1
            else:  # FN
                em.fn += 1

            # ---- per-rule sensitivity (only for casual reps whose clip lists rules) ----
            if label.expected_class == "casual" and label.expected_violated_rules:
                failed_set = set(rep.failed_rules)
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
