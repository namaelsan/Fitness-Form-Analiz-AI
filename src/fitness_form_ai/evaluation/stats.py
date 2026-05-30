"""Uncertainty quantification and baselines for the evaluation metrics.

Point estimates alone (a single F1, a single PA-MPJPE) cannot tell a reader
whether a difference between two models is real or noise.  This module adds:

*   Bootstrap confidence intervals, including a *cluster bootstrap* that
    resamples whole subjects rather than individual clips — appropriate
    because clips from the same FIT3D subject are not independent.
*   Subject-held-out aggregation, so a model is scored on subjects whose data
    did not shape its thresholds.
*   A majority-class baseline, the floor any "real" classifier must beat.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Deterministic by default so reported intervals are reproducible.
_RNG = random.Random(20240530)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[Sequence[float]], float] = _mean,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for *statistic* over *values*.

    Returns ``(low, high)``.  With fewer than two observations the interval
    collapses to the point estimate (there is nothing to resample).
    """
    if not values:
        return (0.0, 0.0)
    if len(values) < 2:
        v = statistic(values)
        return (v, v)

    rng = rng or _RNG
    n = len(values)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def bootstrap_ci_clustered(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[Sequence[float]], float] = _mean,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Cluster bootstrap: resample whole clusters (e.g. subjects) with
    replacement, then take all observations belonging to the drawn clusters.

    This widens the interval appropriately when observations within a cluster
    are correlated.
    """
    if not values:
        return (0.0, 0.0)
    by_cluster: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        by_cluster.setdefault(cluster, []).append(value)

    keys = list(by_cluster.keys())
    if len(keys) < 2:
        # Only one cluster — fall back to the ordinary bootstrap over clips.
        return bootstrap_ci(
            values, n_boot=n_boot, alpha=alpha, statistic=statistic, rng=rng
        )

    rng = rng or _RNG
    k = len(keys)
    stats: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _ in range(k):
            drawn.extend(by_cluster[keys[rng.randrange(k)]])
        stats.append(statistic(drawn))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Classification-specific helpers
# ---------------------------------------------------------------------------

def _f1_from_outcomes(outcomes: Sequence[str]) -> float:
    tp = outcomes.count("TP")
    fp = outcomes.count("FP")
    fn = outcomes.count("FN")
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _accuracy_from_outcomes(outcomes: Sequence[str]) -> float:
    scored = [o for o in outcomes if o in {"TP", "TN", "FP", "FN"}]
    correct = sum(1 for o in scored if o in {"TP", "TN"})
    return correct / len(scored) if scored else 0.0


@dataclass(slots=True)
class MetricWithCI:
    point: float
    ci_low: float
    ci_high: float


@dataclass(slots=True)
class ClassificationCIs:
    accuracy: MetricWithCI
    f1: MetricWithCI
    n_videos: int
    n_subjects: int


def classification_cis(
    video_results: "Sequence[object]",
    *,
    n_boot: int = 2000,
) -> ClassificationCIs:
    """Bootstrap accuracy and F1 CIs over per-video classification results.

    *video_results* are ``VideoClassificationResult`` objects (duck-typed:
    each must expose ``outcome`` and ``video_path``).  The bootstrap is
    clustered by FIT3D subject parsed from the video path.
    """
    from fitness_form_ai.evaluation.mocap import subject_of

    outcomes = [vr.outcome for vr in video_results]
    subjects = [subject_of(vr.video_path) for vr in video_results]

    # Encode each video as its outcome string and bootstrap over the encoded
    # list; the statistic decodes outcomes back into a metric.
    idx = list(range(len(outcomes)))

    def _acc(sample_idx: Sequence[float]) -> float:
        return _accuracy_from_outcomes([outcomes[int(i)] for i in sample_idx])

    def _f1(sample_idx: Sequence[float]) -> float:
        return _f1_from_outcomes([outcomes[int(i)] for i in sample_idx])

    acc_lo, acc_hi = bootstrap_ci_clustered(
        idx, subjects, n_boot=n_boot, statistic=_acc
    )
    f1_lo, f1_hi = bootstrap_ci_clustered(
        idx, subjects, n_boot=n_boot, statistic=_f1
    )

    return ClassificationCIs(
        accuracy=MetricWithCI(_accuracy_from_outcomes(outcomes), acc_lo, acc_hi),
        f1=MetricWithCI(_f1_from_outcomes(outcomes), f1_lo, f1_hi),
        n_videos=len(outcomes),
        n_subjects=len(set(subjects)),
    )


@dataclass(slots=True)
class BaselineResult:
    strategy: str            # e.g. "majority:proper"
    accuracy: float
    f1: float


def majority_class_baseline(labels: "Sequence[object]") -> BaselineResult:
    """Accuracy/F1 of always predicting the most common label.

    *labels* are ``VideoLabel`` objects exposing ``expected_class``.  This is
    the floor a useful classifier must clear; reporting it stops a high
    accuracy on an imbalanced set from looking impressive when a constant
    predictor would match it.
    """
    classes = [lbl.expected_class for lbl in labels]
    if not classes:
        return BaselineResult("majority:none", 0.0, 0.0)

    majority = max(set(classes), key=classes.count)
    # "casual" is the positive class (mirrors classification.outcome).
    outcomes: list[str] = []
    for cls in classes:
        if majority == "casual":
            outcomes.append("TP" if cls == "casual" else "FP")
        else:  # always predict proper
            outcomes.append("TN" if cls == "proper" else "FN")

    return BaselineResult(
        strategy=f"majority:{majority}",
        accuracy=_accuracy_from_outcomes(outcomes),
        f1=_f1_from_outcomes(outcomes),
    )
