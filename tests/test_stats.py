import random

from fitness_form_ai.evaluation.stats import (
    bootstrap_ci,
    bootstrap_ci_clustered,
    majority_class_baseline,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    lo, hi = bootstrap_ci(values, n_boot=500, rng=random.Random(0))
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_degenerate_single_value() -> None:
    assert bootstrap_ci([3.0]) == (3.0, 3.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_cluster_bootstrap_is_wider_than_naive() -> None:
    # Two subjects with very different levels: clustering should reflect the
    # between-subject spread and produce a wider interval than the naive CI.
    values = [1.0] * 10 + [9.0] * 10
    clusters = ["s1"] * 10 + ["s2"] * 10
    rng1, rng2 = random.Random(1), random.Random(1)
    naive_lo, naive_hi = bootstrap_ci(values, n_boot=1000, rng=rng1)
    clus_lo, clus_hi = bootstrap_ci_clustered(values, clusters, n_boot=1000, rng=rng2)
    assert (clus_hi - clus_lo) > (naive_hi - naive_lo)


class _Label:
    def __init__(self, expected_class: str) -> None:
        self.expected_class = expected_class


def test_majority_baseline_on_imbalanced_set() -> None:
    # 8 proper, 2 casual -> always-proper baseline scores 0.8 accuracy but 0 F1.
    labels = [_Label("proper")] * 8 + [_Label("casual")] * 2
    base = majority_class_baseline(labels)
    assert base.strategy == "majority:proper"
    assert abs(base.accuracy - 0.8) < 1e-9
    assert base.f1 == 0.0
