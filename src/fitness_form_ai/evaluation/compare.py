"""Cross-model comparison: continuous metrics (RMSE, MAE, correlation)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fitness_form_ai.evaluation.benchmark import BenchmarkResult


@dataclass(slots=True)
class ContinuousComparison:
    """Comparison of continuous metrics between a model and the reference."""

    model_name: str
    reference_model: str
    video_path: str
    exercise_name: str
    
    angle_rmse: float = 0.0
    angle_mae: float = 0.0
    angle_corr: float = 0.0
    
    ref_rep_count: int = 0
    model_rep_count: int = 0
    overlap_frames: int = 0   # frames where BOTH models produced an angle

    @property
    def r_squared(self) -> float:
        """R-squared (coefficient of determination) approximation from pearson r."""
        return self.angle_corr ** 2


@dataclass(slots=True)
class ComparisonReport:
    """Full comparison output across all models."""

    reference_model: str
    comparisons: list[ContinuousComparison] = field(default_factory=list)


def compare_results(
    all_results: list[BenchmarkResult],
    reference_model: str = "mediapipe-heavy",
) -> ComparisonReport:
    """Compare all non-reference models against the reference model's continuous angles."""

    report = ComparisonReport(reference_model=reference_model)

    # Group results by (video, exercise) → {model_name: BenchmarkResult}
    groups: dict[tuple[str, str], dict[str, BenchmarkResult]] = {}
    for r in all_results:
        key = (r.video_path, r.exercise_name)
        groups.setdefault(key, {})[r.model_name] = r

    for (video_path, exercise_name), model_map in groups.items():
        ref = model_map.get(reference_model)
        if ref is None:
            continue

        for model_name, candidate in model_map.items():
            if model_name == reference_model:
                continue

            # Align by frame_index, NOT by position: different models drop
            # different frames, so a positional zip would silently compare
            # temporally different frames. We index each model's angles by the
            # source frame number and intersect on the frames both detected.
            ref_by_idx = {
                f.frame_index: f.primary_angle
                for f in ref.frames
                if f.primary_angle is not None
            }
            cand_by_idx = {
                f.frame_index: f.primary_angle
                for f in candidate.frames
                if f.primary_angle is not None
            }
            shared = sorted(ref_by_idx.keys() & cand_by_idx.keys())
            ref_angles = [ref_by_idx[i] for i in shared]
            cand_angles = [cand_by_idx[i] for i in shared]

            if ref_angles:
                diffs = [c - r for c, r in zip(cand_angles, ref_angles)]
                mae = sum(abs(d) for d in diffs) / len(diffs)
                rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
                
                # Pearson correlation
                mean_ref = sum(ref_angles) / len(ref_angles)
                mean_cand = sum(cand_angles) / len(cand_angles)
                
                num = sum((r - mean_ref) * (c - mean_cand) for r, c in zip(ref_angles, cand_angles))
                den1 = sum((r - mean_ref) ** 2 for r in ref_angles)
                den2 = sum((c - mean_cand) ** 2 for c in cand_angles)
                
                if den1 > 0 and den2 > 0:
                    corr = num / math.sqrt(den1 * den2)
                else:
                    corr = 0.0
            else:
                mae = 0.0
                rmse = 0.0
                corr = 0.0

            comp = ContinuousComparison(
                model_name=model_name,
                reference_model=reference_model,
                video_path=video_path,
                exercise_name=exercise_name,
                angle_rmse=rmse,
                angle_mae=mae,
                angle_corr=corr,
                ref_rep_count=len(ref.reps),
                model_rep_count=len(candidate.reps),
                overlap_frames=len(ref_angles),
            )
            report.comparisons.append(comp)

    return report
