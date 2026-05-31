"""Ground-truth evaluation against FIT3D marker-based mocap.

The FIT3D dataset ships per-clip 3D joint positions (``joints3d_25``) captured
with a motion-capture system.  These are the gold standard: comparing a pose
model's output against *them* tells us how accurate the model actually is, as
opposed to ``compare.py`` which only measures agreement between two estimators.

Two facts make the comparison non-trivial and are handled explicitly here:

1.  **Different coordinate frames.**  FIT3D world coordinates are Z-up, in
    metres, with an arbitrary global position/orientation.  MediaPipe world
    landmarks are hip-centred with a different axis convention.  We therefore
    align each predicted frame to the ground truth with a similarity
    (Umeyama: rotation + uniform scale + translation) transform before
    measuring positional error.  The reported metric is Procrustes-Aligned
    MPJPE (PA-MPJPE), the standard in the 3D human-pose literature.

2.  **Joint correspondence.**  Only a subset of the 25 FIT3D joints have a
    clean MediaPipe analogue; the mapping below is the authoritative place to
    correct it.  The FIT3D 25-joint order was verified empirically from the
    s03 clips (limb-chain ordering: pelvis, then left leg, right leg, spine,
    head, left arm, right arm, then extremities).

Joint *angles* (used for the per-exercise angle error) are invariant to any
similarity transform, so those are computed directly on the raw coordinates
without alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import sys
import warnings

import cv2
import numpy as np

from fitness_form_ai.app.catalog import EXERCISE_REGISTRY
from fitness_form_ai.inference.base import PoseModel
from fitness_form_ai.inference.factory import create_pose_model
from fitness_form_ai.utils.geometry import letterbox_resize


# ---------------------------------------------------------------------------
# FIT3D joints3d_25 layout  (index -> joint), verified from the data.
# ---------------------------------------------------------------------------

FIT3D_JOINT_NAMES: dict[int, str] = {
    0: "pelvis",
    1: "left_hip",
    2: "left_knee",
    3: "left_ankle",
    4: "right_hip",
    5: "right_knee",
    6: "right_ankle",
    7: "spine",
    8: "thorax",
    9: "neck",
    10: "head",
    11: "left_shoulder",
    12: "left_elbow",
    13: "left_wrist",
    14: "right_shoulder",
    15: "right_elbow",
    16: "right_wrist",
    17: "left_toe",
    18: "left_heel",
    19: "right_toe",
    20: "right_heel",
    21: "left_hand",
    22: "left_fingertip",
    23: "right_hand",
    24: "right_fingertip",
}

# MediaPipe PoseLandmark name -> FIT3D joint index.  Only joints with a
# reliable one-to-one correspondence are included; this is the single place to
# adjust if the mapping is ever found to be off.
MP_TO_FIT3D: dict[str, int] = {
    "LEFT_HIP": 1,
    "LEFT_KNEE": 2,
    "LEFT_ANKLE": 3,
    "RIGHT_HIP": 4,
    "RIGHT_KNEE": 5,
    "RIGHT_ANKLE": 6,
    "LEFT_SHOULDER": 11,
    "LEFT_ELBOW": 12,
    "LEFT_WRIST": 13,
    "RIGHT_SHOULDER": 14,
    "RIGHT_ELBOW": 15,
    "RIGHT_WRIST": 16,
    # NOTE: foot joints (FIT3D 17-20) are intentionally excluded — the
    # toe/heel ordering could not be verified with certainty, and a wrong
    # correspondence would inflate PA-MPJPE. Add them here once confirmed.
}


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_joints3d(gt_path: Path) -> np.ndarray:
    """Load a FIT3D ``joints3d_25`` file as an ``(F, 25, 3)`` float array."""
    with open(gt_path) as fh:
        data = json.load(fh)
    arr = np.asarray(data["joints3d_25"], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (25, 3):
        raise ValueError(f"Unexpected joints3d_25 shape {arr.shape} in {gt_path}")
    return arr


def resolve_gt_path(video_path: Path) -> Path | None:
    """Map ``.../videos/<camera>/<clip>.mp4`` to ``.../joints3d_25/<clip>.json``.

    Returns ``None`` when no matching ground-truth file exists.
    """
    video_path = Path(video_path)
    # Layout: <subject>/videos/<camera>/<clip>.mp4
    #         <subject>/joints3d_25/<clip>.json
    # video_path.parent        = <camera dir>
    # video_path.parent.parent = videos/
    # video_path.parent.parent.parent = <subject dir>  ← joints3d_25 lives here
    gt = video_path.parent.parent.parent / "joints3d_25" / f"{video_path.stem}.json"
    return gt if gt.exists() else None


def subject_of(video_path: Path) -> str:
    """Extract the FIT3D subject id (e.g. ``s03``) from a clip path."""
    for part in Path(video_path).parts:
        if len(part) == 3 and part[0] == "s" and part[1:].isdigit():
            return part
    return "unknown"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def umeyama_align(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True) -> np.ndarray:
    """Return *src* rigidly + scale aligned onto *dst* (least-squares).

    *src*, *dst* are ``(N, 3)`` corresponding point sets.  Implements the
    Umeyama (1991) closed-form similarity solution.
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)

    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rot = u @ s @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = float(np.trace(np.diag(d) @ s) / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    t = mu_dst - scale * (rot @ mu_src)
    return (scale * (rot @ src.T)).T + t


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Interior angle at *b* of the segment a-b-c, in degrees (full 3D)."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return None
    cos = float(np.dot(ba, bc) / (nba * nbc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MocapEvalResult:
    """Ground-truth accuracy for one (model, clip) pair."""

    model_name: str
    video_path: str
    exercise_name: str
    subject: str

    per_frame_mpjpe_mm: list[float] = field(default_factory=list)
    per_frame_angle_abs_err: list[float] = field(default_factory=list)
    evaluated_frames: int = 0
    total_video_frames: int = 0

    @property
    def pa_mpjpe_mm(self) -> float:
        return float(np.mean(self.per_frame_mpjpe_mm)) if self.per_frame_mpjpe_mm else 0.0

    @property
    def pa_mpjpe_p95_mm(self) -> float:
        if not self.per_frame_mpjpe_mm:
            return 0.0
        return float(np.percentile(self.per_frame_mpjpe_mm, 95))

    @property
    def angle_mae_deg(self) -> float:
        return (
            float(np.mean(self.per_frame_angle_abs_err))
            if self.per_frame_angle_abs_err
            else 0.0
        )

    @property
    def coverage(self) -> float:
        """Fraction of video frames for which a GT-comparable pose was produced."""
        return self.evaluated_frames / self.total_video_frames if self.total_video_frames else 0.0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_clip_against_mocap(
    model_name: str,
    video_path: Path,
    exercise_name: str,
    *,
    gt_path: Path | None = None,
    resize: tuple[int, int] = (640, 480),
    gt_offset: int = 0,
    min_points: int = 6,
    model: PoseModel | None = None,
) -> MocapEvalResult:
    """Run *model_name* over *video_path* and score it against FIT3D mocap.

    *gt_offset* shifts the ground-truth frame index relative to the video frame
    index (FIT3D clips are normally already synchronised, so the default is 0).
    *min_points* is the minimum number of corresponded joints required before a
    frame contributes to PA-MPJPE.

    If *model* is supplied the caller is responsible for its lifecycle (no
    ``model.release()`` is called here).  Pass a pre-created instance to amortise
    model initialisation cost across many clips.
    """
    video_path = Path(video_path)
    if gt_path is None:
        gt_path = resolve_gt_path(video_path)
    if gt_path is None:
        raise FileNotFoundError(f"No FIT3D ground truth found for {video_path}")

    gt = load_joints3d(gt_path)  # (F, 25, 3)
    n_gt = gt.shape[0]

    exercise_cls = EXERCISE_REGISTRY.get(exercise_name, EXERCISE_REGISTRY["curl"])
    primary = exercise_cls().primary_joints  # three MediaPipe joint names

    own_model = model is None
    if own_model:
        model = create_pose_model(model_name)

    # Warn once per model class when it cannot produce real depth.
    if not getattr(model, "provides_3d_landmarks", True):
        warnings.warn(
            f"{model.__class__.__name__} is a 2D model (z=0 for all joints). "
            "PA-MPJPE against 3D mocap ground truth reflects 2D-projected alignment "
            "only and is not comparable to a true 3D model.",
            stacklevel=2,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if own_model:
            model.release()
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    result = MocapEvalResult(
        model_name=model_name,
        video_path=str(video_path),
        exercise_name=exercise_name,
        subject=subject_of(video_path),
        total_video_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    # Ordered list of (mp_name, fit3d_idx) for positional accuracy.
    pairs = list(MP_TO_FIT3D.items())

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gt_idx = frame_idx + gt_offset
            if 0 <= gt_idx < n_gt:
                frame = letterbox_resize(frame, resize[0], resize[1])
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = model.extract_landmarks(model.process_image(image_rgb))

                if landmarks is not None:
                    gt_frame = gt[gt_idx]
                    _score_frame(result, landmarks, gt_frame, pairs, primary, min_points)

            frame_idx += 1
    finally:
        cap.release()
        if own_model:
            model.release()

    return result


def _score_frame(
    result: MocapEvalResult,
    landmarks: dict[str, object],
    gt_frame: np.ndarray,
    pairs: list[tuple[str, int]],
    primary: list[str],
    min_points: int,
) -> None:
    """Accumulate PA-MPJPE and angle error for one detected frame."""
    pred_pts: list[list[float]] = []
    gt_pts: list[list[float]] = []
    for mp_name, gt_i in pairs:
        lm = landmarks.get(mp_name)
        if lm is None:
            continue
        pred_pts.append([float(lm.x), float(lm.y), float(lm.z)])
        gt_pts.append([float(x) for x in gt_frame[gt_i]])

    if len(pred_pts) >= min_points:
        src = np.asarray(pred_pts)
        dst = np.asarray(gt_pts)
        aligned = umeyama_align(src, dst, with_scale=True)
        per_joint = np.linalg.norm(aligned - dst, axis=1)  # metres
        result.per_frame_mpjpe_mm.append(float(per_joint.mean() * 1000.0))
        result.evaluated_frames += 1

    # Primary-joint angle error (similarity-invariant -> no alignment needed).
    angle_err = _primary_angle_error(landmarks, gt_frame, primary)
    if angle_err is not None:
        result.per_frame_angle_abs_err.append(angle_err)


def _primary_angle_error(
    landmarks: dict[str, object],
    gt_frame: np.ndarray,
    primary: list[str],
) -> float | None:
    """Absolute difference between predicted and GT primary-joint angle (deg)."""
    if len(primary) != 3 or any(j not in MP_TO_FIT3D for j in primary):
        return None

    pred = []
    for name in primary:
        lm = landmarks.get(name)
        if lm is None:
            return None
        pred.append(np.array([float(lm.x), float(lm.y), float(lm.z)]))

    gt = [gt_frame[MP_TO_FIT3D[name]] for name in primary]

    pred_angle = angle_3d(pred[0], pred[1], pred[2])
    gt_angle = angle_3d(gt[0], gt[1], gt[2])
    if pred_angle is None or gt_angle is None:
        return None
    return abs(pred_angle - gt_angle)


def run_mocap_evaluation(
    model_names: list[str],
    video_paths: list[Path],
    exercise_name: str,
    *,
    resize: tuple[int, int] = (640, 480),
    verbose: bool = True,
) -> list[MocapEvalResult]:
    """Evaluate every (model, clip) pair that has FIT3D ground truth."""
    results: list[MocapEvalResult] = []
    for video_path in video_paths:
        gt_path = resolve_gt_path(Path(video_path))
        if gt_path is None:
            if verbose:
                print(f"  ! skipping {Path(video_path).name}: no FIT3D ground truth")
            continue
        for model_name in model_names:
            if verbose:
                print(f"  mocap-eval: {model_name} on {Path(video_path).name}")
            res = evaluate_clip_against_mocap(
                model_name,
                Path(video_path),
                exercise_name,
                gt_path=gt_path,
                resize=resize,
            )
            if verbose:
                print(
                    f"    PA-MPJPE={res.pa_mpjpe_mm:.1f}mm "
                    f"(p95={res.pa_mpjpe_p95_mm:.1f}mm)  "
                    f"angle MAE={res.angle_mae_deg:.1f}°  "
                    f"coverage={res.coverage:.1%}"
                )
            results.append(res)
    return results
