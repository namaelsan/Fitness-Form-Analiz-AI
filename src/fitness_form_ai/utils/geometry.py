from __future__ import annotations

import cv2
import numpy as np


def letterbox_resize(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize *frame* to fit within (target_w, target_h) without stretching.

    The image is scaled down (or up) to fit the longest side, then the
    remaining space is filled with *pad_color* so the output is exactly
    target_w × target_h.
    """
    src_h, src_w = frame.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, frame.shape[2]), pad_color, dtype=frame.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def calculate_angle_3d(points: list[object]) -> float:
    """Interior angle (degrees) at ``points[1]`` of the chain a-b-c.

    Uses the full 3D coordinates of all three points. The previous
    implementation projected onto the image x/y plane only — despite the
    ``_3d`` name — which discarded depth and made the angle dependent on the
    subject's orientation to the camera. This computes the true spatial angle
    via the dot product of the two bone vectors, so it is invariant to how the
    body is rotated relative to the camera.

    Returns ``0.0`` when either bone has zero length (degenerate / missing
    landmark), preserving the float contract callers rely on.
    """
    a, b, c = (np.array([p.x, p.y, p.z], dtype=float) for p in points)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0
    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
