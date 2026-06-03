from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np

from fitness_form_ai.inference.base import LandmarkPoint, PoseModel


class MeTRAbsModel(PoseModel):
    """Direct single-image **3D** pose estimator (https://github.com/isarandi/metrabs).

    Unlike MoveNet / YOLOv8 (which are 2D-only and hardcode ``z=0``), MeTRAbs
    regresses metric-scale 3D joint positions directly from a single RGB image,
    so its landmarks carry a meaningful depth (z) component. This makes it a
    valid alternative to MediaPipe BlazePose for the 3D-dependent parts of the
    system: PA-MPJPE against the FIT3D mocap ground truth, ``calculate_angle_3d``,
    and depth-sensitive rules (shoulder-press symmetry, wrist orientation, ...).

    The bundled TF SavedModel exposes ``detect_poses(image, skeleton=...)`` which
    performs person detection + 3D pose estimation in one call and returns a dict
    of tensors (``boxes``, ``poses3d``, ``poses2d``). We request the ``h36m_17``
    skeleton because every joint the rest of the system needs (shoulders, elbows,
    wrists, hips, knees, ankles) has an unambiguous named analogue there, and we
    map those names onto the MediaPipe ``PoseLandmark`` naming convention used
    everywhere else.
    """

    # MeTRAbs produces real metric-scale 3D coordinates (z is meaningful).
    provides_3d_landmarks = True

    # Official SavedModel archives (zipped). The lighter ``mob3l`` variant is the
    # default; ``eff2l`` is the heavier / more accurate one. Override the download
    # host or point at a pre-extracted SavedModel via the METRABS_MODEL_DIR env var
    # if these URLs ever move.
    _MODEL_URLS = {
        "mob3l": "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3l_y4t.zip",
        "eff2l": "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4.zip",
    }

    # MeTRAbs h36m_17 short joint names -> MediaPipe PoseLandmark names. Only the
    # joints with a clean one-to-one correspondence (and that the downstream
    # MP_TO_FIT3D mapping in mocap.py actually consumes) are included.
    _METRABS_TO_MP = {
        "lsho": "LEFT_SHOULDER",
        "lelb": "LEFT_ELBOW",
        "lwri": "LEFT_WRIST",
        "rsho": "RIGHT_SHOULDER",
        "relb": "RIGHT_ELBOW",
        "rwri": "RIGHT_WRIST",
        "lhip": "LEFT_HIP",
        "lkne": "LEFT_KNEE",
        "lank": "LEFT_ANKLE",
        "rhip": "RIGHT_HIP",
        "rkne": "RIGHT_KNEE",
        "rank": "RIGHT_ANKLE",
    }

    def __init__(self, variant: str = "mob3l", *, skeleton: str = "h36m_17") -> None:
        import tensorflow as tf

        self.variant = variant
        self.skeleton = skeleton

        # Test-time augmentation count for detect_poses. The SavedModel default
        # is 5 (5 augmented forward passes per frame -> ~5x slower). For batch
        # evaluation 1 (plain single-crop inference) is dramatically faster with
        # only a small accuracy cost. Override with METRABS_NUM_AUG.
        try:
            self.num_aug = max(1, int(os.environ.get("METRABS_NUM_AUG", "1")))
        except ValueError:
            self.num_aug = 1

        model_path = self._resolve_model_dir(variant)
        self.model = tf.saved_model.load(str(model_path))

        # Resolve joint name -> index for the requested skeleton from the loaded
        # model itself (rather than hardcoding an order), then precompute the
        # MeTRAbs-index -> MediaPipe-name mapping we actually emit.
        joint_names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in self.model.per_skeleton_joint_names[skeleton].numpy()
        ]
        self._joint_names = joint_names
        self.keypoint_mapping: dict[int, str] = {
            joint_names.index(mb_name): mp_name
            for mb_name, mp_name in self._METRABS_TO_MP.items()
            if mb_name in joint_names
        }
        # Edges (for drawing the 2D skeleton overlay).
        try:
            self._edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
        except Exception:
            self._edges = np.empty((0, 2), dtype=np.int64)

    def _resolve_model_dir(self, variant: str) -> Path:
        """Return a local SavedModel directory, downloading/extracting if needed."""
        env_dir = os.environ.get("METRABS_MODEL_DIR")
        if env_dir:
            return Path(env_dir)

        if variant not in self._MODEL_URLS:
            raise ValueError(
                f"Unknown MeTRAbs variant '{variant}'. "
                f"Valid variants: {', '.join(self._MODEL_URLS)}"
            )

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        extracted = model_dir / f"metrabs_{variant}"

        if not (extracted / "saved_model.pb").exists():
            # The archive may extract into a nested folder; find the SavedModel root.
            url = self._MODEL_URLS[variant]
            zip_path = model_dir / f"metrabs_{variant}.zip"
            if not zip_path.exists():
                print(f"Downloading MeTRAbs {variant} to {zip_path}...")
                urlretrieve(url, str(zip_path))
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted)
            if not (extracted / "saved_model.pb").exists():
                for candidate in extracted.rglob("saved_model.pb"):
                    return candidate.parent

        return extracted

    def process_image(self, image: np.ndarray) -> Any:
        import tensorflow as tf

        # detect_poses expects a uint8 RGB HxWx3 tensor and runs detection +
        # 3D pose estimation in one shot.
        image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        try:
            return self.model.detect_poses(
                image_tensor,
                skeleton=self.skeleton,
                detector_threshold=0.3,
                num_aug=self.num_aug,
                detector_flip_aug=False,
            )
        except (TypeError, ValueError):
            # Older/newer SavedModel signature without these kwargs — fall back.
            return self.model.detect_poses(
                image_tensor,
                skeleton=self.skeleton,
                detector_threshold=0.3,
            )

    def extract_landmarks(self, results: Any) -> dict[str, LandmarkPoint] | None:
        if results is None:
            return None

        poses3d = np.asarray(results["poses3d"])  # (N, J, 3), millimetres
        if poses3d.size == 0 or poses3d.shape[0] == 0:
            return None

        # Use the first detected person. poses3d is in camera coordinates (mm);
        # convert to metres to match MediaPipe's world-landmark scale.
        pose = poses3d[0].astype(np.float64) / 1000.0

        landmarks: dict[str, LandmarkPoint] = {}
        for index, name in self.keypoint_mapping.items():
            if index >= len(pose):
                continue
            x, y, z = pose[index]
            landmarks[name] = LandmarkPoint(x=float(x), y=float(y), z=float(z))

        if not landmarks:
            return None

        # ---- Coordinate-system normalization -----------------------------
        # MediaPipe world landmarks are hip-centred (origin at the midpoint of
        # the two hips). MeTRAbs returns root-relative-but-camera-positioned
        # coordinates, so we re-centre on the hip midpoint here. This puts the
        # output in the same convention the rest of the system expects and lets
        # the Umeyama similarity alignment in mocap.py recover the remaining
        # rotation/scale cleanly.
        lh = landmarks.get("LEFT_HIP")
        rh = landmarks.get("RIGHT_HIP")
        if lh is not None and rh is not None:
            cx = (lh.x + rh.x) / 2.0
            cy = (lh.y + rh.y) / 2.0
            cz = (lh.z + rh.z) / 2.0
            for lm in landmarks.values():
                lm.x -= cx
                lm.y -= cy
                lm.z -= cz

        return landmarks or None

    def draw_landmarks(self, image: np.ndarray, results: Any) -> None:
        if results is None:
            return
        poses2d = np.asarray(results["poses2d"])  # (N, J, 2) image pixels
        if poses2d.size == 0 or poses2d.shape[0] == 0:
            return
        pose = poses2d[0]

        for j1, j2 in self._edges:
            if j1 < len(pose) and j2 < len(pose):
                p1 = (int(pose[j1][0]), int(pose[j1][1]))
                p2 = (int(pose[j2][0]), int(pose[j2][1]))
                cv2.line(image, p1, p2, (0, 255, 0), 2)

        for x, y in pose:
            cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    def release(self) -> None:
        self.model = None
