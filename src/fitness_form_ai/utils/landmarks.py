from __future__ import annotations


def read_landmark(name: str, landmarks: dict[str, object], mp_pose: object = None) -> object | None:
    if isinstance(landmarks, dict):
        return landmarks.get(name)

    if mp_pose is None:
        return None
    return landmarks[mp_pose.PoseLandmark[name].value]
