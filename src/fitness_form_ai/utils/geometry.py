from __future__ import annotations

import numpy as np


def calculate_angle_3d(points: list[object]) -> float:
    coordinates = [np.array([point.x, point.y, point.z]) for point in points]
    radians = np.arctan2(
        coordinates[2][1] - coordinates[1][1],
        coordinates[2][0] - coordinates[1][0],
    ) - np.arctan2(
        coordinates[0][1] - coordinates[1][1],
        coordinates[0][0] - coordinates[1][0],
    )
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return float(angle)
