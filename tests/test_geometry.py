import numpy as np

from fitness_form_ai.inference.base import LandmarkPoint
from fitness_form_ai.utils.geometry import calculate_angle_3d


def test_right_angle_in_plane() -> None:
    pts = [
        LandmarkPoint(0, 1, 0),
        LandmarkPoint(0, 0, 0),
        LandmarkPoint(1, 0, 0),
    ]
    assert abs(calculate_angle_3d(pts) - 90.0) < 1e-6


def test_straight_limb_is_180_degrees() -> None:
    pts = [
        LandmarkPoint(0, 1, 0),
        LandmarkPoint(0, 0, 0),
        LandmarkPoint(0, -1, 0),
    ]
    assert abs(calculate_angle_3d(pts) - 180.0) < 1e-6


def test_depth_is_not_discarded() -> None:
    # a-b-c form a right angle that lives purely in the y/z plane: it projects
    # onto a single line in x/y, so the old 2D implementation reported 0/180.
    # The true spatial angle is 90 degrees.
    pts = [
        LandmarkPoint(0, 1, 0),
        LandmarkPoint(0, 0, 0),
        LandmarkPoint(0, 0, 1),
    ]
    assert abs(calculate_angle_3d(pts) - 90.0) < 1e-6


def test_degenerate_zero_length_returns_zero() -> None:
    pts = [
        LandmarkPoint(0, 0, 0),
        LandmarkPoint(0, 0, 0),
        LandmarkPoint(1, 0, 0),
    ]
    assert calculate_angle_3d(pts) == 0.0
