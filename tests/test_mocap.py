import numpy as np

from fitness_form_ai.evaluation.mocap import angle_3d, subject_of, umeyama_align


def test_umeyama_recovers_known_similarity_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))

    # Apply a known rotation + scale + translation.
    theta = 0.7
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    dst = (2.5 * (rot @ src.T)).T + np.array([1.0, -2.0, 3.0])

    aligned = umeyama_align(src, dst, with_scale=True)
    # Perfect correspondence -> alignment error is ~0.
    assert np.linalg.norm(aligned - dst, axis=1).mean() < 1e-9


def test_angle_3d_right_angle() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(angle_3d(a, b, c) - 90.0) < 1e-6


def test_angle_3d_is_similarity_invariant() -> None:
    a, b, c = np.array([1.0, 0, 0]), np.array([0.0, 0, 0]), np.array([0.0, 2.0, 0])
    base = angle_3d(a, b, c)
    # Scale + translate: interior angle must not change.
    s, t = 3.0, np.array([5.0, 5.0, 5.0])
    assert abs(angle_3d(s * a + t, s * b + t, s * c + t) - base) < 1e-6


def test_subject_parsing() -> None:
    assert subject_of("fit3d/train/s03/videos/cam/squat.mp4") == "s03"
    assert subject_of("/tmp/whatever.mp4") == "unknown"
