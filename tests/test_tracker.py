from fitness_form_ai.domain.tracker import RepTracker


def test_concentric_tracker_completes_rep() -> None:
    tracker = RepTracker(start_phase="concentric", min_rom=20.0)
    tracker.smoothing_window = 1
    tracker.velocity_threshold = 10.0

    frames = [
        (120.0, 0.0),
        (90.0, 1.0),
        (60.0, 2.0),
        (95.0, 3.0),
        (125.0, 4.0),
        (155.0, 5.0),
        (158.0, 6.0),
    ]

    completed = False
    for angle, timestamp in frames:
        completed = tracker.add_frame(angle, timestamp) or completed

    assert completed is True
    rep = tracker.extract_rep()
    assert rep is not None
    assert tracker.state == "IDLE"


def test_eccentric_tracker_completes_rep() -> None:
    tracker = RepTracker(start_phase="eccentric", min_rom=20.0)
    tracker.smoothing_window = 1
    tracker.velocity_threshold = 10.0

    frames = [
        (60.0, 0.0),
        (90.0, 1.0),
        (120.0, 2.0),
        (85.0, 3.0),
        (55.0, 4.0),
        (25.0, 5.0),
        (22.0, 6.0),
    ]

    completed = False
    for angle, timestamp in frames:
        completed = tracker.add_frame(angle, timestamp) or completed

    assert completed is True
