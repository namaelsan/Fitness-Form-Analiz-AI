"""Tracker tests against the current RepTracker API.

The tracker derives its smoothing window from an injected ``AdaptiveSmoother``
(there is no settable ``smoothing_window`` attribute), and it debounces every
state transition by ``confirm_frames`` consecutive confirming frames. These
tests therefore:

* inject a no-op smoother (``min_frames == max_frames == 1``) so the smoothed
  angle equals the raw angle and velocity is a clean first difference, and
* use ``confirm_frames=1`` so a single confirming frame advances the state
  machine, keeping the fixtures short and the expected transitions explicit.

Both exercise variants begin on a *decreasing* primary-joint angle (a curl
flexes the elbow, a squat bends the knee), so each fixture starts with a
downward swing to trigger rep onset, reverses upward to enter the return
phase, and then satisfies that phase's completion condition.
"""

from fitness_form_ai.domain.smoother import AdaptiveSmoother
from fitness_form_ai.domain.tracker import RepTracker


def _no_smoothing() -> AdaptiveSmoother:
    """A smoother whose window is always exactly one frame (no smoothing)."""
    return AdaptiveSmoother(min_frames=1, max_frames=1)


def _make_tracker(start_phase: str) -> RepTracker:
    tracker = RepTracker(
        start_phase=start_phase,
        min_rom=20.0,
        smoother=_no_smoothing(),
        confirm_frames=1,
    )
    tracker.velocity_threshold = 10.0
    return tracker


def _run(tracker: RepTracker, frames: list[tuple[float, float]]) -> bool:
    completed = False
    for angle, timestamp in frames:
        completed = tracker.add_frame(angle, timestamp) or completed
    return completed


def test_concentric_tracker_completes_rep() -> None:
    tracker = _make_tracker("concentric")

    # Down (onset + concentric phase), reverse up (enter eccentric return),
    # continue up to complete the rep at the top.
    frames = [
        (120.0, 0.0),
        (100.0, 1.0),  # v=-20 -> rep onset, enter CONCENTRIC (start angle 100)
        (80.0, 2.0),
        (60.0, 3.0),
        (100.0, 4.0),  # reversal up but ROM from start still 0
        (140.0, 5.0),  # v=+40, ROM=40 -> enter ECCENTRIC return (peak 140)
        (160.0, 6.0),  # moving up, ROM from peak=20 -> COMPLETED
    ]

    assert _run(tracker, frames) is True
    rep = tracker.extract_rep()
    assert rep is not None
    assert tracker.state == "IDLE"


def test_eccentric_tracker_completes_rep() -> None:
    tracker = _make_tracker("eccentric")

    # Down (onset + eccentric phase), reverse up (enter concentric return),
    # then slow near the top so the eccentric-variant completion (velocity
    # below threshold with sufficient ROM from the reversal point) fires.
    frames = [
        (120.0, 0.0),
        (100.0, 1.0),  # v=-20 -> rep onset, enter ECCENTRIC (start angle 100)
        (80.0, 2.0),
        (60.0, 3.0),
        (90.0, 4.0),   # reversal up but ROM from start still <20
        (120.0, 5.0),  # v=+30, ROM=20 -> enter CONCENTRIC return (peak 120)
        (145.0, 6.0),  # still rising fast (v=25), not yet complete
        (146.0, 7.0),  # v=+1 < threshold, ROM from peak=26 -> COMPLETED
    ]

    assert _run(tracker, frames) is True
    rep = tracker.extract_rep()
    assert rep is not None
    assert tracker.state == "IDLE"
