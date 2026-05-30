"""Session logger — writes a structured JSON log of every tracked frame and rep.

Log file layout
---------------
{
  "session": {
    "started_at":  "2026-05-29T14:03:22.456789",
    "exercise":    "curl",
    "model":       "mediapipe-full",
    "video_source": "webcam" | "<filename>"
  },
  "frames": [
    {
      "t":             1.234,          // seconds since session start
      "angle":         87.3,           // smoothed primary joint angle (°)
      "tracker_state": "CONCENTRIC",
      "smoothing_window": 5,           // adaptive window size at this frame
      "rule_states": {                 // live rule descriptions keyed by name
        "Full contraction": "Elbow flexion: 92 (target depth <= 75)",
        ...
      }
    },
    ...
  ],
  "reps": [
    {
      "rep":          1,
      "valid":        true,
      "failed_rules": [],
      "passed_rules": ["Full contraction", "Full extension", ...],
      "metrics": {                     // per-rule numeric summary
        "Full contraction": {"min": 62.1, "max": 158.3, "mean": 108.7},
        ...
      },
      "duration_s":   2.14,
      "frame_count":  64
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class SessionLogger:
    """Accumulates frame and rep data in memory, flushes to JSON on stop.

    Parameters
    ----------
    log_dir:
        Directory where the log file will be written.
        Created automatically if it does not exist.
    exercise:
        Exercise name string (e.g. ``"curl"``).
    model:
        Model name string (e.g. ``"mediapipe-full"``).
    video_source:
        Path to the video file, or ``None`` for webcam.
    """

    def __init__(
        self,
        log_dir: Path,
        exercise: str,
        model: str,
        video_source: Path | None,
    ) -> None:
        self._log_dir = log_dir
        self._exercise = exercise
        self._model = model
        self._video_source = str(video_source) if video_source else "webcam"

        self._started_at = datetime.now(tz=timezone.utc)
        self._t0: float | None = None          # wall-clock seconds of first frame
        self._frames: list[dict] = []
        self._reps: list[dict] = []
        self._rep_counter = 0
        self._path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path | None:
        """Path the log was (or will be) written to, once logging has started."""
        return self._path

    def log_frame(
        self,
        timestamp: float,
        angle: float | None,
        tracker_state: str,
        smoothing_window: int,
        rule_states: list[tuple[str, str]],
    ) -> None:
        """Record one processed frame.

        Parameters
        ----------
        timestamp:
            Wall-clock time (``time.time()``).
        angle:
            Smoothed primary joint angle in degrees, or ``None`` if no
            landmarks were detected this frame.
        tracker_state:
            Current ``RepTracker.state`` string.
        smoothing_window:
            The adaptive window size that was active this frame.
        rule_states:
            List of ``(rule_name, description)`` tuples from the exercise.
        """
        if self._t0 is None:
            self._t0 = timestamp

        entry: dict = {
            "t": round(timestamp - self._t0, 4),
            "tracker_state": tracker_state,
            "smoothing_window": smoothing_window,
        }
        if angle is not None:
            entry["angle"] = round(angle, 2)
        if rule_states:
            entry["rule_states"] = {name: desc for name, desc in rule_states}

        self._frames.append(entry)

    def log_rep(
        self,
        is_valid: bool,
        failed_rules: list[str],
        passed_rules: list[str],
        metric_series: dict[str, list[float]],
        duration_s: float,
        frame_count: int,
    ) -> None:
        """Record one completed repetition.

        Parameters
        ----------
        is_valid:
            Whether all rules passed.
        failed_rules:
            Names of rules that failed.
        passed_rules:
            Names of rules that passed.
        metric_series:
            Raw value series for each rule, keyed by rule name.
            Used to compute min/max/mean summaries.
        duration_s:
            Wall-clock duration of the rep in seconds.
        frame_count:
            Number of frames in the rep.
        """
        self._rep_counter += 1

        metrics: dict[str, dict] = {}
        for rule_name, values in metric_series.items():
            if values:
                metrics[rule_name] = {
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "mean": round(sum(values) / len(values), 3),
                }

        self._reps.append({
            "rep": self._rep_counter,
            "valid": is_valid,
            "failed_rules": failed_rules,
            "passed_rules": passed_rules,
            "metrics": metrics,
            "duration_s": round(duration_s, 3),
            "frame_count": frame_count,
        })

    def save(self) -> Path:
        """Flush accumulated data to disk and return the log file path."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        ts = self._started_at.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self._exercise}_{ts}.json"
        self._path = self._log_dir / filename

        payload = {
            "session": {
                "started_at": self._started_at.isoformat(),
                "exercise": self._exercise,
                "model": self._model,
                "video_source": self._video_source,
                "total_frames": len(self._frames),
                "total_reps": len(self._reps),
                "valid_reps": sum(1 for r in self._reps if r["valid"]),
            },
            "frames": self._frames,
            "reps": self._reps,
        }

        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        return self._path
