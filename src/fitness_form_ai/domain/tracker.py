from __future__ import annotations

from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.smoother import AdaptiveSmoother


class RepTracker:
    def __init__(
        self,
        start_phase: str = "concentric",
        min_rom: float = 20.0,
        smoother: AdaptiveSmoother | None = None,
        confirm_frames: int = 3,
    ) -> None:
        self.start_phase = start_phase
        self.min_rom = min_rom
        self.velocity_threshold = 10.0
        self.confirm_frames = confirm_frames

        self._smoother = smoother if smoother is not None else AdaptiveSmoother()

        self.angle_history: list[RepFrame] = []
        self.state = "IDLE"
        self._current_rep_data: list[RepFrame] = []

        self.phase_start_angle: float | None = None
        self._phase2_peak: float | None = None

        # Bug 1 fix: track previous smoothed angle so velocity is
        # always smoothed-vs-smoothed, never smoothed-vs-raw.
        self._last_smoothed: float | None = None
        self._last_smoothed_ts: float | None = None

        # Bug 2 fix: count consecutive frames that satisfy the current
        # transition condition before committing to a state change.
        self._trigger_count: int = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def smoothing_window(self) -> int:
        """Current adaptive smoothing window size (in frames)."""
        return self._smoother.window_size

    @property
    def active_phase(self) -> str:
        return self.start_phase.upper()

    @property
    def return_phase(self) -> str:
        return "ECCENTRIC" if self.start_phase == "concentric" else "CONCENTRIC"

    @property
    def current_velocity(self) -> float:
        if not self.angle_history:
            return 0.0
        return self.angle_history[-1].velocity

    @property
    def current_rep_duration(self) -> float:
        if len(self._current_rep_data) < 2:
            return 0.0
        return self._current_rep_data[-1].timestamp - self._current_rep_data[0].timestamp

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def add_frame(self, angle: float, timestamp: float, landmarks: object = None) -> bool:
        # Update adaptive smoother with inter-frame interval.
        if self.angle_history:
            interval_ms = (timestamp - self.angle_history[-1].timestamp) * 1000.0
            self._smoother.update(interval_ms)

        frame = RepFrame(angle=angle, timestamp=timestamp, landmarks=landmarks)
        self.angle_history.append(frame)

        smoothed_angle = self._smooth_angle()

        # Bug 1 fix: compute velocity from smoothed-vs-smoothed.
        velocity = self._calculate_velocity(smoothed_angle, timestamp)

        # Store smoothed values for the next frame's velocity calculation.
        self._last_smoothed = smoothed_angle
        self._last_smoothed_ts = timestamp

        self.angle_history[-1].velocity = velocity

        self._current_rep_data.append(
            RepFrame(
                angle=smoothed_angle,
                timestamp=timestamp,
                velocity=velocity,
                landmarks=landmarks,
            )
        )
        self._update_state(smoothed_angle, velocity)
        return self.state == "COMPLETED"

    def extract_rep(self) -> list[RepFrame] | None:
        if self.state != "COMPLETED":
            return None

        rep_data = self._current_rep_data.copy()
        self._current_rep_data.clear()
        self.state = "IDLE"
        self._trigger_count = 0

        if len(self.angle_history) > 100:
            self.angle_history = self.angle_history[-50:]
        return rep_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_angle(self) -> float:
        window = self._smoother.window_size
        recent = (
            self.angle_history[-window:]
            if len(self.angle_history) >= window
            else self.angle_history
        )
        return sum(f.angle for f in recent) / len(recent)

    def _calculate_velocity(self, smoothed_angle: float, timestamp: float) -> float:
        """Velocity in degrees/second, always computed from smoothed-vs-smoothed."""
        if self._last_smoothed is None or self._last_smoothed_ts is None:
            return 0.0
        dt = timestamp - self._last_smoothed_ts
        if dt <= 0:
            return 0.0
        return (smoothed_angle - self._last_smoothed) / dt

    def _update_state(self, angle: float, velocity: float) -> None:
        """State machine with per-transition debounce (confirm_frames)."""

        if self.state == "IDLE":
            # Both concentric-first and eccentric-first exercises begin with the
            # primary joint angle *decreasing* (curl: elbow flexes; squat: knee
            # bends), so the onset trigger is always a sufficiently negative velocity.
            triggered = velocity < -self.velocity_threshold
            if triggered:
                self._trigger_count += 1
                if self._trigger_count >= self.confirm_frames:
                    self.state = self.active_phase
                    self.phase_start_angle = angle
                    self._trigger_count = 0
            else:
                self._trigger_count = 0
            return

        if self.state == self.active_phase:
            if self.phase_start_angle is None:
                return
            rom = abs(angle - self.phase_start_angle)
            # The return phase starts when the angle reverses (increases) after
            # enough ROM — same direction for both start_phase variants.
            triggered = velocity > self.velocity_threshold and rom >= self.min_rom
            if triggered:
                self._trigger_count += 1
                if self._trigger_count >= self.confirm_frames:
                    self._phase2_peak = angle
                    self.state = self.return_phase
                    self._trigger_count = 0
            else:
                self._trigger_count = 0
            return

        if self.state == self.return_phase:
            if self._phase2_peak is None:
                return
            rom = abs(angle - self._phase2_peak)
            triggered = (
                (self.start_phase == "concentric" and velocity > -self.velocity_threshold and rom >= self.min_rom)
                or (self.start_phase == "eccentric" and velocity < self.velocity_threshold and rom >= self.min_rom)
            )
            if triggered:
                self._trigger_count += 1
                if self._trigger_count >= self.confirm_frames:
                    self.state = "COMPLETED"
                    self._trigger_count = 0
            else:
                self._trigger_count = 0
