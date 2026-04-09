from __future__ import annotations

from fitness_form_ai.domain.rep_frame import RepFrame


class RepTracker:
    def __init__(self, start_phase: str = "concentric", min_rom: float = 20.0) -> None:
        self.angle_history: list[RepFrame] = []
        self.state = "IDLE"
        self._current_rep_data: list[RepFrame] = []

        self.start_phase = start_phase
        self.min_rom = min_rom
        self.phase_start_angle: float | None = None
        self._phase2_peak: float | None = None

        self.smoothing_window = 5
        self.velocity_threshold = 10.0

    def add_frame(self, angle: float, timestamp: float, landmarks: object = None) -> bool:
        frame = RepFrame(angle=angle, timestamp=timestamp, landmarks=landmarks)
        self.angle_history.append(frame)

        smoothed_angle = self._smooth_angle()
        velocity = self._calculate_velocity(smoothed_angle, timestamp)
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

        if len(self.angle_history) > 100:
            self.angle_history = self.angle_history[-50:]
        return rep_data

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

    def _smooth_angle(self) -> float:
        if len(self.angle_history) < self.smoothing_window:
            return self.angle_history[-1].angle

        recent_angles = [frame.angle for frame in self.angle_history[-self.smoothing_window :]]
        return sum(recent_angles) / len(recent_angles)

    def _calculate_velocity(self, smoothed_angle: float, timestamp: float) -> float:
        if len(self.angle_history) < 2:
            return 0.0

        prev_frame = self.angle_history[-2]
        time_diff = timestamp - prev_frame.timestamp
        if time_diff <= 0:
            return 0.0

        angle_diff = smoothed_angle - prev_frame.angle
        return angle_diff / time_diff

    def _update_state(self, angle: float, velocity: float) -> None:
        if self.state == "IDLE":
            if self.start_phase == "concentric" and velocity < -self.velocity_threshold:
                self.state = "PHASE_1"
                self.phase_start_angle = angle
            elif self.start_phase == "eccentric" and velocity > self.velocity_threshold:
                self.state = "PHASE_1"
                self.phase_start_angle = angle
            return

        if self.state == "PHASE_1":
            if self.phase_start_angle is None:
                return

            rom = abs(angle - self.phase_start_angle)
            if self.start_phase == "concentric" and velocity > self.velocity_threshold and rom >= self.min_rom:
                self.state = "PHASE_2"
                self._phase2_peak = angle
            elif self.start_phase == "eccentric" and velocity < -self.velocity_threshold and rom >= self.min_rom:
                self.state = "PHASE_2"
                self._phase2_peak = angle
            return

        if self.state == "PHASE_2":
            if self._phase2_peak is None:
                return

            rom = abs(angle - self._phase2_peak)
            if self.start_phase == "concentric":
                if velocity < self.velocity_threshold and rom >= self.min_rom:
                    self.state = "COMPLETED"
            elif self.start_phase == "eccentric":
                if velocity > -self.velocity_threshold and rom >= self.min_rom:
                    self.state = "COMPLETED"
