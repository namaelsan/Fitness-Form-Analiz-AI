from typing import List, Optional
import time
from core.repframe import RepFrame

class RepTracker:
    def __init__(self, start_phase="concentric", min_rom=20.0):
        self.angle_history: List[RepFrame] = [] 
        self.state = "IDLE"
        self._current_rep_data: List[RepFrame] = []
        
        self.start_phase = start_phase
        self.min_rom = min_rom
        self.phase_start_angle = None
        self._phase2_peak = None
        
        # Hyperparameters
        self.smoothing_window = 5
        self.velocity_threshold = 10.0 # degrees per second

    def add_frame(self, angle: float, timestamp: float) -> bool:
        """
        Adds a new frame, calculates smoothed angle and velocity.
        Updates FSM state based on velocity and angle.
        Returns True if a repetition has been completed this frame.
        """
        frame = RepFrame(angle=angle, timestamp=timestamp)
        self.angle_history.append(frame)
        
        smoothed_angle = self._smooth_angle()
        
        velocity = self._calculate_velocity(smoothed_angle, timestamp)
        self.angle_history[-1].velocity = velocity
        
        self._current_rep_data.append(RepFrame(angle=smoothed_angle, timestamp=timestamp, velocity=velocity))
        
        self._update_state(smoothed_angle, velocity)
        
        return self.state == "COMPLETED"

    def _smooth_angle(self) -> float:
        if len(self.angle_history) < self.smoothing_window:
            return self.angle_history[-1].angle
        
        recent_angles = [frame.angle for frame in self.angle_history[-self.smoothing_window:]]
        return sum(recent_angles) / len(recent_angles)

    def _calculate_velocity(self, smoothed_angle: float, timestamp: float) -> float:
        if len(self.angle_history) < 2:
            return 0.0
        
        prev_data = self.angle_history[-2]
        time_diff = timestamp - prev_data.timestamp
        if time_diff <= 0:
            return 0.0
            
        angle_diff = smoothed_angle - prev_data.angle
        return angle_diff / time_diff

    def _update_state(self, angle: float, velocity: float):
        """
        Finite State Machine for adaptive Repetition phase tracking based on extrema.
        """
        if self.state == "IDLE":
            if self.start_phase == "concentric" and velocity < -self.velocity_threshold:
                self.state = "PHASE_1"
                self.phase_start_angle = angle
            elif self.start_phase == "eccentric" and velocity > self.velocity_threshold:
                self.state = "PHASE_1"
                self.phase_start_angle = angle

        elif self.state == "PHASE_1":
            rom = abs(angle - self.phase_start_angle)
            if self.start_phase == "concentric" and velocity > self.velocity_threshold and rom >= self.min_rom:
                self.state = "PHASE_2"
                self._phase2_peak = angle
            elif self.start_phase == "eccentric" and velocity < -self.velocity_threshold and rom >= self.min_rom:
                self.state = "PHASE_2"
                self._phase2_peak = angle
                
        elif self.state == "PHASE_2":
            rom = abs(angle - self._phase2_peak)
            if self.start_phase == "concentric":
                if velocity < self.velocity_threshold and rom >= self.min_rom:
                    self.state = "COMPLETED"
            elif self.start_phase == "eccentric":
                if velocity > -self.velocity_threshold and rom >= self.min_rom:
                    self.state = "COMPLETED"

    def extract_rep(self) -> Optional[List[RepFrame]]:
        """
        Returns the data for the completed rep and clears the internal buffer.
        """
        if self.state == "COMPLETED":
            rep_data = self._current_rep_data.copy()
            self._current_rep_data.clear()
            self.state = "IDLE"
            
            if len(self.angle_history) > 100:
                self.angle_history = self.angle_history[-50:]
            return rep_data
        return None
