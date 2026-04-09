from abc import ABC, abstractmethod
from typing import List, Tuple
from core.repframe import RepFrame

class Rule(ABC):
    def __init__(self, rule_name: str):
        self.rule_name = rule_name

    @abstractmethod
    def apply(self, rep_data: List[RepFrame]) -> bool:
        """
        Returns True if rule is satisfied over the completed repetition, otherwise False.
        """
        pass

    @abstractmethod
    def get_current_value(self, landmarks: dict) -> float:
        """
        Returns the current value (e.g. angle) for this rule based on landmarks.
        """
        pass
    
class SpeedRule(Rule):
    def __init__(self, rule_name: str, joints: List[str], max_speed: float):
        super().__init__(rule_name)
        self.joints = joints          # 3 joint names
        self.max_speed = max_speed    # degree/second threshold

    def apply(self, rep_data: List[RepFrame]) -> bool:
        if not rep_data:
            return True
            
        max_rep_speed = max(abs(frame.velocity) for frame in rep_data)
        return max_rep_speed <= self.max_speed

    def get_current_value(self, landmarks: dict) -> float:
        # SpeedRule doesn't depend on landmarks alone, but we could return 0.0 or last frame velocity if we had it.
        # For now, let's just return 0.0.
        return 0.0
    

class AngleRule(Rule):
    def __init__(self, rule_name: str, joints: List[str], angle_range: Tuple[float, float]):
        super().__init__(rule_name)
        self.joints = joints                # 3 joint names
        self.angle_range = angle_range      # (min_angle, max_angle)

    def apply(self, rep_data: List[RepFrame]) -> bool:
        if not rep_data:
            return False
            
        if self.joints and rep_data[0].landmarks:
            from util import read_landmark, calculate_angle_3d
            
            angles = []
            for frame in rep_data:
                try:
                    joints_data = [read_landmark(j, frame.landmarks) for j in self.joints]
                    angles.append(calculate_angle_3d(joints_data))
                except Exception:
                    angles.append(frame.angle)
            min_angle = min(angles)
            max_angle = max(angles)
        else:
            min_angle = min(frame.angle for frame in rep_data)
            max_angle = max(frame.angle for frame in rep_data)
        
        # Ensure the entire repetition stays within the defined boundaries (0 to 180)
        return min_angle >= self.angle_range[0] and max_angle <= self.angle_range[1]

    def get_current_value(self, landmarks: dict) -> float:
        if not landmarks or not self.joints:
            return 0.0
        
        from util import read_landmark, calculate_angle_3d
        try:
            joints_data = [read_landmark(j, landmarks) for j in self.joints]
            if None in joints_data:
                return 0.0
            return calculate_angle_3d(joints_data)
        except Exception:
            return 0.0