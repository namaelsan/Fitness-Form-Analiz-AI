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
    

class AngleRule(Rule):
    def __init__(self, rule_name: str, joints: List[str], angle_range: Tuple[float, float]):
        super().__init__(rule_name)
        self.joints = joints                # 3 joint names
        self.angle_range = angle_range      # (min_angle, max_angle)

    def apply(self, rep_data: List[RepFrame]) -> bool:
        if not rep_data:
            return False
            
        min_angle = min(frame.angle for frame in rep_data)
        max_angle = max(frame.angle for frame in rep_data)
        
        # Ensure the entire repetition stays within the defined boundaries (0 to 180)
        return min_angle >= self.angle_range[0] and max_angle <= self.angle_range[1]