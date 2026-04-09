from abc import ABC
from typing import List, Tuple

from core.rule import Rule
from core.repframe import RepFrame


class Exercise(ABC):
    def __init__(self, name: str, rules: List[Rule], primary_joints: List[str], start_phase: str = "concentric"):
        self.name = name
        self.rules = rules
        self.primary_joints = primary_joints
        self.start_phase = start_phase
        self.progress_criteria = None  # Can be extended (rep count, tempo avg, etc.)

    def apply_rules(self, rep_data: List[RepFrame]) -> Tuple[bool, str]:
        """
        Returns (is_valid, failure_reason)
        """
        failed_reasons = []
        for rule in self.rules:
            if not rule.apply(rep_data):
                failed_reasons.append(rule.rule_name)
        
        if failed_reasons:
            return False, f"Failed: {', '.join(failed_reasons)}"
            
        return True, ""

