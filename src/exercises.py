from typing import List

from core.exercise import Exercise
from core.rule import SpeedRule,AngleRule

class OneArmDumbellCurl(Exercise):
    def __init__(self):
        super().__init__(
            "One Arm Dumbell Curl", 
            [AngleRule("Elbow angle rule", ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'], (0, 150))],
            primary_joints=['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'],
            start_phase="concentric"
        )