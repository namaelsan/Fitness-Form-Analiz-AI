from typing import List

from core.exercise import Exercise
from core.rule import SpeedRule,AngleRule

class OneArmDumbellCurl(Exercise):
    def __init__(self):
        super().__init__(
            "One Arm Dumbell Curl", 
            [
                # AngleRule("Elbow angle rule", ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'], (0, 150)),
                AngleRule("Upper Arm Still", ['RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'], (0, 70))
            ],
            primary_joints=['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'],
            start_phase="concentric"
        )

class Squat(Exercise):
    def __init__(self):
        super().__init__(
            "Squat", 
            [
                AngleRule("Knee angle rule", ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'], (50, 180)),
                # AngleRule("Straight Back", ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'], (100, 180))
            ],
            primary_joints=['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'],
            start_phase="eccentric"
        )

class PushUp(Exercise):
    def __init__(self):
        super().__init__(
            "Push Up", 
            [
                AngleRule("Elbow angle rule", ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'], (50, 180)),
                AngleRule("Straight Back", ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_ANKLE'], (160, 180))
            ],
            primary_joints=['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'],
            start_phase="concentric"
        )