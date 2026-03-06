from dataclasses import dataclass

@dataclass
class RepFrame:
    angle: float
    timestamp: float
    velocity: float = 0.0
