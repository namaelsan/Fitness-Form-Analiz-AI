from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RepFrame:
    angle: float
    timestamp: float
    velocity: float = 0.0
    landmarks: Any = None
