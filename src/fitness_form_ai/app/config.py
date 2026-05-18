from dataclasses import dataclass
from pathlib import Path
from typing import Final

SUPPORTED_EXERCISES: Final[tuple[str, ...]] = (
    "curl",
    "squat",
    "deadlift",
    "shoulder_press",
    "lateral_raise",
)
SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "mediapipe-lite",
    "mediapipe-full",
    "mediapipe-heavy",
    "yolov8",
    "movenet-lightning",
    "movenet-thunder",
)


@dataclass(slots=True)
class AppConfig:
    exercise: str = "curl"
    model: str = "mediapipe-full"
    video: Path | None = None


@dataclass(slots=True)
class TrackingConfig:
    min_rom: float = 20.0
    texture_width: int = 1200
    texture_height: int = 800
