from dataclasses import dataclass, field
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
    # Adaptive smoother parameters
    smooth_target_window_ms: float = 150.0  # time span to smooth over
    smooth_ema_alpha: float = 0.15          # EMA adaptation speed (0 < α ≤ 1)
    smooth_min_frames: int = 3              # minimum window size
    smooth_max_frames: int = 15             # maximum window size
    # State-machine debounce: consecutive frames required to confirm a phase transition
    confirm_frames: int = 3
