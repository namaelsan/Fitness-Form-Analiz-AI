from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class LandmarkPoint:
    x: float
    y: float
    z: float


class PoseModel(ABC):
    # Subclasses that only produce 2D landmarks (z=0) should override this to False.
    # PA-MPJPE against 3D ground truth is still computable but reflects 2D-projected
    # alignment only; results are not comparable to a true 3D model.
    provides_3d_landmarks: bool = True

    @abstractmethod
    def process_image(self, image: np.ndarray) -> Any:
        raise NotImplementedError

    @abstractmethod
    def extract_landmarks(self, results: Any) -> dict[str, LandmarkPoint] | None:
        raise NotImplementedError

    @abstractmethod
    def draw_landmarks(self, image: np.ndarray, results: Any) -> None:
        raise NotImplementedError

    def release(self) -> None:
        return None
