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
