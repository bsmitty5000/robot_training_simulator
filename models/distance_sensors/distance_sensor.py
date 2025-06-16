from abc import ABC, abstractmethod
from typing import Sequence, Optional
import pygame

from models import constants

class DistanceSensor(ABC):
    angle_deg: float
    max_range_m: float
    max_range_px: float
    last_distance_m: Optional[float]

    def __init__(self, angle_deg: float, max_range_m: float) -> None:
        self.angle_deg = angle_deg
        self.max_range_m = max_range_m
        self.max_range_px = max_range_m * constants.PIXELS_PER_METER
        self.last_distance_m: Optional[float] = None

    @abstractmethod
    def measure(
        self,
        sensor_position: pygame.Vector2,
        robot_angle: float,
        obstacles: Sequence[pygame.sprite.Sprite],
    ) -> float:
        """Return the measured distance to the nearest obstacle, or max_range if none."""
        pass