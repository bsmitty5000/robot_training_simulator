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
    
    def get_sensor_direction(self, robot_angle: float = 0) -> pygame.Vector2:
        """Returns a unit vector for the sensor's world direction."""
        return pygame.Vector2(0, -1).rotate(self.angle_deg - robot_angle)

    @abstractmethod
    def draw(self, surface: pygame.Surface, sensor_position: pygame.Vector2) -> None:
        """Draw the sensor visualization on the given surface."""
        pass

    @abstractmethod
    def measure(
        self,
        sensor_position: pygame.Vector2,
        robot_angle: float,
        obstacles: Sequence[pygame.sprite.Sprite],
    ) -> float:
        """Return the measured distance to the nearest obstacle, or max_range if none."""
        pass