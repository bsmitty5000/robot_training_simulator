import pygame
from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional

from models.distance_sensors.distance_sensor import DistanceSensor

class RobotBase(pygame.sprite.Sprite, ABC):
    position: pygame.Vector2
    angle: float
    velocity: float
    angular_velocity: float
    distance_sensors: Optional[Sequence[DistanceSensor]]

    def __init__(self, 
                 x: float, 
                 y: float, 
                 distance_sensors: Optional[Sequence[DistanceSensor]]=None) -> None:
        super().__init__()
        self.position = pygame.Vector2(x, y)
        self.angle = 0.0
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.distance_sensors = distance_sensors if distance_sensors is not None else []

    @abstractmethod
    def update(self, dt: float, obstacles: Sequence[pygame.sprite.Sprite]) -> None:
        """Update the robot's state (position, sensors, etc)."""
        pass

    @abstractmethod
    def update_kinematics(self, dt: float) -> None:
        """Update the robot's kinematics (velocity, angle, etc)."""
        pass

    @abstractmethod
    def update_coordinates(self) -> None:
        """Update the robot's coordinates and image for rendering."""
        pass

    @abstractmethod
    def detect(self, obstacles: Sequence[pygame.sprite.Sprite]) -> None:
        """Update the robot's sensors based on obstacles."""
        pass