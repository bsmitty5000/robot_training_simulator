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
    
    @property
    @abstractmethod
    def control_input_size(self) -> int:
        """Number of control inputs expected by this robot in update_kinematics."""
        pass
    
    @property
    @abstractmethod
    def control_input_upper_limit(self) -> int:
        """Upper limit on control input inclusive."""
        pass
    
    @property
    @abstractmethod
    def control_input_lower_limit(self) -> int:
        """Lower limit on control input inclusive."""
        pass

    @abstractmethod
    def update(self, dt: float, obstacles: Sequence[pygame.sprite.Sprite]) -> None:
        """Update the robot's state (position, sensors, etc)."""
        pass

    @abstractmethod
    def update_kinematics(self, dt: float, control_inputs: Sequence[float]) -> None:
        """
        Update the robot's kinematics using the provided control inputs.
        """
        pass

    @abstractmethod
    def update_coordinates(self) -> None:
        """Update the robot's coordinates and image for rendering."""
        pass

    @abstractmethod
    def detect(self, obstacles: Sequence[pygame.sprite.Sprite]) -> None:
        """Update the robot's sensors based on obstacles."""
        pass