from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional

from models.distance_sensors.distance_sensor import DistanceSensor
from sim import core

class RobotBase(ABC):
    position: core.Vector2
    angle_deg: float
    velocity: float
    angular_velocity: float
    distance_sensors: Optional[Sequence[DistanceSensor]]

    def __init__(self, 
                 distance_sensors: Optional[Sequence[DistanceSensor]]=None) -> None:
        super().__init__()
        self.position = core.Vector2(0, 0)
        self.angle_deg = 0.0
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
    
    @property
    @abstractmethod
    def x_coordinate(self):
        """Current X position."""
        pass
    
    @property
    @abstractmethod
    def y_coordinate(self):
        """Current Y position."""
        pass

    @abstractmethod
    def reset(self,
              x: float,
              y: float,) -> None:
        """Reset all internals and set position."""
        pass

    @abstractmethod
    def move(self, dt: float, control_inputs: Sequence[float]) -> None:
        """
        Use control_inputs to update the robot's position after dt seconds
        """
        pass

    @abstractmethod
    def detect(self, obstacles: Sequence[core.Shape]) -> None:
        """Update the robot's sensors based on obstacles."""
        pass