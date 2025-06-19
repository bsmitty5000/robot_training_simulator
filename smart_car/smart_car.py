from sim import core
from typing import Sequence, Optional, Any
from ml_stuff.decision_base import DecisionBase
from models.robots.robot import RobotBase
from models.distance_sensors.distance_sensor import DistanceSensor

class SmartCar():
    def __init__(
        self,
        robot: Optional[RobotBase] = None,
        decision_maker: Optional[DecisionBase] = None):
        
        # Compose a RobotBase instance
        self.robot = robot
        self.decision_maker = decision_maker

        # --- Consistency check ---
        if self.robot and self.decision_maker:
            decision_maker_output = getattr(self.decision_maker, "output_size", None)
            robot_input = getattr(self.robot, "control_input_size", None)
            if decision_maker_output != robot_input:
                raise ValueError(
                    f"Decision maker output length ({decision_maker_output}) does not match "
                    f"robot control input size ({robot_input})"
                )

    def update(self, dt: float, obstacles: Sequence[core.Shape]) -> None:
        # Update sensors
        self.robot.detect(obstacles)
        
        # Gather sensor readings
        sensor_distances = [
            s.last_distance_m if s.last_distance_m is not None else 0.0
            for s in self.robot.distance_sensors
        ]
        # Decide on actions using the neural net
        # Update kinematics and coordinates
        o = self.decision_maker.decide(sensor_distances)
        o = [oi * self.robot.control_input_upper_limit for oi in o]
        self.robot.move(dt, o)