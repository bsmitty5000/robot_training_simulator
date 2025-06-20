from typing import Sequence

from visualization import core
from .distance_sensor import DistanceSensor

class SharpIR(DistanceSensor):
    def __init__(self, angle_deg: float, max_range_m: float) -> None:
        super().__init__(angle_deg, max_range_m)

    def measure(
        self,
        sensor_position: core.Vector2,
        robot_angle: float,
        obstacles: Sequence[core.Shape]) -> float:

        end = self.get_sensor_direction(robot_angle)
        end = sensor_position + end * self.max_range_px

        # Raycast: check for intersection with each obstacle rect
        min_dist = self.max_range_px
        sensor_ray = core.LineSegment(sensor_position, end)
        for obstacle in obstacles:
            clipped = obstacle.clipline(sensor_ray)
            if clipped:
                hit_point = clipped[0]
                dist = sensor_position.distance_to(hit_point)
                if dist < min_dist:
                    min_dist = dist
        self.last_distance_m = min_dist
        return min_dist
    
    # def draw(self, surface, sensor_position):
    #     # Compute start and end points
    #     # Draw the line up to the collision point
    #     end = self.get_sensor_direction()
    #     end = sensor_position + end * self.last_distance_m

    #     pygame.draw.line(surface, (255, 180, 255), sensor_position, end, 1)
    