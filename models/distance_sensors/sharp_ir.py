import pygame
from typing import Sequence
from .distance_sensor import DistanceSensor

class SharpIR(DistanceSensor):
    def __init__(self, angle_deg: float, max_range_m: float) -> None:
        super().__init__(angle_deg, max_range_m)

    def measure(
        self,
        sensor_position: pygame.Vector2,
        robot_angle: float,
        obstacles: Sequence[pygame.sprite.Sprite]) -> float:

        end = pygame.Vector2(0, -1).rotate(self.angle_deg - robot_angle)
        end = sensor_position + end * self.max_range_px

        # end = pygame.Vector2()
        # end.from_polar((self.max_range_px, self.angle_deg - robot_angle))
        #direction = pygame.Vector2(0, -1).rotate(robot_angle + self.angle_deg)
        # end = end + sensor_position

        # Raycast: check for intersection with each obstacle rect
        min_dist = self.max_range_px
        for obstacle in obstacles:
            clipped = obstacle.rect.clipline(sensor_position, end)
            if clipped:
                hit_point = pygame.Vector2(clipped[0])
                dist = sensor_position.distance_to(hit_point)
                if dist < min_dist:
                    min_dist = dist
        self.last_distance_m = min_dist
        return min_dist
    
    def detect(self, obstacles):

        for i, dir_vec in enumerate(self.sensor_dirs):

            self.sensor_distance_measurements[i] = self.max_range_px
            for obstacle in obstacles:
                start_pos = self.position + dir_vec.rotate(-self.angle) * self.robot_radius_px
                end_pos = self.position + dir_vec.rotate(-self.angle) * self.max_range_px

                clipped = obstacle.rect.clipline(start_pos, end_pos)

                if clipped:
                    # Calculate the distance from the start position to the intersection point
                    intersection_point = clipped[0]
                    self.sensor_distance_measurements[i] =  \
                        pygame.math.Vector2(start_pos).distance_to(intersection_point)
                    break