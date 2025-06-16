
from collections.abc import Sequence
import pygame
import math
import models.constants as constants
from models.distance_sensors.distance_sensor import DistanceSensor
from models.robots.robot import RobotBase

class TwoWheelTT(RobotBase):
    def __init__(
        self,
        x: float,
        y: float,
        distance_sensors: list[DistanceSensor] = None,
        robot_diameter_m: float = 0.1):
        
        super().__init__(x, y, distance_sensors)

        longest_sensor_range_m = 0
        for sensor in distance_sensors:
            if sensor.max_range_m > longest_sensor_range_m:
                longest_sensor_range_m = sensor.max_range_m

        robot_footprint_radius_px = int((robot_diameter_m + longest_sensor_range_m) * constants.PIXELS_PER_METER)
        surface_size = (robot_footprint_radius_px * 2, robot_footprint_radius_px * 2)
        self.original_image = pygame.Surface(surface_size, pygame.SRCALPHA)
        self.robot_radius_px = int((robot_diameter_m * constants.PIXELS_PER_METER) / 2)

        pygame.draw.circle(self.original_image, 
                           (255, 0, 0), 
                           (robot_footprint_radius_px, robot_footprint_radius_px), 
                           self.robot_radius_px)
        
        # direction line
        pygame.draw.line(self.original_image, 
                         (0, 255, 0), 
                         (robot_footprint_radius_px, robot_footprint_radius_px), 
                         (robot_footprint_radius_px, robot_footprint_radius_px - self.robot_radius_px),
                         5)

        self.position = pygame.Vector2(x, y)

        # Sensors start on outer edge of robot
        for sensor in self.distance_sensors:
            
            sensor_angle_rad = math.radians(sensor.angle_deg - 90)  
            sensor_range_px = int(sensor.max_range_m * constants.PIXELS_PER_METER)

            start_pos = (robot_footprint_radius_px + self.robot_radius_px * math.cos(sensor_angle_rad), 
                         robot_footprint_radius_px + self.robot_radius_px * math.sin(sensor_angle_rad))
            end_pos = (robot_footprint_radius_px + sensor_range_px * math.cos(sensor_angle_rad), 
                       robot_footprint_radius_px + sensor_range_px * math.sin(sensor_angle_rad))

            pygame.draw.line(self.original_image, (0, 0, 255), start_pos, end_pos, 2)

        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))

        self.velocity = 0.0  # Current forward speed
        self.angular_velocity = 0.0  # Current rotation speed

        self.max_acceleration = 100.0  # M/sec^2
        self.max_angular_acceleration = 180.0  # deg/sec^2

        self.left_pwm = 0.0
        self.right_pwm = 0.0

        self.angle = 0
        self.speed = 0.0
        self.rotation_speed = 0.0
        self.max_speed = 50.0
        self.max_rotation_speed = 90.0

    def update(self, dt: float, obstacles: Sequence[pygame.sprite.Sprite]) -> None:

        self.update_kinematics(dt)

        self.update_coordinates()

        self.detect(obstacles)

    def update_kinematics(self, dt: float) -> None:

        # 1. Calculate target speeds from PWM
        avg_pwm = (self.left_pwm + self.right_pwm) / 2
        target_speed = self.max_speed * (avg_pwm / 255)

        diff_pwm = self.left_pwm - self.right_pwm
        target_rotation_speed = self.max_rotation_speed * (diff_pwm / 255)

        # 2. Accelerate/decelerate toward target speeds
        # Linear velocity
        speed_diff = target_speed - self.velocity
        max_speed_change = self.max_acceleration * dt
        if abs(speed_diff) > max_speed_change:
            speed_diff = max_speed_change if speed_diff > 0 else -max_speed_change
        self.velocity += speed_diff

        # Angular velocity
        rot_diff = target_rotation_speed - self.angular_velocity
        max_rot_change = self.max_angular_acceleration * dt
        if abs(rot_diff) > max_rot_change:
            rot_diff = max_rot_change if rot_diff > 0 else -max_rot_change
        self.angular_velocity += rot_diff

        # 3. Update angle and position
        self.angle += self.angular_velocity * dt
        direction = pygame.Vector2(0, -1).rotate(-self.angle)
        self.position += direction * self.velocity * dt

    def update_coordinates(self) -> None:
        ### Update coordinates for rect.center and each sensor
        ### should be called after self.position and self.angle are updated
        self.rect.center = self.position
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def detect(self, obstacles: Sequence[pygame.sprite.Sprite]) -> None:
        # Use each sensor's measure method
        for sensor in self.distance_sensors:
            
            sensor_direction = pygame.Vector2(0, -1).rotate(sensor.angle_deg - self.angle)
            sensor_position = self.position + sensor_direction * self.robot_radius_px
            #sensor_position = self.position + pygame.Vector2().from_polar((self.robot_radius_px, sensor.angle_deg-self.angle))
            #sensor_position = self.position + pygame.Vector2().from_polar((self.robot_radius_px, self.angle + sensor.angle_deg))
            # vec = pygame.Vector2()
            # vec.from_polar((self.robot_radius_px, sensor.angle_deg - self.angle))
            # sensor_position = self.position + vec
            sensor.measure(sensor_position, self.angle, obstacles)
                

