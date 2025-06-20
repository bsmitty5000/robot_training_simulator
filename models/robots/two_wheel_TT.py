
from collections.abc import Sequence
from sim import core
import simulator.constants as constants
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

        # robot_footprint_radius_px = int((robot_diameter_m + longest_sensor_range_m) * constants.PIXELS_PER_METER)
        # surface_size = (robot_footprint_radius_px * 2, robot_footprint_radius_px * 2)
        # self.original_image = pygame.Surface(surface_size, pygame.SRCALPHA)
        robot_radius_px = int((robot_diameter_m * constants.PIXELS_PER_METER) / 2)
        position = core.Vector2(x, y)
        self.circle = core.Circle(position, robot_radius_px)

        # if constants.DEMO_RUN or not constants.HEADLESS_MODE:
        #     pygame.draw.circle(self.original_image, 
        #                     (255, 0, 0), 
        #                     (robot_footprint_radius_px, robot_footprint_radius_px), 
        #                     self.robot_radius_px)
            
        #     # direction line
        #     pygame.draw.line(self.original_image, 
        #                     (0, 255, 0), 
        #                     (robot_footprint_radius_px, robot_footprint_radius_px), 
        #                     (robot_footprint_radius_px, robot_footprint_radius_px - self.robot_radius_px),
        #                     5)

        self.velocity = 0.0  # Current forward speed
        self.angular_velocity = 0.0  # Current rotation speed

        self.max_acceleration = 100.0  # M/sec^2
        self.max_angular_acceleration_deg = 180.0  # deg/sec^2

        self.left_pwm = 0.0
        self.right_pwm = 0.0

        self.angle_deg = -90.0  # Initial angle in degrees, facing right
        self.speed = 0.0
        self.rotation_speed = 0.0
        self.max_speed = 50.0
        self.max_rotation_speed = 90.0
    
    @property
    def control_input_size(self) -> int:
        return 2
    
    @property
    def control_input_upper_limit(self) -> int:
        return 255
    
    @property
    def control_input_lower_limit(self) -> int:
        return -255
    
    @property
    def  x_coordinate(self):
        return self.circle.center.x
    
    @property
    def  y_coordinate(self):
        return self.circle.center.y

    def move(self, dt: float, control_inputs: Sequence[float]) -> None:

        self.left_pwm = control_inputs[0] if len(control_inputs) > 0 else self.left_pwm
        self.right_pwm = control_inputs[1] if len(control_inputs) > 1 else self.left_pwm

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
        max_rot_change = self.max_angular_acceleration_deg * dt
        if abs(rot_diff) > max_rot_change:
            rot_diff = max_rot_change if rot_diff > 0 else -max_rot_change
        self.angular_velocity += rot_diff

        # 3. Update angle and position
        self.angle_deg = (self.angle_deg + self.angular_velocity * dt) % 360
        direction = core.Vector2(0, -1).rotate(-self.angle_deg)

        self.circle.center += direction * self.velocity * dt

    # def update_coordinates(self):
    #     # 1. Start with a fresh copy of the base image
    #     self.image = self.original_image.copy()
    #     center_offset = core.Vector2(self.image.get_width() // 2, self.image.get_height() // 2)

    #     # 2. Have the sensors draw themselves on an unrotated image
    #     for sensor in self.distance_sensors:
    #         sensor_direction = core.Vector2(0, -1).rotate(sensor.angle_deg)
    #         sensor_position = center_offset + sensor_direction * self.robot_radius_px
    #         if constants.DEMO_RUN or not constants.HEADLESS_MODE:
    #             sensor.draw(self.image, sensor_position)

    #     # 3. Rotate
    #     self.image = pygame.transform.rotate(self.image, self.angle_deg)
    #     self.rect = self.image.get_rect(center=self.position)

    def detect(self, obstacles: Sequence[core.Shape]) -> None:
        # Use each sensor's measure method
        for sensor in self.distance_sensors:
            
            sensor_direction = core.Vector2(0, -1).rotate(sensor.angle_deg - self.angle_deg)
            sensor_position = self.circle.center + sensor_direction * self.circle.radius
            #sensor_position = self.position + pygame.Vector2().from_polar((self.robot_radius_px, sensor.angle_deg-self.angle))
            #sensor_position = self.position + pygame.Vector2().from_polar((self.robot_radius_px, self.angle + sensor.angle_deg))
            # vec = pygame.Vector2()
            # vec.from_polar((self.robot_radius_px, sensor.angle_deg - self.angle))
            # sensor_position = self.position + vec
            sensor.measure(sensor_position, self.angle_deg, obstacles)
                

