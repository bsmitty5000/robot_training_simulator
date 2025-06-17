import sys
import pygame
from courses.grid_coverage_course import GridCoverageCourse
from ml_stuff.decision_base import DecisionBase
from models.obstacle import Obstacle
from models.robots.robot import RobotBase
from smart_car.smart_car import SmartCar

class DummyDecision(DecisionBase):
    @property
    def output_size(self) -> int:
        return 2  # For left_pwm, right_pwm

    def decide(self, sensor_distances):
        # Always go forward at medium speed
        return [128.0, 128.0]
    
def circle_rect_collision(robot_sprite : RobotBase, obstacle_sprite : Obstacle) -> bool:
    # Get robot center and radius
    center = pygame.Vector2(robot_sprite.rect.center)
    radius = robot_sprite.robot_radius_px

    # Get the closest point on the obstacle's rect to the robot's center
    closest = pygame.Vector2(
        max(obstacle_sprite.rect.left, min(center.x, obstacle_sprite.rect.right)),
        max(obstacle_sprite.rect.top, min(center.y, obstacle_sprite.rect.bottom))
    )

    # Check if the distance is less than the radius
    return center.distance_to(closest) < radius

def show_coverage_screen(screen, clock, course: GridCoverageCourse):
    font = pygame.font.SysFont("consolas", 36)
    coverage = course.coverage_ratio() * 100
    text = f"Course Coverage: {coverage:.2f}%"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.fill("black")
    screen.blit(text_surface, (screen.get_width() // 2 - text_surface.get_width() // 2,
                               screen.get_height() // 2 - text_surface.get_height() // 2))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting = False
        clock.tick(30)

def show_debug_info(screen, smart_car: SmartCar):
    font = pygame.font.SysFont("consolas", 12)
    sensor_text = "Sensors: " + ", ".join(
        f"{s.last_distance_m:5.2f}" if s.last_distance_m is not None else "N/A"
        for s in smart_car.sprite.robot.distance_sensors)

    # Get neural net outputs and scaled control inputs
    # (Repeat the logic from SmartCar.update for display)
    sensor_distances = [
        s.last_distance_m if s.last_distance_m is not None else 0.0
        for s in smart_car.sprite.robot.distance_sensors
    ]
    nn_outputs = smart_car.sprite.decision_maker.decide(sensor_distances)
    scaled_outputs = [
        oi * smart_car.sprite.robot.control_input_upper_limit for oi in nn_outputs
    ]
    left_pwm = smart_car.sprite.robot.left_pwm
    right_pwm = smart_car.sprite.robot.right_pwm
    angle = smart_car.sprite.robot.angle_deg

    debug_text = (
        f"NN outputs: {', '.join(f'{o:8.4}' for o in nn_outputs)} | "
        f"Scaled: {', '.join(f'{o:8.4}' for o in scaled_outputs)} | "
        f"PWM L:{left_pwm:8.4} R:{right_pwm:8.4} | "
        f"Angle: {angle:8.4}"
    )

    text_surface = font.render(sensor_text, True, (255, 255, 255))
    debug_surface = font.render(debug_text, True, (255, 255, 0))
    screen.blit(text_surface, (10, 10))
    screen.blit(debug_surface, (10, 30))
