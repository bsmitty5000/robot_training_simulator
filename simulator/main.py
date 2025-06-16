import sys
import numpy as np
#import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import random
from courses.course1 import GridCoverageCourseA
from courses.grid_coverage_course import GridCoverageCourse
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from models.distance_sensors.sharp_ir import SharpIR
from models.robots.robot import RobotBase
from models.robots.two_wheel_TT import TwoWheelTT
from models.obstacle import Obstacle
from smart_car.smart_car import SmartCar
from ml_stuff.decision_base import DecisionBase

WIDTH = 1280
HEIGHT = 720
FRAME_RATE = 60
COVERAGE_ABORT_S = 5

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
        for s in smart_car.sprite.robot.distance_sensors
    )

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

def run_simulation(screen, clock, course : GridCoverageCourse) -> bool:
    
    running = True
    dt = 0
    previous_coverage = 0.0
    coverage_stale_count = 0

    # Instantiate three SharpIR sensors at different angles
    sensors = [
        SharpIR(-45, 0.3),  # angle in degrees, max_range in meters
        SharpIR(0, 0.3),
        SharpIR(45, 0.3)
    ]
    
    # Create the robot and decision maker
    robot_instance = TwoWheelTT(75, 75, distance_sensors=sensors)
    decision_maker = FFNetDecisionMaker([3, 4, 2])
    #set_weights(self, weights: list[list[list[float]]], biases: list[list[float]])
    weights = [
            np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            for n_in, n_out in zip(decision_maker.layer_sizes, decision_maker.layer_sizes[1:])
        ]
    biases = [
        np.zeros(n_out)
        for n_out in decision_maker.layer_sizes[1:]
    ]
    decision_maker.set_weights(weights, biases)
    smart_car = pygame.sprite.GroupSingle()
    smart_car.add(SmartCar(robot=robot_instance, decision_maker=decision_maker))

    obstacles = pygame.sprite.Group()
    for obs in course.make_course():
        obstacles.add(obs)

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Exit the simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                return True  # Restart the simulation

        smart_car.update(dt, obstacles)

        course.mark_visited(smart_car.sprite.robot.position.x,
                            smart_car.sprite.robot.position.y)
        
        current_coverage = course.coverage_ratio()
        if current_coverage > previous_coverage:
            coverage_stale_count = 0
        else:
            coverage_stale_count += 1

        previous_coverage = current_coverage

        if coverage_stale_count > COVERAGE_ABORT_S * FRAME_RATE:
            # If coverage hasn't improved for a while, restart the simulation
            return True

        if pygame.sprite.spritecollide(
            smart_car.sprite.robot, obstacles, dokill=False, collided=circle_rect_collision):
            return True

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        smart_car.draw(screen)
        obstacles.draw(screen)

        show_debug_info(screen, smart_car)

        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(FRAME_RATE) / 1000

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    while True:
        course = GridCoverageCourseA(WIDTH, HEIGHT)
        restart = run_simulation(screen, clock, course)
        show_coverage_screen(screen, clock, course)
        if not restart:
            break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()