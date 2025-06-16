import sys
import numpy as np
import pygame
import random
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from models.distance_sensors.sharp_ir import SharpIR
from models.robots.robot import RobotBase
from models.robots.two_wheel_TT import TwoWheelTT
from models.obstacle import Obstacle
from smart_car.smart_car import SmartCar
from ml_stuff.decision_base import DecisionBase

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

def run_simulation(screen, clock):
    running = True
    dt = 0

    # Instantiate three SharpIR sensors at different angles
    sensors = [
        SharpIR(-45, 0.3),  # angle in degrees, max_range in meters
        SharpIR(0, 0.3),
        SharpIR(45, 0.3)
    ]
    
    # Create the robot and decision maker
    robot_instance = TwoWheelTT(640, 360, distance_sensors=sensors)
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
    obstacles.add(Obstacle(200, 300, 100, 30))
    obstacles.add(Obstacle(400, 500, 50, 100))
    obstacles.add(Obstacle(600, 250, 100, 25))

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Exit the simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                return True  # Restart the simulation

        # Example: simple autonomous behavior
        # if random.random() < 0.05:  # Occasionally change direction
        #     robot.sprite.left_pwm = random.randint(0, 255)
        #     robot.sprite.right_pwm = random.randint(0, 255)
        # keys = pygame.key.get_pressed()

        # # Adjust left PWM with A/D
        # if keys[pygame.K_a]:
        #     robot.sprite.right_pwm = max(0, robot.sprite.right_pwm - 1)
        #     robot.sprite.left_pwm = max(0, robot.sprite.left_pwm +1)
        # if keys[pygame.K_d]:
        #     robot.sprite.right_pwm = max(0, robot.sprite.right_pwm + 1)
        #     robot.sprite.left_pwm = max(0, robot.sprite.left_pwm - 1)

        # # Adjust both left and right PWM with W/S
        # if keys[pygame.K_s]:
        #     robot.sprite.right_pwm = max(0, robot.sprite.right_pwm - 2)
        #     robot.sprite.left_pwm = max(0, robot.sprite.left_pwm - 2)
        # if keys[pygame.K_w]:
        #     robot.sprite.right_pwm = min(255, robot.sprite.right_pwm + 2)
        #     robot.sprite.left_pwm = min(255, robot.sprite.left_pwm + 2)

        smart_car.update(dt, obstacles)

        if pygame.sprite.spritecollide(
            smart_car.sprite.robot, obstacles, dokill=False, collided=circle_rect_collision):
            return True

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        smart_car.draw(screen)
        obstacles.draw(screen)

        # --- Display sensor measurements ---
        font = pygame.font.Font(None, 28)
        sensor_text = "Sensors: " + ", ".join(
            f"{s.last_distance_m:.2f}" if s.last_distance_m is not None else "N/A"
            for s in smart_car.sprite.robot.distance_sensors
        )
        text_surface = font.render(sensor_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        # --- End sensor display ---

        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    while True:
        restart = run_simulation(screen, clock)
        if not restart:
            break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()