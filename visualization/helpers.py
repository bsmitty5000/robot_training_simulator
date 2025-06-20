import sys
import pygame
from courses.grid_coverage_course import GridCoverageCourse
from ml_stuff.decision_base import DecisionBase
from models.obstacle import Obstacle
from models.robots.robot import RobotBase
from smart_car.smart_car import SmartCar

def show_debug_info(screen, 
                    sensor_readings:        float32[:, :],
                    robot_state:            float32[:, :],
                    controller_outputs:     float32[:, :],
                    robot_inputs:           float32[:]):
    
    font = pygame.font.SysFont("consolas", 12)
    sensor_text = "Sensors: " + ", ".join(
        f"{s:5.2f}" if s is not None else "N/A"
        for s in sensor_readings)

    debug_text = (
        f"Controller outputs: {', '.join(f'{o:8.4}' for o in controller_outputs)} | "
        f"Robot Inputs: {', '.join(f'{o:8.4}' for o in robot_inputs)} | "
        f"Robot State: {', '.join(f'{o:8.4}' for o in robot_state)} | "
    )

    text_surface = font.render(sensor_text, True, (255, 255, 255))
    debug_surface = font.render(debug_text, True, (255, 255, 0))
    screen.blit(text_surface, (10, 10))
    screen.blit(debug_surface, (10, 30))
