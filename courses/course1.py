# Example: courses/boundary_course.py
from courses.grid_coverage_course import GridCoverageCourse
from simulator import constants
from sim import core

class GridCoverageCourseA(GridCoverageCourse):
    def __init__(self, width: int, height: int, cell_size: int = 10):
        super().__init__(width, height, cell_size)

    def make_course(
                    self,
                    thickness: int = 10,
                    min_distance_between_obstacles_m: int = 0.3) -> list[core.Rect]:
        
        min_distance_between_obstacles_px = int(min_distance_between_obstacles_m * constants.PIXELS_PER_METER)

        course = [
            # Top boundary
            core.Rect(0, 0, self.width, thickness),
            # Bottom boundary
            core.Rect(0, self.height - thickness, self.width, thickness),
            # Left boundary
            core.Rect(0, 0, thickness, self.height),
            # Right boundary
            core.Rect(self.width - thickness, 0, thickness, self.height),

        ]

        next_y_start = min_distance_between_obstacles_px
        obstacle_num = 0

        while next_y_start < self.height:
        
            course.append(
                core.Rect(
                    obstacle_num % 2 * min_distance_between_obstacles_px, 
                    next_y_start, 
                    self.width-min_distance_between_obstacles_px, 
                    thickness))
            next_y_start += thickness + min_distance_between_obstacles_px
            obstacle_num += 1

        return course
