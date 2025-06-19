import math
from typing import Sequence
from numba import njit
import numpy as np
from numba import njit, float32
import math

from sim import core
from .distance_sensor import DistanceSensor

class SharpIR(DistanceSensor):
    def __init__(self, angle_deg: float, max_range_m: float) -> None:
        super().__init__(angle_deg, max_range_m)

    def measure(
        self,
        sensor_position: core.Vector2,
        robot_angle: float,
        obstacles: Sequence[core.Shape]) -> float:

        sensor_dir = self.get_sensor_direction(robot_angle)
        end = sensor_position + sensor_dir * self.max_range_px

        # Raycast: check for intersection with each obstacle rect
        # min_dist = self.max_range_px
        # sensor_ray = core.LineSegment(sensor_position, end)
        # for obstacle in obstacles:
        #     # clipped = obstacle.clipline(sensor_ray)
        #     # if clipped:
        #     #     hit_point = clipped[0]
        #     #     dist = sensor_position.distance_to(hit_point)
        #     #     if dist < min_dist:
        #     #         min_dist = dist
        #     dist = obstacle.min_distance_to(sensor_position)
        #     if dist < min_dist:
        #         min_dist = dist
        # self.last_distance_m = min_dist
        rect_arr = np.array([[r.left, r.right, r.top, r.bottom] for r in obstacles],
                dtype=np.float32)
        self.last_distance_m = ray_aabb_min_dist(
            sensor_position.x, sensor_position.y, sensor_dir.x, sensor_dir.y, rect_arr, self.max_range_px)
        return self.last_distance_m
    
@njit(fastmath=True, cache=True)
def ray_aabb_min_dist(px, py, dx, dy, rects, max_range):
    """
    Liang–Barsky: return the distance along the ray (>=0) to the
    nearest rectangle hit, or max_range if no hit within range.

    Parameters
    ----------
    px, py : float   ray origin
    dx, dy : float   ray direction (normalised!)
    rects  : (N,4) float32 array [left, right, top, bottom]
    max_range : float

    Returns
    -------
    dist : float
    """
    best_t = max_range          # current nearest hit (param t along ray)

    for i in range(rects.shape[0]):
        left, right, top, bot = rects[i]

        # Parametric slabs
        if dx != 0.0:
            tx1 = (left  - px) / dx
            tx2 = (right - px) / dx
            tmin = min(tx1, tx2)
            tmax = max(tx1, tx2)
        else:
            # Ray parallel to Y axis; reject if outside slab
            if px < left or px > right:
                continue
            tmin = -math.inf
            tmax =  math.inf

        if dy != 0.0:
            ty1 = (top - py) / dy
            ty2 = (bot - py) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
        else:
            if py < top or py > bot:
                continue
            # keep tmin/tmax from x-slab

        if tmax < 0.0:          # rectangle is behind ray start
            continue
        if tmin > tmax:         # no overlap — miss
            continue

        # First positive entry point
        entry = tmin if tmin >= 0.0 else 0.0
        if entry < best_t:
            best_t = entry
            if best_t == 0.0:   # already inside a rectangle
                return 0.0

    return best_t


    # def draw(self, surface, sensor_position):
    #     # Compute start and end points
    #     # Draw the line up to the collision point
    #     end = self.get_sensor_direction()
    #     end = sensor_position + end * self.last_distance_m

    #     pygame.draw.line(surface, (255, 180, 255), sensor_position, end, 1)
    