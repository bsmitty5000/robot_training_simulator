import math
from numba import njit, float32
import numpy as np
from jit_sim.core_kernels import ray_aabb_min_dist

SENSOR_OFFSETS = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
SENSOR_RANGE = 150.0  # 500px/m * 0.3m range

@njit(fastmath=True, cache=True)
def sense(
    px: np.float32, py: np.float32,     # robot position at t
    angle_deg: np.float32,              # robot heading (CW screen-coords)
    rects,                              # (8,4) float32 # obstacles as [left, right, top, bottom] rectangles
    robot_r: np.float32                 # robot radius (px)
    ):

    readings = np.empty(3, np.float32)
    for s in range(3):
        # 1) global heading for this sensor
        head_deg = angle_deg + SENSOR_OFFSETS[s]
        rad      = math.radians(head_deg)

        # screen coords: +x right, +y down  →  unit vector (cos, -sin)
        dx =  math.cos(rad)
        dy = -math.sin(rad)

        # 2) sensor *position* → edge of robot circle
        sx = px + dx * robot_r
        sy = py + dy * robot_r

        # 3) ray-cast to nearest obstacle
        dist = ray_aabb_min_dist(
                    sx, sy,
                    dx, dy,
                    rects,
                    SENSOR_RANGE)

        readings[s] = dist

    return readings