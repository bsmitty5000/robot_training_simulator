import math
from numba import njit, float32
import numpy as np
from jit_sim.core_kernels import ray_aabb_min_dist

SENSOR_OFFSETS = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
SENSOR_RANGE_MIN_M = 0.06
SENSOR_RANGE_MAX_M = 0.6

NOISE_STD_FRACTION = 0.02  # 2% typical noise
OUTLIER_PROB = 0.01  # 1% chance of outlier
OUTLIER_HIGH = True  # True for max spikes, False for zero spikes

class SharpIR3BeamStandardArray:
    NUM_SENSORS = 3
    SENSOR_OFFSETS = SENSOR_OFFSETS
    SENSOR_RANGE_MIN_M = SENSOR_RANGE_MIN_M
    SENSOR_RANGE_MAX_M = SENSOR_RANGE_MAX_M

    @staticmethod
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
            dist_px = ray_aabb_min_dist(
                        sx, sy,
                        dx, dy,
                        rects,
                        SENSOR_RANGE_MAX_M)

            dist_m = dist_px

            # 4) add multiplicative Gaussian noise
            noise = NOISE_STD_FRACTION * dist_m * np.random.randn()
            noisy_reading = dist_m + noise

            # Clamp to valid sensor range
            noisy_reading = max(SENSOR_RANGE_MIN_M, min(SENSOR_RANGE_MAX_M, noisy_reading))

             # ---- ADD OUTLIER SPIKES ----
            if np.random.rand() < OUTLIER_PROB:
                if OUTLIER_HIGH:
                    noisy_reading = SENSOR_RANGE_MAX_M  # simulate missed detection
                else:
                    noisy_reading = SENSOR_RANGE_MIN_M  # simulate false obstacle

            readings[s] = noisy_reading

        return readings