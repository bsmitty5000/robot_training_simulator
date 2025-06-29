import math
from numba import njit
import numpy as np

MAX_ROT_SPEED = 480.0               # deg/s
DEG_TO_RAD = math.pi / 180.0
WHEEL_BASE_M = 0.10
WHEEL_DIAMETER_M = 0.060
MAX_NO_LOAD_ROTATION = 3.33 # rev/s
PERCENT_NO_LOAD_ROTATION = 0.60  # 75% of max no-load speed
MODELED_MAX_VELOCITY = PERCENT_NO_LOAD_ROTATION * MAX_NO_LOAD_ROTATION * WHEEL_DIAMETER_M * math.pi  # m/s


@njit(fastmath=True, cache=True)
def move_step(
    px: np.float32, py: np.float32,                 # position at t
    angle_deg: np.float32,                          # heading (CW screen-coords)
    cntrl_out: np.array,                            # controller output (e.g. NN)
    dt: np.float32
):
    """
    Differential-drive kinematics update (exact arc integration).

    Parameters
    ----------
    x : float
        Current x position (m).
    y : float
        Current y position (m).
    theta : float
        Current heading (rad).
    vL : float
        Left wheel linear velocity (m/s).
    vR : float
        Right wheel linear velocity (m/s).
    L : float
        Track width (center-to-center distance between wheels, m).
    dt : float
        Time step duration (s).

    Returns
    -------
    x_new, y_new, theta_new : tuple of floats
    """

    vL = cntrl_out[0] * MODELED_MAX_VELOCITY  # left wheel speed (rad/s)
    vR = cntrl_out[1] * MODELED_MAX_VELOCITY  # right wheel speed (rad/s)
    angle_rad = math.radians(angle_deg)

    # Body-frame forward and angular velocities
    v = 0.5 * (vR + vL)
    omega = (vL - vR) / WHEEL_BASE_M

    # Straight or near-straight motion
    if abs(omega) < 1e-8:
        x_new = px + v * math.cos(angle_rad) * dt
        y_new = py - v * math.sin(angle_rad) * dt
        theta_new = angle_rad
    else:
        theta_new = angle_rad + omega * dt
        x_new = px + (v / omega) * (math.sin(theta_new) - math.sin(angle_rad))
        y_new = py + (v / omega) * (math.cos(theta_new) - math.cos(angle_rad))

    # Optionally normalize theta_new to [-pi, pi]
    # theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi

    angle_deg_new = math.degrees(theta_new)

    return x_new, y_new, angle_deg_new, (angle_deg_new - angle_deg) / dt
