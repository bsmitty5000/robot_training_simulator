import math
from numba import njit
import numpy as np

MAX_SPEED = 50.0            # px/s
MAX_ROT_SPEED = 90.0        # deg/s
MAX_ACC = 100.0             # px/s²
MAX_ROT_ACC = 180.0         # deg/s²

@njit(fastmath=True, cache=True)
def move_step(
    px: np.float32, py: np.float32,         # position at t
    angle_deg: np.float32,                  # heading (CW screen-coords)
    velocity: np.float32,                   # linear speed  (px/s or m/s)
    ang_vel: np.float32,                    # angular speed (deg/s, CW+)
    pwmL: np.float32, pwmR: np.float32,     # last PWM commands  −255 … +255
    cmdL: np.float32, cmdR: np.float32,     # new PWM commands  (network outputs *255)
    dt: np.float32
):
    """Return updated state tuple (px,py,angle_deg,velocity,ang_vel,pwmL,pwmR)."""

    # 0) latch new commands (clip to ±255 just in case)
    pwmL = max(-255.0, min(255.0, cmdL))
    pwmR = max(-255.0, min(255.0, cmdR))

    # 1) map PWM -> target linear / angular speeds
    avg_pwm  = (pwmL + pwmR) * 0.5
    diff_pwm = pwmL - pwmR

    tgt_speed = MAX_SPEED      * (avg_pwm  / 255.0)      # px/s
    tgt_rot   = MAX_ROT_SPEED  * (diff_pwm / 255.0)      # deg/s

    # 2) throttle linear acceleration
    dv  = tgt_speed - velocity
    max_dv = MAX_ACC * dt
    if   dv >  max_dv: dv =  max_dv
    elif dv < -max_dv: dv = -max_dv
    velocity += dv

    #    throttle angular acceleration
    dw  = tgt_rot - ang_vel
    max_dw = MAX_ROT_ACC * dt
    if   dw >  max_dw: dw =  max_dw
    elif dw < -max_dw: dw = -max_dw
    ang_vel += dw

    # 3) update pose
    angle_deg = (angle_deg + ang_vel * dt) % 360.0
    # heading unit vector (screen y grows downward → −sin)
    hd_x =  math.cos(math.radians(angle_deg))
    hd_y = -math.sin(math.radians(angle_deg))

    px += hd_x * velocity * dt
    py += hd_y * velocity * dt

    return px, py, angle_deg, velocity, ang_vel, pwmL, pwmR