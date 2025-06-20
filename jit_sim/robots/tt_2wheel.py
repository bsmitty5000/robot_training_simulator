import math
from numba import njit, float32

@njit(fastmath=True, cache=True)
def move_step(
    px: float32, py: float32,         # position at t
    angle_deg: float32,               # heading (CW screen-coords)
    velocity: float32,                # linear speed  (px/s or m/s)
    ang_vel: float32,                 # angular speed (deg/s, CW+)
    pwmL: float32, pwmR: float32,     # last PWM commands  −255 … +255
    cmdL: float32, cmdR: float32,     # new PWM commands  (network outputs *255)
    dt: float32,
    max_speed: float32,               # px/s
    max_rot_speed: float32,           # deg/s
    max_acc: float32,                 # px/s²
    max_rot_acc: float32              # deg/s²
):
    """Return updated state tuple (px,py,angle_deg,velocity,ang_vel,pwmL,pwmR)."""

    # 0) latch new commands (clip to ±255 just in case)
    pwmL = max(-255.0, min(255.0, cmdL))
    pwmR = max(-255.0, min(255.0, cmdR))

    # 1) map PWM -> target linear / angular speeds
    avg_pwm  = (pwmL + pwmR) * 0.5
    diff_pwm = pwmL - pwmR

    tgt_speed = max_speed      * (avg_pwm  / 255.0)      # px/s
    tgt_rot   = max_rot_speed  * (diff_pwm / 255.0)      # deg/s

    # 2) throttle linear acceleration
    dv  = tgt_speed - velocity
    max_dv = max_acc * dt
    if   dv >  max_dv: dv =  max_dv
    elif dv < -max_dv: dv = -max_dv
    velocity += dv

    #    throttle angular acceleration
    dw  = tgt_rot - ang_vel
    max_dw = max_rot_acc * dt
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