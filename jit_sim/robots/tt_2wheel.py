import math
from numba import njit
import numpy as np

PWM_TO_VELOCITY = 0.0017                  # m/s/pwm
PWM_TO_VELOCITY_PX = PWM_TO_VELOCITY * 500.0     # px/s
MAX_ROT_SPEED = 480.0               # deg/s
MAX_PWM_CHANGE = 10
DEG_TO_RAD = math.pi / 180.0

MAX_PWM = 255.0              # max PWM command

# ---- PID constants (rad/s error to PWM) --------------
KP = 2.0      # proportional
KI = 0.8      # integral
KD = 0.05     # derivative

# ---- MPU-6050 tunables -------------------------------------------------------
GYRO_RANGE_DPS      = 1000.0          # ±1000 °/s
ACC_RANGE_G         = 4.0            # ±4 g
GYRO_NOISE_STD      = 0.033            # °/s   (white noise)
ACC_NOISE_STD       = 3.4 / 1000.0           # g
GYRO_BIAS_RW_STD    = 0.01           # °/s√s  (random walk)
ACC_BIAS_RW_STD     = 0.002          # g√s

@njit(fastmath=True, cache=True)
def move_step(
    px: np.float32, py: np.float32,                 # position at t
    angle_deg: np.float32,                          # heading (CW screen-coords)
    velocity: np.float32,                           # linear speed  (px/s or m/s)
    ang_vel: np.float32,                            # angular speed (deg/s, CW+)
    pwmL: np.float32, pwmR: np.float32,             # last PWM commands  −255 … +255
    cntrl_out: np.array,                            # controller output (e.g. NN)
    dt: np.float32
):
    """Return updated state tuple (px,py,angle_deg,velocity,ang_vel, gyro_bias, acc_bias)."""

    # expecting controller output to be a 2-element array:
    yaw_cmd = cntrl_out[0]                            # [-1.0, 1.0] range
    fwd_mag = cntrl_out[1]                            # [0, 1.0] range, forward motion magnitude

    throttle = MAX_PWM * fwd_mag
    pwm_delta = MAX_PWM * yaw_cmd * 0.5

    pwmL_cmd = throttle - pwm_delta
    pwmR_cmd = throttle + pwm_delta

    # 2) slew-limit
    pwmL = max(-255.0, min(255.0,
            pwmL + max(-MAX_PWM_CHANGE,
                        min(MAX_PWM_CHANGE, pwmL_cmd - pwmL))))
    pwmR = max(-255.0, min(255.0,
            pwmR + max(-MAX_PWM_CHANGE,
                        min(MAX_PWM_CHANGE, pwmR_cmd - pwmR))))

    avg_pwm = (pwmL + pwmR) / 2.0

    # 3) some experimental plant model for yaw rate
    a=-0.943078214921608
    b=1.1178174139163448
    c=-25.21535005918709
    tau=0.1284900161283904

    K = a * pwmL + b * pwmR + c
    alpha = 1 - np.exp(-dt / tau)
    ang_vel = ang_vel + (K - ang_vel) * alpha

    # hand-wavy experimental model for linear velocity
    velocity = PWM_TO_VELOCITY_PX * avg_pwm

    # 4) update pose
    angle_deg = (angle_deg + ang_vel * dt) % 360.0
    hd_x = math.cos(math.radians(angle_deg))
    hd_y = -math.sin(math.radians(angle_deg))

    px += hd_x * velocity * dt
    py += hd_y * velocity * dt

    # 5) linear accel for next step (finite diff in run loop)
    return (px, py, angle_deg,
            velocity, ang_vel,
            pwmL, pwmR)

@njit(fastmath=True, cache=True)
def pid_yaw_step(err_int, omega_cmd, omega_meas, dt):
    err = omega_cmd - omega_meas
    derr = -omega_meas             # cmd is quasi-constant per step
    err_int += err * dt
    pwm_out = KP*err + KI*err_int + KD*derr
    pwm_out = max(-255.0, min(255.0, pwm_out))
    return pwm_out, err_int


# modeled mpu-6050 IMU sensor
@njit(fastmath=True, cache=True)
def imu_read(ang_vel_z, lin_acc_x, dt, gyro_bias, acc_bias):
    # random-walk bias
    gyro_bias += GYRO_BIAS_RW_STD * math.sqrt(dt) * np.random.randn()
    acc_bias  += ACC_BIAS_RW_STD  * math.sqrt(dt) * np.random.randn()

    ax_g = lin_acc_x / 9.81  # to g

    gyro_meas = ang_vel_z / GYRO_RANGE_DPS + gyro_bias + \
                GYRO_NOISE_STD * np.random.randn()
    acc_meas  = ax_g         / ACC_RANGE_G  + acc_bias  + \
                ACC_NOISE_STD * np.random.randn()

    gyro_meas = max(-1.0, min(1.0, gyro_meas))
    acc_meas  = max(-1.0, min(1.0, acc_meas))
    return gyro_meas, acc_meas, gyro_bias, acc_bias