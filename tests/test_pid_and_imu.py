# tests/test_pid_and_imu.py
import math, numpy as np
import jit_sim.robots.tt_2wheel as model   # adjust import path if needed

# ----------------------------------------------------------------------
# 1. PID tests
# ----------------------------------------------------------------------

def test_pid_zero_error():
    """If cmd == meas the PID output should stay zero."""
    pwm, i = model.pid_yaw_step(
        err_int=0.0,
        omega_cmd=0.0,
        omega_meas=0.0,
        dt=0.02)
    assert pwm == 0.0
    assert i   == 0.0

def test_pid_small_positive_error():
    """For a small positive error, PWM delta should be positive."""
    pwm, i = model.pid_yaw_step(
        err_int=0.0,
        omega_cmd=math.radians(30),      # +30 °/s target
        omega_meas=0.0,
        dt=0.02)
    assert pwm > 0.0
    # Integral term should grow in sign of error
    assert i > 0.0

def test_pid_saturation():
    """Large error should clamp to ±255."""
    pwm, _ = model.pid_yaw_step(
        err_int=0.0,
        omega_cmd=math.radians(8000),     # huge command
        omega_meas=0.0,
        dt=0.02)
    assert pwm == 255.0

# ----------------------------------------------------------------------
# 2. IMU model tests
# ----------------------------------------------------------------------

def test_imu_bias_random_walk():
    """Bias should change very slowly over many steps."""
    gyro_bias, acc_bias = 0.0, 0.0
    g_vals = []
    for _ in range(1000):
        g, a, gyro_bias, acc_bias = model.imu_read(
            ang_vel_z=0.0,
            lin_acc_x=0.0,
            dt=0.02,
            gyro_bias=gyro_bias,
            acc_bias=acc_bias)
        g_vals.append(g)

    # Std-dev of readings should be ~noise level
    assert np.std(g_vals) < 0.1
    # Bias should drift but stay within ±0.5
    assert abs(gyro_bias) < 0.5

def test_imu_clamp():
    """IMU output should saturate to ±1."""
    g, a, *_ = model.imu_read(
        ang_vel_z=1e4,                 # absurd rate
        lin_acc_x=1e4,
        dt=0.02,
        gyro_bias=0.0,
        acc_bias=0.0)
    assert -1.0 <= g <= 1.0
    assert -1.0 <= a <= 1.0
