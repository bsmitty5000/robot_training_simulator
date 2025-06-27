import numpy as np
import matplotlib.pyplot as plt

# ---------- SIMPLE PID PLAYGROUND ------------------------------------
# Plant model:      dX/dt = -(X/τ) + (K * u)
# Target:           X → 1 (unit-step reference)
# You change KP, KI, KD and re-run to see how response changes.
# ---------------------------------------------------------------------

KP = 15.0        # ← play with these three
KI = 20.0
KD = 0.25

# 
tau    = 1.0    # plant time-constant (s)
Kplant = 1.0    # control-to-output gain
dt     = 0.01   # integration step (s)
Tsim   = 10.0   # total simulation time (s)

N = int(Tsim/dt)
t = np.arange(N)*dt

ref = np.ones(N)           # step input
x   = np.zeros(N)          # plant state
u   = np.zeros(N)          # control signal

err_i  = 0.0               # integral accumulator
prev_e = 0.0

for k in range(1, N):
    # -------- PID controller ----------------------
    e     = ref[k] - x[k-1]
    err_i += e*dt
    derr  = (e - prev_e) / dt
    u[k]  = KP*e + KI*err_i + KD*derr
    prev_e = e

    # -------- first-order plant -------------------
    a=-0.943078214921608
    b=1.1178174139163448
    c=-25.21535005918709
    tau=0.1284900161283904

    K = a * pwmL + b * pwmR + c
    dx = (-x[k-1]/tau + Kplant*u[k])
    x[k] = x[k-1] + dx*dt

# ------------------- plots ------------------------
plt.figure(figsize=(6,4))
plt.plot(t, ref, 'k--', label='Reference')
plt.plot(t, x, label='Output x(t)')
plt.title(f"PID step response   KP={KP},  KI={KI},  KD={KD}")
plt.xlabel('Time (s)'); plt.ylabel('x'); plt.grid(True); plt.legend()

plt.figure(figsize=(6,2))
plt.plot(t, u, color='tab:orange')
plt.title("Control effort u(t)")
plt.xlabel('Time (s)'); plt.ylabel('u'); plt.grid(True)
plt.tight_layout()
plt.show()
