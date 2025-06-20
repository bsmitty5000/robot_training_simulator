import math
from numba import njit, prange, float32, int32
import numpy as np

# @njit(parallel=True, fastmath=True, cache=True)
@njit(fastmath=True, cache=True)
def run_generation( weights, biases, pop_size,
                    steps, dt,
                    rects,  # (8,4) float32
                    sensor_dirs,  # (3,2) unit vecs
                    sensor_range,
                    robot_radius,
                    MAX_SPEED, MAX_ROT_SPEED,
                    MAX_ACC,  MAX_ROT_ACC):

    fitness = np.zeros(pop_size, dtype=np.float32)

    # per-robot state vectors
    x  = np.full(pop_size, 75, dtype=np.float32)
    y  = np.full(pop_size, 75, dtype=np.float32)
    vel = np.zeros(pop_size, dtype=np.float32)
    ang_v = np.zeros(pop_size, dtype=np.float32)
    pwmL = np.zeros(pop_size, dtype=np.float32)
    pwmR = np.zeros(pop_size, dtype=np.float32)
    hd = np.zeros(pop_size, dtype=np.float32)   # heading (rad)

    for step in range(steps):
        # -------- sensor pass (vectorised over pop & 3 sensors) ----------
        for p in prange(pop_size):              # outer loop parallelised
            readings = np.empty(3, np.float32)
            for s in range(3):
                dx =  math.cos(hd[p]) * sensor_dirs[s,0] \
                    - math.sin(hd[p]) * sensor_dirs[s,1]
                dy =  math.sin(hd[p]) * sensor_dirs[s,0] \
                    + math.cos(hd[p]) * sensor_dirs[s,1]
                readings[s] = ray_aabb_min_dist(
                                x[p], y[p], dx, dy,
                                rects, sensor_range)

            # -------- tiny neural net (3×4×2) fully in njit ------------
            # inlining removes Python call overhead entirely
            h0 = math.tanh(readings[0]*weights[p,0] +
                        readings[1]*weights[p,1] +
                        readings[2]*weights[p,2] + biases[p,0])
            h1 = math.tanh(readings[0]*weights[p,3] +
                        readings[1]*weights[p,4] +
                        readings[2]*weights[p,5] + biases[p,1])
            h2 = math.tanh(readings[0]*weights[p,6] +
                        readings[1]*weights[p,7] +
                        readings[2]*weights[p,8] + biases[p,2])
            h3 = math.tanh(readings[0]*weights[p,9] +
                        readings[1]*weights[p,10] +
                        readings[2]*weights[p,11] + biases[p,3])
            
            vl = math.tanh(h0*weights[p,12] +
                        h1*weights[p,13] +
                        h2*weights[p,14] +
                        h3*weights[p,15] + biases[p,4])
            vr = math.tanh(h0*weights[p,16] +
                        h1*weights[p,17] +
                        h2*weights[p,18] +
                        h3*weights[p,19] + biases[p,5])
            
            vl *= 255.0
            vr *= 255.0

            # -------- kinematics (differential drive) -------------------
            (x[p], y[p], hd[p], vel[p], ang_v[p], pwmL[p], pwmR[p]) = move_step(
                    x[p],   y[p],
                    hd[p],
                    vel[p], ang_v[p],
                    pwmL[p], pwmR[p],
                    vl, vr,
                    dt,
                    MAX_SPEED, MAX_ROT_SPEED,
                    MAX_ACC,  MAX_ROT_ACC)
            
            if step < 10:
                print(p, step, x[p], y[p], hd[p], vel[p])
            
            if circle_rect_collides(x[p], y[p], robot_radius, rects):
                # crash: cut episode short
                #fitness[p] += crash_penalty                 # or simply leave fitness as-is
                #alive[p] = False                            # optional flag
                continue                                    # skip to next individual


            # -------- fitness bookkeeping ------------------------------
            fitness[p] += 1                         # e.g. alive-time
            # you can also deduct penalty if readings.min()<crash_thresh

    return fitness * dt

@njit(parallel=True, fastmath=True, cache=True)
def run_generation(population: float32[:, :],      # (P, N) array of P chromosomes (N floats each) dependent on controller_fn
                   rects:       float32[:, :],      # (N, 4) obstacles
                   controller_fn,                  # callable(chrom, sensors) -> (pwmL,pwmR)
                   sensor_fn,                      # callable(px,py,hd, rects,r) -> 3-array
                   move_fn,                        # callable(state, cmdL, cmdR, dt) -> new state
                   steps:       int32,
                   dt:          float32,
                   robot_r:     float32):

    pop_size  = population.shape[0]
    fitness   = np.zeros(pop_size, dtype=np.float32)

    # ── per-robot state vectors (all float32) ────────────────────────
    x        = np.full(pop_size, 120.0,  np.float32)   # start X
    y        = np.full(pop_size, 120.0,  np.float32)   # start Y
    heading  = np.zeros(pop_size,        np.float32)   # deg CW
    velocity = np.zeros(pop_size,        np.float32)   # px/s
    ang_vel  = np.zeros(pop_size,        np.float32)   # deg/s
    pwmL     = np.zeros(pop_size,        np.float32)
    pwmR     = np.zeros(pop_size,        np.float32)

    # ────────────────── main GA batch loop (parallel) ────────────────
    for p in prange(pop_size):

        chrom = population[p]            # view into 26-float chromosome

        for step in range(steps):

            # 1) sense
            sensors = sensor_fn(x[p], y[p], heading[p], rects, robot_r)

            # 2) decide (NN or heuristic)
            cmdL, cmdR = controller_fn(chrom, sensors)   # returns (−1…1)
            cmdL *= 255.0
            cmdR *= 255.0

            # 3) move
            (x[p], y[p], heading[p],
             velocity[p], ang_vel[p],
             pwmL[p], pwmR[p]) = move_fn(
                    x[p], y[p], heading[p],
                    velocity[p], ang_vel[p],
                    pwmL[p], pwmR[p],
                    cmdL, cmdR,
                    dt)

            # 4) crash?
            if circle_rect_collides(x[p], y[p], robot_r, rects):
                break                     # episode ends early

            # 5) reward (1 point per alive step)
            fitness[p] += 1.0

    return fitness


@njit(fastmath=True, cache=True)
def circle_rect_collides(px: float, py: float,
                         radius: float,
                         rects: np.ndarray) -> bool:
    """
    Returns True if the circle (px,py,radius) intersects *any*
    axis-aligned rectangle in `rects` (shape (N,4): left,right,top,bottom).
    """
    r2 = radius * radius

    for i in range(rects.shape[0]):
        left, right, top, bot = rects[i]

        # --- clamp the circle centre to the rectangle ---
        cx = px
        if cx < left:   cx = left
        elif cx > right: cx = right

        cy = py
        if cy < top:    cy = top
        elif cy > bot:  cy = bot

        # squared distance circle-centre → closest point
        dx = px - cx
        dy = py - cy
        if dx*dx + dy*dy <= r2:
            return True        # early exit on first hit

    return False

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