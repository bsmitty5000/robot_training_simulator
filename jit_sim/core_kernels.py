import math
from numba import njit, prange
import numpy as np

JITTER_PENALTY = 0.1  # penalty for PWM jitter
CLEARANCE_REWARD = 0.2   # per sensor unit
NEW_CELL_REWARD = 1.0      # per fresh cell
STALE_LIMIT     = 40       # steps (~2 s at 20 Hz)

@njit(parallel=True, fastmath=True, cache=True)
# @njit(fastmath=True, cache=True)
def run_generation(population:  np.ndarray,      # (P, N) array of P chromosomes (N floats each) dependent on controller_fn
                   rects:       np.ndarray,      # (N, 4) obstacles
                   controller_fn,                  # callable(chrom, sensors) -> (pwmL,pwmR)
                   sensor_fn,                      # callable(px,py,hd, rects,r) -> 3-array
                   sensor_range: np.float32,         # sensor range (px)
                   move_fn,                        # callable(state, cmdL, cmdR, dt) -> new state
                   steps:       np.int32,
                   dt:          np.float32,
                   robot_r:     np.float32,
                   world_width: np.float32,
                   world_height: np.float32,
                   starting_x: np.float32,
                   starting_y: np.float32) -> np.ndarray:

    pop_size        = population.shape[0]
    fitness         = np.zeros(pop_size, dtype=np.float32)
    grid_cell_size  = robot_r * 2.0
    inverted_gcs    = 1.0 / grid_cell_size
    W_GRID   = int(np.ceil(world_width  * inverted_gcs))
    H_GRID   = int(np.ceil(world_height * inverted_gcs))

    inverted_sensor_range    = 1.0 / sensor_range

    # ── per-robot state vectors (all float32) ────────────────────────
    x        = np.full(pop_size, starting_x,  np.float32)   # start X
    y        = np.full(pop_size, starting_y,  np.float32)   # start Y
    heading  = np.zeros(pop_size,        np.float32)   # deg CW
    velocity = np.zeros(pop_size,        np.float32)   # px/s
    ang_vel  = np.zeros(pop_size,        np.float32)   # deg/s
    pwmL     = np.zeros(pop_size,        np.float32)
    pwmR     = np.zeros(pop_size,        np.float32)
    visited = np.zeros((pop_size, W_GRID, H_GRID), dtype=np.uint8)  # bit-mask
    visit_ct = np.zeros(pop_size, dtype=np.int32)                   # counter
    stale_ctr = np.zeros(pop_size, dtype=np.int32)                  # stagnation


    # ────────────────── main GA batch loop (parallel) ────────────────
    for p in prange(pop_size):

        chrom = population[p]            # view into 26-float chromosome
        # crashed = False
        # spinner = False

        for step in range(steps):

            # 1) sense
            sensors = sensor_fn(x[p], y[p], heading[p], rects, robot_r)

            # clearance reward
            fitness[p] += min(sensors) * inverted_sensor_range * CLEARANCE_REWARD

            # 2) decide (NN or heuristic)
            cmdL, cmdR = controller_fn(chrom, sensors)   # returns (−1…1)
            cmdL *= 255.0
            cmdR *= 255.0

            # smoothness reward
            if step > 0:
                fitness[p] -= (abs(cmdL - pwmL[p]) + abs(cmdR - pwmR[p])) * JITTER_PENALTY

            # 3) move
            (x[p], y[p], heading[p],
             velocity[p], ang_vel[p],
             pwmL[p], pwmR[p]) = move_fn(
                    x[p], y[p], heading[p],
                    velocity[p], ang_vel[p],
                    pwmL[p], pwmR[p],
                    cmdL, cmdR,
                    dt)
            
            # 3.1) update visited grid cells
            # integer cell indices
            gx = int(math.floor(x[p] * inverted_gcs))
            gy = int(math.floor(y[p] * inverted_gcs))

            # within map bounds?
            if 0 <= gx < W_GRID and 0 <= gy < H_GRID:
                if visited[p, gx, gy] == 0:
                    visited[p, gx, gy] = 1
                    visit_ct[p]      += 1
                    stale_ctr[p]      = 0          # reset spinner timer
                    # optional fitness bump
                    fitness[p] += NEW_CELL_REWARD
                else:
                    stale_ctr[p] += 1
                    if stale_ctr[p] >= STALE_LIMIT:
                        # print("\tspinner!!!!\t\t", p, step, x[p], y[p])
                        # spinner = True
                        break                      # terminate early

            
            # if step < 10:
            #     print(p, step, x[p], y[p], heading[p], velocity[p])

            # 4) crash?
            if circle_rect_collides(x[p], y[p], robot_r, rects):
                # print("\tcrash at\t\t", p, step, x[p], y[p])
                # crashed = True
                break                     # episode ends early

            # 5) reward (1 point per alive step)
            fitness[p] += 1.0

        # if not crashed and not spinner:
        #     print("!!!!! (timeout??)", p, step, x[p], y[p])

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