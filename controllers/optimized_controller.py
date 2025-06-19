import random
import numpy as np
from courses.grid_coverage_course import GridCoverageCourse
import simulator.constants as constants
from typing import List

from .vanilla_ga_controller import VanillaGaController
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from smart_car.smart_car import SmartCar
from models.distance_sensors.sharp_ir import ray_aabb_min_dist
import logging
from datetime import datetime
from numba import njit, float32, prange
import math

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"logs/ga_{timestamp}.log"

# Configure logging (you can adjust level and format as needed)
logging.basicConfig(
    filename=filename,  # Log to a file; use None or remove filename for console
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class OptimizedGAController(VanillaGaController):
    def __init__(self, 
                 smart_car: SmartCar,
                 course: GridCoverageCourse,
                 pop_size=20, 
                 n_generations=10,
                 initial_genotype : List[np.ndarray] = None):
        super().__init__(smart_car, course, pop_size, n_generations, initial_genotype)
    
    def run(self,
            cx_rate, mut_rate,
            max_steps=None) -> List[float]:
        
        obstacles = self.course.make_course()
        obstacles_optimized = np.array([[r.left, r.right, r.top, r.bottom] for r in obstacles],
                dtype=np.float32)
        sensor_dirs = np.array([(v.x, v.y) for v in [s.get_sensor_direction(self.smart_car.robot.angle).normalize() for s in self.smart_car.robot.distance_sensors]],
            dtype=np.float32)

        best_history = []

        #TODO: retrieve these from the smart_car
        MAX_SPEED = 50.0
        MAX_ROT_SPEED = 90.0
        MAX_ACC = 100.0
        MAX_ROT_ACC = 180.0
        SENSOR_MAX_RANGE = 0.3 * constants.PIXELS_PER_METER

        for gen in range(self.n_generations):
            weights = np.array([np.concatenate([w.flatten() for w in ind[0]]) for ind in self.population], dtype=np.float32)
            biases = np.array([np.concatenate([b.flatten() for b in ind[1]]) for ind in self.population], dtype=np.float32)
            fits = run_generation(  weights,
                                    biases,
                                    self.pop_size,
                                    max_steps, constants.SIM_DT,
                                    obstacles_optimized,
                                    sensor_dirs,
                                    SENSOR_MAX_RANGE,
                                    0.1 * constants.PIXELS_PER_METER,
                                    MAX_SPEED, MAX_ROT_SPEED,
                                    MAX_ACC,  MAX_ROT_ACC)
            best = max(fits)
            best_history.append(best)
            logging.info(f"Generation {gen}: Best fitness={best:.4f}")
            
            # Create new generation
            new_pop = []
            while len(new_pop) < len(self.population):
                if constants.NO_RANDOM:
                    top_indices = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:2]
                    p1 = self.population[top_indices[0]]
                    p2 = self.population[top_indices[1]]
                else:
                    p1, p2 = self.tournament_selection(self.population, fits)
                p1 = FFNetDecisionMaker.encode(weights=p1[0], biases=p1[1])
                p2 = FFNetDecisionMaker.encode(weights=p2[0], biases=p2[1])
                if random.random() < cx_rate:
                    #o1, o2 = self.single_point_crossover(p1, p2)
                    o1, o2 = self.layer_crossover(p1, p2)
                else:
                    o1, o2 = p1[:], p2[:]
                new_pop.append(FFNetDecisionMaker.from_genotype(self.mutate(o1, mut_rate), layer_sizes=self.layer_sizes))
                if len(new_pop) < len(self.population):
                    new_pop.append(FFNetDecisionMaker.from_genotype(self.mutate(o2, mut_rate), layer_sizes=self.layer_sizes))
            
            self.population = new_pop

        return best_history

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

