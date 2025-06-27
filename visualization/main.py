
import time
import pygame
import math
import json, os, re, sys
from pathlib import Path

import numpy as np

from jit_sim.helpers        import load_spec, load_map
import jit_sim.core_kernels as core_kernels
from .helpers               import show_debug_info

PIX_PER_M = 500.0   # pixels per meter  (world→screen scale)

# ───────────────────────────────────────── helpers ────────────────────
def world_to_screen(wx, wy):
    """Convert world coords (y-down) to screen ints (pygame expects ints)."""
    return int(wx), int(wy)            # 1-to-1 mapping; flip if needed

def draw_course(screen, rects, color=(50, 50, 50)):
    """rects : (N,4) [l,r,t,b] world px."""
    for l, r, t, b in rects:
        w, h = r - l, b - t
        pygame.draw.rect(screen, color, \
                         pygame.Rect(round(l*PIX_PER_M), round(t*PIX_PER_M), round(w*PIX_PER_M), round(h*PIX_PER_M)))

def draw_robot(screen, x_m, y_m, r_m, hd_deg):
    x = round(x_m * PIX_PER_M)
    y = round(y_m * PIX_PER_M)
    r = round(r_m * PIX_PER_M)
    pygame.draw.circle(screen, (0, 120, 255), world_to_screen(x, y), int(r), 2)
    # heading tick
    hd_rad = np.radians(hd_deg)
    end = (x + r * np.cos(hd_rad),
           y - r * np.sin(hd_rad))     # minus because screen-y down
    pygame.draw.line(screen, (0, 120, 255),
                     world_to_screen(x, y), world_to_screen(*end), 2)

def draw_sensors(screen, x_m, y_m, hd_deg,
                 sensor_vals,          # length-3 array (px)
                 max_range_m,            # scalar px
                 robot_r_m):
    """Draw left, center, right IR beams."""
    x = round(x_m * PIX_PER_M)
    y = round(y_m * PIX_PER_M)
    robot_r = robot_r_m * PIX_PER_M
    max_range = max_range_m * PIX_PER_M

    # todo get this from the sensor class
    offsets = np.array([-45.0, 0.0, 45.0], dtype=np.float32)

    for idx, dist_m in enumerate(sensor_vals):
        dist = round(dist_m * PIX_PER_M)
        head = np.radians(hd_deg + offsets[idx])
        # sensor origin = circle edge
        sx = x + math.cos(head) * robot_r
        sy = y - math.sin(head) * robot_r

        # endpoint — use max_range if no hit (for ghost beam)
        eff_dist = dist
        color    = (255, 80, 80) if dist < max_range else (140, 140, 140)

        ex = sx + math.cos(head) * eff_dist
        ey = sy - math.sin(head) * eff_dist

        pygame.draw.line(screen, color,
                         world_to_screen(sx, sy),
                         world_to_screen(ex, ey), 2)


# ───────────────────────────────────────── main loop ───────────────────
def run_simulation():
    pygame.init()

    # ── config file path ──────────────────────────────────────────────
    cfg_file = Path(sys.argv[1]) if len(sys.argv) > 1 else \
               Path("jit_sim/configs/tt_sharpir_feedfwNN_vanillaGA.json")
    cfg = json.load(cfg_file.open())

    # sim params
    steps_per_episode = cfg.get("steps_per_episode", 5000)
    dt                = cfg.get("dt", 0.05)

    # plug-ins
    ControllerClass,    ctrl_kwargs     = load_spec(cfg["controller"])
    SensorClass,    sens_kwargs      = load_spec(cfg["sensor"])
    move_fn,       robot_kwargs     = load_spec(cfg["robot"])
    _,             opt_kwargs       = load_spec(cfg["optimizer"])

    # map + size
    rects, map_kwargs = load_map(cfg["map"])
    world_width     = map_kwargs.get("width_m")
    world_height    = map_kwargs.get("height_m")
    starting_x      = map_kwargs.get("starting_x_m")
    starting_y      = map_kwargs.get("starting_y_m")

    # robot constants
    robot_r_m = robot_kwargs["wheel_radius_m"]
    sensor  = SensorClass()
    controller  = ControllerClass(**ctrl_kwargs)
    controller_fn = controller.fwd if hasattr(controller, "fwd") else controller

    # best chromosome lookup ----------------------
    best_chrom = None
    out_dir = Path("saved_chromosomes")
    if(os.path.exists(out_dir / opt_kwargs.get("seed_chrom", "xxxx"))):
        # load seed chromosome from file
        print(f"Loading seed chromosome from {out_dir / opt_kwargs["seed_chrom"]}")
        chromosome = np.load(out_dir / opt_kwargs["seed_chrom"]).astype(np.float32)
    else:
        files = list(out_dir.glob("seed_chromosome*.npy"))
        if not files:
            raise FileNotFoundError("No chromosome file found!")
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading latest chromosome from {latest_file}")
        chromosome = np.load(latest_file).astype(np.float32)

    controller.print_chromosome(chromosome)

    # state vars
    x, y            = starting_x, starting_y
    heading_deg     = 0.0
    velocity        = 0.0
    ang_vel         = 0.0
    pwmL = pwmR     = 0.0

    fitness        = 0.0
    grid_cell_size  = robot_r_m * 2.0
    inverted_gcs    = 1.0 / grid_cell_size
    W_GRID   = int(np.ceil(world_width  * inverted_gcs))
    H_GRID   = int(np.ceil(world_height * inverted_gcs))
    visited = np.zeros((W_GRID, H_GRID), dtype=np.uint8)
    visit_ct = 0

    sensor_reward_multiplier =  \
        core_kernels.CLEARANCE_REWARD / (sensor.NUM_SENSORS * sensor.SENSOR_RANGE_MAX_M)

    running = True
    step    = 0
    
    screen = pygame.display.set_mode((world_width * PIX_PER_M, world_height * PIX_PER_M))
    clock  = pygame.time.Clock()

    t0 = time.perf_counter()
    while running and step < steps_per_episode:
        # ---------------- Pygame events ------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                running = False

        # ---------------- JIT pipeline -------------------------------
        sensors = sensor.sense(x, y, heading_deg, rects, robot_r_m)

        clearance_reward = sensors.sum() * sensor_reward_multiplier
        fitness += clearance_reward

        
        open_space_bonus = np.sum(sensors >= \
                                  (sensor.SENSOR_RANGE_MAX_M * core_kernels.OPEN_SPACE_REWARD_CUTOFF))\
                                      * core_kernels.OPEN_SPACE_REWARD
        fitness += open_space_bonus

        cntrl_out = controller_fn(chromosome, controller.chrom_fmt(), sensors)

        if step > 0:
            jitter_penalty = abs(cntrl_out[0] - ang_vel) * core_kernels.JITTER_PENALTY
            fitness -= jitter_penalty

        (x, y, heading_deg,
            velocity, ang_vel,
            pwmL, pwmR) = move_fn(
                    x, y,
                    heading_deg,
                    velocity,
                    ang_vel,
                    pwmL, pwmR,
                    cntrl_out,
                    dt)
        
        # state_log.write(f"0,{step},{sensors[0]},{sensors[1]},{sensors[2]},{x},{y},{heading_deg},{velocity},{ang_vel},{pwmL},{pwmR},{fitness}\n")

        gx = int(math.floor(x * inverted_gcs))
        gy = int(math.floor(y * inverted_gcs))

        # within map bounds?
        if 0 <= gx < W_GRID and 0 <= gy < H_GRID:
            if visited[gx, gy] == 0:
                visited[gx, gy] = 1
                visit_ct      += 1
                stale_ctr      = 0          # reset spinner timer
                # optional fitness bump
                fitness += core_kernels.NEW_CELL_REWARD
            else:
                stale_ctr += 1
                if stale_ctr >= core_kernels.STALE_LIMIT:
                    fitness -= core_kernels.TIMEOUT_PENALTY
                    print("Spinner detected! Terminating early.")
                    break
        else:
            print(f"Out of bounds: gx={gx}, gy={gy} (x={x}, y={y})")
            break

        if core_kernels.circle_rect_collides(x, y, robot_r_m, rects):
            print("Crash!")
            break

        fitness += 1.0 * core_kernels.KEEP_ALIVE_REWARD

        # ---------------- drawing ------------------------------------
        screen.fill((30, 30, 30))
        draw_course(screen, rects)
        draw_robot(screen, x, y, robot_r_m, heading_deg)
        draw_sensors(screen, x, y, heading_deg, sensors, sensor.SENSOR_RANGE_MAX_M, robot_r_m)

        show_debug_info(screen,
                        sensors,
                        np.array([x, y, heading_deg]),
                        cntrl_out,
                        np.array([pwmL, pwmR]),
                        fitness)

        pygame.display.flip()
        clock.tick(1 / dt)
        step += 1

    t1 = time.perf_counter()
    print(f"Simulation ended after {step} steps and {t1-t0:0.3f} s with fitness: {fitness:.2f}")
    pygame.quit()

if __name__ == "__main__":
    run_simulation()
