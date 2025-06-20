
import pygame
import math
import json, os, re, sys
from pathlib import Path

import numpy as np

from jit_sim.helpers        import load_spec, load_map
from jit_sim.core_kernels   import circle_rect_collides
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
        pygame.draw.rect(screen, color, pygame.Rect(int(l), int(t), int(w), int(h)))

def draw_robot(screen, x, y, r, hd_deg):
    pygame.draw.circle(screen, (0, 120, 255), world_to_screen(x, y), int(r), 2)
    # heading tick
    hd_rad = np.radians(hd_deg)
    end = (x + r * np.cos(hd_rad),
           y - r * np.sin(hd_rad))     # minus because screen-y down
    pygame.draw.line(screen, (0, 120, 255),
                     world_to_screen(x, y), world_to_screen(*end), 2)

def draw_sensors(screen, x, y, hd_deg,
                 sensor_vals,          # length-3 array (px)
                 max_range,            # scalar px
                 robot_r):
    """Draw left, center, right IR beams."""
    offsets = np.array([-45.0, 0.0, 45.0], dtype=np.float32)

    for idx, dist in enumerate(sensor_vals):
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
    controller_fn, ctrl_kwargs      = load_spec(cfg["controller"])
    sensor_fn,     sens_kwargs      = load_spec(cfg["sensor"])
    move_fn,       robot_kwargs     = load_spec(cfg["robot"])
    _,             opt_kwargs       = load_spec(cfg["optimizer"])

    # map + size
    rects, map_kwargs = load_map(cfg["map"])
    world_w = map_kwargs.get("width_px", 1280.0)
    world_h = map_kwargs.get("height_px", 720.0)
    start_x = map_kwargs.get("starting_x", 75.0)
    start_y = map_kwargs.get("starting_y", 75.0)

    # robot constants
    robot_r = robot_kwargs["wheel_radius_m"] * PIX_PER_M

    # best chromosome lookup ----------------------
    best_chrom = None
    out_dir = Path("saved_chromosomes")
    if(os.path.exists(out_dir / opt_kwargs.get("seed_chrom", "xxxx"))):
        # load seed chromosome from file
        chromosome = np.load(out_dir / opt_kwargs["seed_chrom"]).astype(np.float32)
    else:
        best_fit = -np.inf
        pat = re.compile(r"seed_chromosome(_gen\d+)?_(\d+)fitness\.npy")
        for f in out_dir.glob("seed_chromosome*.npy"):
            m = pat.match(f.name)
            if m and int(m.group(2)) > best_fit:
                best_fit, best_chrom = int(m.group(1)), f
        if best_chrom is None:
            raise FileNotFoundError("No chromosome file found!")
        chromosome = np.load(best_chrom).astype(np.float32)

    # state vars
    x, y            = start_x, start_y
    heading_deg     = 0.0
    velocity        = 0.0
    ang_vel         = 0.0
    pwmL = pwmR     = 0.0

    running = True
    step    = 0
    sensor_max = sens_kwargs.get("max_range", 150.0)
    
    screen = pygame.display.set_mode((world_w, world_h))
    clock  = pygame.time.Clock()

    while running and step < steps_per_episode:
        # ---------------- Pygame events ------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                running = False

        # ---------------- JIT pipeline -------------------------------
        sensors = sensor_fn(x, y, heading_deg, rects, robot_r)
        cmdL_raw, cmdR_raw = controller_fn(chromosome, sensors)
        cmdL = cmdL_raw * 255.0
        cmdR = cmdR_raw * 255.0

        x, y, heading_deg, velocity, ang_vel, pwmL, pwmR = move_fn(
            x, y, heading_deg, velocity, ang_vel, pwmL, pwmR, cmdL, cmdR, dt)

        if circle_rect_collides(x, y, robot_r, rects):
            print("Crash!")
            break

        # ---------------- drawing ------------------------------------
        screen.fill((30, 30, 30))
        draw_course(screen, rects)
        draw_robot(screen, x, y, robot_r, heading_deg)
        draw_sensors(screen, x, y, heading_deg, sensors, sensor_max, robot_r)

        show_debug_info(screen,
                        sensors,
                        np.array([x, y, heading_deg]),
                        np.array([cmdL_raw, cmdR_raw]),
                        np.array([pwmL, pwmR]))

        pygame.display.flip()
        clock.tick(1 / dt)
        step += 1

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
