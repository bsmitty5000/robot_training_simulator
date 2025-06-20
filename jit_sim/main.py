"""
main.py – wiring layer
Loads the plug-ins named in sim_config.json, then runs one GA generation.
"""

import re
import json, importlib, time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import sys
from .helpers import load_spec, load_map

# ────────────────────────────────────────────────────────────────────
# 2. config
# ────────────────────────────────────────────────────────────────────import sys

here = Path(__file__).parent  # directory containing main.py
# Check for a command line argument for config file
if len(sys.argv) > 1:
    cfg_file = Path(sys.argv[1])
else:
    cfg_file = here / "configs" / "tt_sharpir_feedfwNN_vanillaGA.json"

cfg = json.load(cfg_file.open())

steps_per_episode   = cfg.get("steps_per_episode", 5000)
dt                  = cfg.get("dt", 0.05)
generations         = cfg.get("generations", 5)

# plug-ins
controller_fn,  ctrl_kwargs     = load_spec(cfg["controller"])
sensor_fn,      sens_kwargs     = load_spec(cfg["sensor"])
move_fn,        robot_kwargs    = load_spec(cfg["robot"])
OptClass,       opt_kwargs      = load_spec(cfg["optimizer"])

# map
obstacles,      map_kwargs      = load_map(cfg["map"])     # shape (N,4)
world_width     = map_kwargs.get("height_px", 1280.0)
world_height    = map_kwargs.get("width_px", 720.0)
starting_x      = map_kwargs.get("starting_x", 75.0)
starting_y      = map_kwargs.get("staring_y", 75.0)

# Bookkeeping
global_best_fitness = 3000
best_fitnesses = []
best_chromosomes = []
output_dir = Path("saved_chromosomes")

# Regex to extract fitness score from filename
pattern = re.compile(r"seed_chromosome(_gen\d+)?_(\d+)fitness\.npy")

for file in output_dir.glob("seed_chromosome*.npy"):
    match = pattern.match(file.name)
    if match:
        fitness = int(match.group(2))
        if fitness > global_best_fitness:
            global_best_fitness = fitness

# ────────────────────────────────────────────────────────────────────
# 3. optimiser & population
# ────────────────────────────────────────────────────────────────────
optimizer   = OptClass(**opt_kwargs)           # e.g. GAOptimizer(population=256,…)
population  = optimizer.initial_population()   # (pop, chrom_len) float32
fitness_buf = np.empty(population.shape[0], dtype=np.float32)

# ────────────────────────────────────────────────────────────────────
# 4. import fast kernels
# ────────────────────────────────────────────────────────────────────
from jit_sim.core_kernels import run_generation        # your JIT kernel

# ────────────────────────────────────────────────────────────────────
# 5. spin the wheel!
# ────────────────────────────────────────────────────────────────────

t0 = time.perf_counter()
for g in range(generations):

    fitness_buf[:] = run_generation(population,
                                    obstacles,
                                    controller_fn,
                                    sensor_fn,
                                    sens_kwargs["max_range_m"] * 500.0,  # px/m * 0.3m range
                                    move_fn,
                                    steps_per_episode,
                                    dt,
                                    robot_kwargs["wheel_radius_m"] * 500.0,  # px/m * 0.3m range
                                    world_width,
                                    world_height,
                                    starting_x,
                                    starting_y)

    best_idx = fitness_buf.argmax()
    best_fitnesses.append(fitness_buf[best_idx])
    best_chromosomes.append(population[best_idx].copy())

    # evolve
    population = optimizer.next_generation(population, fitness_buf)

t1 = time.perf_counter()
print(f"{generations} Generations took: {t1-t0:0.3f} s")

best_of_the_best_idx = np.argmax(best_fitnesses)

if( best_fitnesses[best_of_the_best_idx] > global_best_fitness or cfg.get("save_best", False) ):
    output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist

    seed_file_path = output_dir / f"seed_chromosome_{best_fitnesses[best_of_the_best_idx]:.0f}fitness.npy"
    np.save(seed_file_path, best_chromosomes[best_of_the_best_idx].astype(np.float32))

plt.plot(best_fitnesses)
plt.title("Best Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.ioff()
plt.show()
