"""
main.py – wiring layer
Loads the plug-ins named in sim_config.json, then runs one GA generation.
"""

import json, importlib, time
from pathlib import Path
import numpy as np
import sys
from optimisers import ga as GAOptimizer
from controllers import nn_3x4x2


# ────────────────────────────────────────────────────────────────────
# 1. helpers
# ────────────────────────────────────────────────────────────────────
def load_spec(spec: dict):
    """
    Import `spec["module"]`, return (callable_or_class, kwargs).

    spec examples
    -------------
    {"module": "controllers.nn_3x4x2", "func": "controller"}
    {"module": "optimisers.ga",        "class": "GAOptimizer", "population": 256}
    """
    mod  = importlib.import_module(spec["module"])
    name = spec.get("func") or spec.get("class")
    obj  = getattr(mod, name)
    kwargs = {k: v for k, v in spec.items()
                        if k not in ("module", "func", "class")}
    return obj, kwargs


def load_map(map_module_path: str):
    """
    Dynamically import a map module and return its `obstacles` array.

    Expected map module pattern:
        obstacles = [[left, right, top, bottom], ...]  # floats or ints
    """
    mod = importlib.import_module(map_module_path)
    if not hasattr(mod, "obstacles"):
        raise AttributeError(f"{map_module_path} must define `obstacles`.")
    return np.asarray(mod.obstacles, dtype=np.float32)


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

steps_per_episode = cfg.get("steps_per_episode", 500)
dt                = cfg.get("dt", 0.05)

# plug-ins
controller_fn, ctrl_kwargs = load_spec(cfg["controller"])
sensor_fn,     sens_kwargs = load_spec(cfg["sensor"])
OptClass,      opt_kwargs  = load_spec(cfg["optimizer"])

# map
obstacles = load_map(cfg["map"]["module"])     # shape (N,4)

# ────────────────────────────────────────────────────────────────────
# 3. optimiser & population
# ────────────────────────────────────────────────────────────────────
optimizer   = OptClass(**opt_kwargs)           # e.g. GAOptimizer(population=256,…)
population  = optimizer.initial_population()   # (pop, chrom_len) float32
fitness_buf = np.empty(population.shape[0], dtype=np.float32)

# ────────────────────────────────────────────────────────────────────
# 4. import fast kernels
# ────────────────────────────────────────────────────────────────────
from core_kernels import run_generation        # your JIT kernel

# ────────────────────────────────────────────────────────────────────
# 5. run one generation
# ────────────────────────────────────────────────────────────────────
print("▶  Running one generation …")
t0 = time.perf_counter()

fitness_buf[:] = run_generation(population,
                                obstacles,
                                controller_fn,
                                sensor_fn,
                                steps_per_episode,
                                dt,
                                **ctrl_kwargs,
                                **sens_kwargs)

t1 = time.perf_counter()
print(f"⏱  Sim time: {t1-t0:0.3f} s")

# evolve
population = optimizer.next_generation(population, fitness_buf)

best_idx = fitness_buf.argmax()
print("\n🥇  Best fitness:", fitness_buf[best_idx])
print("🧬  Best genome :", population[best_idx])
