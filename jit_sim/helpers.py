# ────────────────────────────────────────────────────────────────────
# 1. helpers
# ────────────────────────────────────────────────────────────────────
import importlib

import numpy as np


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


def load_map(spec: dict):
    """
    Dynamically import a map module and return its `obstacles` array.

    Expected map module pattern:
        obstacles = [[left, right, top, bottom], ...]  # floats or ints
    """
    mod = importlib.import_module(spec["module"])
    if not hasattr(mod, "obstacles"):
        raise AttributeError(f"{spec["module"]} must define `obstacles`.")
    kwargs = {k: v for k, v in spec.items()
                        if k not in ("module", "func", "class")}
    return np.asarray(mod.obstacles, dtype=np.float32), kwargs

