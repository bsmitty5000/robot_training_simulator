# optimisers/ga.py
"""
A minimal (but solid) genetic algorithm:
    • real-valued chromosomes  (NumPy float32)
    • tournament selection     (k-way, keeps diversity)
    • elitism                  (top N copied unchanged)
    • uniform crossover        (50 % gene-wise mix)
    • Gaussian mutation        (σ decays each generation)

Usage (see main.py):
    OptClass, kwargs = load_spec(cfg["optimizer"])
    opt = OptClass(**kwargs)
    pop = opt.initial_population()
    pop = opt.next_generation(pop, fitness)
"""
from __future__ import annotations
import numpy as np

class GAOptimizer:
    # ───── constructor ────────────────────────────────────────────────
    def __init__(self,
                 population: int        = 256,
                 chrom_len:  int        = 26,
                 elite_fraction: float  = 0.05,
                 tournament_k:  int     = 3,
                 mutation_sigma: float  = 0.15,
                 sigma_decay:   float   = 0.99,
                 rng_seed:      int|None = None):
        self.pop_size       = population
        self.chrom_len      = chrom_len
        self.elite_n        = max(1, int(population * elite_fraction))
        self.k              = tournament_k
        self.sigma          = mutation_sigma
        self.sigma_decay    = sigma_decay
        self.rng            = np.random.default_rng(rng_seed)

    # ───── population bootstrap ───────────────────────────────────────
    def initial_population(self) -> np.ndarray:
        """
        Returns (pop_size, chrom_len) float32 array.
        Weights initialised N(0, 0.3); biases zero.
        """
        W = self.rng.normal(0.0, 0.3, size=(self.pop_size, 20)).astype(np.float32)
        b = np.zeros((self.pop_size, 6), dtype=np.float32)
        return np.concatenate([W, b], axis=1)

    # ───── GA step: produce next gen from current pop & fitness ───────
    def next_generation(self,
                        population: np.ndarray,
                        fitness:    np.ndarray) -> np.ndarray:

        # 1. sort by fitness (higher is better)
        idx_sorted = np.argsort(fitness)[::-1]        # descending
        elite      = population[idx_sorted[:self.elite_n]]

        # 2. make mating pool via k-way tournaments
        def tournament_pick():
            contenders = self.rng.choice(population, size=self.k, replace=False)
            contender_fit = self.rng.choice(fitness,   size=self.k, replace=False)
            return contenders[contender_fit.argmax()]

        children = []
        while len(children) < self.pop_size - self.elite_n:
            p1 = tournament_pick()
            p2 = tournament_pick()

            # 3. uniform crossover
            mask = self.rng.random(self.chrom_len) < 0.5
            child = np.where(mask, p1, p2).copy()

            # 4. Gaussian mutation
            mutation = self.rng.normal(0.0, self.sigma, self.chrom_len)
            child += mutation.astype(np.float32)

            children.append(child)

        # 5. assemble next generation & decay σ
        next_pop = np.vstack([elite] + children)[:self.pop_size]
        self.sigma *= self.sigma_decay
        return next_pop
