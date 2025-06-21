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
from pathlib import Path

class GAOptimizer:
    # ───── constructor ────────────────────────────────────────────────
    def __init__(self,
                 population:        np.int32        = 256,
                 chrom_len:         np.int32        = 26,
                 elite_fraction:    np.float32  = 0.05,
                 tournament_k:      np.int32     = 3,
                 mutation_sigma:    np.float32  = 0.15,
                 sigma_decay:       np.float32   = 0.99,
                 seed_chrom:        str|None = None,
                 population_file:   str|None = None,
                 rng_seed:          np.int32|None = None):
        
        self.pop_size       = population
        self.chrom_len      = chrom_len
        self.elite_n        = max(1, np.int32(population * elite_fraction))
        self.k              = tournament_k
        self.sigma          = mutation_sigma
        self.sigma_decay    = sigma_decay
        self.rng            = np.random.default_rng(rng_seed)

        out_dir = Path("saved_chromosomes") # todo: how to pass this?
        self._seed_chrom_path   = out_dir / seed_chrom     if seed_chrom   else None
        self._population_path   = out_dir / population_file if population_file else None

    # ───── population bootstrap ───────────────────────────────────────
    def initial_population(self) -> np.ndarray:
        # 1) Resume from saved population file
        if self._population_path and self._population_path.exists():
            pop = np.load(self._population_path).astype(np.float32)
            assert pop.shape == (self.pop_size, self.chrom_len), \
                   "population_file shape mismatch"
            return pop

        # 2) Seed around a known chromosome
        if self._seed_chrom_path and self._seed_chrom_path.exists():
            seed = np.load(self._seed_chrom_path).astype(np.float32)
            assert seed.size == self.chrom_len, "seed_chrom length mismatch"

            noise = self.rng.normal(0.0, self.sigma,
                                    size=(self.pop_size, self.chrom_len)
                                    ).astype(np.float32)
            return (seed + noise)

        # 3) Pure random initialisation (default)
        return self.rng.normal(0.0, 0.3, size=(self.pop_size, self.chrom_len)).astype(np.float32)

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
