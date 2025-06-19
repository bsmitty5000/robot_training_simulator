from abc import ABC, abstractmethod
import random
import numpy as np
from courses.grid_coverage_course import GridCoverageCourse
import simulator.constants as constants
from typing import List, Tuple

from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from smart_car.smart_car import SmartCar

class VanillaGaController(ABC):
    def __init__(self, 
                 smart_car: SmartCar,
                 course: GridCoverageCourse,
                 genotype_length,
                 pop_size=20, 
                 n_generations=10,
                 initial_genotype : List[np.ndarray] = None):
        
        self.smart_car = smart_car
        self.course = course
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.genotype_length = genotype_length
        self.population: List[np.ndarray] = []
         
        if constants.NO_RANDOM:
            for _ in range(pop_size):
                self.population.append(initial_genotype)

        else:
            if initial_genotype is not None:
                for _ in range(pop_size // 2):
                    mutated = self.mutate(initial_genotype, rate=1.0)

                    self.population.append(mutated)
            for _ in range(len(self.population), pop_size):
                self.population.append(self.random_individual())

    def random_individual(self) -> np.ndarray:
        return np.random.randn(self.genotype_length) * np.sqrt(2.0 / self.genotype_length)
    
    def tournament_selection(self, pop, fitnesses, k=3):
        """Select two parents via tournament selection."""
        selected = []
        for _ in range(2):
            aspirants = random.sample(list(zip(pop, fitnesses)), k)
            selected.append(max(aspirants, key=lambda af: af[1])[0])
        return selected
    
    def single_point_crossover(self, p1, p2):
        point = random.randrange(1, len(p1))
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    
    def layer_crossover(self, p1, p2):
        idx = 0
        child0 = []
        child1 = []
        num_layers = len(self.layer_sizes) - 1
        for i in range(num_layers):
            weights_len = self.layer_sizes[i] * self.layer_sizes[i + 1]
            biases_len = self.layer_sizes[i + 1]
            w1 = p1[idx:idx + weights_len]
            w2 = p2[idx:idx + weights_len]
            b1 = p1[idx + weights_len:idx + weights_len + biases_len]
            b2 = p2[idx + weights_len:idx + weights_len + biases_len]
            idx += (weights_len + biases_len)
            
            # Crossover weights
            cut = random.randrange(1, weights_len)
            child0 += w1[:cut] + w2[cut:]
            child1 += w2[:cut] + w1[cut:]
            
            # Crossover biases
            cut = random.randrange(1, weights_len)
            child0 += b1[:cut] + b2[cut:]
            child1 += b2[:cut] + b1[cut:]
        
        return child0, child1
    
    def mutate(self, params, rate=0.01, sigma=0.1):
        return [w + random.gauss(0, sigma) if random.random()<rate else w
                for w in params]
    
    @abstractmethod
    def run(self,
            cx_rate, 
            mut_rate,
            max_steps=None) -> List[float]:
        """
        Main entry point
        """
        pass