import random
import numpy as np
from courses.grid_coverage_course import GridCoverageCourse
import simulator.constants as constants
from typing import List

from sim import core
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from smart_car.smart_car import SmartCar
import logging
from datetime import datetime
from .vanilla_ga_controller import VanillaGaController

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"logs/ga_{timestamp}.log"

# Configure logging (you can adjust level and format as needed)
logging.basicConfig(
    filename=filename,  # Log to a file; use None or remove filename for console
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class GeneticAlgorithmController(VanillaGaController):
    def __init__(self, 
                 smart_car: SmartCar,
                 course: GridCoverageCourse,
                 pop_size=20, 
                 n_generations=10,
                 initial_genotype : List[np.ndarray] = None):
        super().__init__(smart_car, course, pop_size, n_generations, initial_genotype)
        self.obstacles = course.make_course()

    def evaluate_individual(self, 
                            weights, 
                            biases,
                            max_steps=None,
                            individual_idx=None, 
                            generation=None) -> float:
        
        self.smart_car.robot.reset()
        self.smart_car.decision_maker.set_weights(weights, biases)
        self.course.reset()
        sim_time = self.run_simulation(max_steps) * constants.SIM_DT
        coverage = self.course.coverage_ratio()
        fitness = coverage * 1000.0 / sim_time
        logging.info(
            f"Gen {generation} Ind {individual_idx}: Coverage={coverage:.4f}: Time={sim_time:.4f}: Fitness={fitness:.4f} Genetics={self.smart_car.decision_maker.print_info()}"
        )
        return fitness

    def run_simulation(self, max_steps=None):
        
        running = True
        total_frames = 0
        previous_coverage = 0.0
        coverage_stale_count = 0

        while running:

            self.smart_car.update(constants.SIM_DT, self.obstacles)

            if(total_frames < 10):
                print(total_frames, 
                      self.smart_car.robot.x_coordinate, 
                      self.smart_car.robot.y_coordinate, 
                      self.smart_car.robot.angle_deg, 
                      self.smart_car.robot.velocity)

            self.course.mark_visited(self.smart_car.robot.x_coordinate,
                                     self.smart_car.robot.y_coordinate)
            
            current_coverage = self.course.coverage_ratio()
            if current_coverage > previous_coverage:
                coverage_stale_count = 0
            else:
                coverage_stale_count += 1

            previous_coverage = current_coverage

            if coverage_stale_count > constants.COVERAGE_ABORT_S * constants.FRAME_RATE:
                # If coverage hasn't improved for a while, restart the simulation
                break

            for obstacle in self.obstacles:
                if(core.circle_rect_collision(self.smart_car.robot.circle, obstacle)):
                    break

            total_frames += 1
            if max_steps is not None and total_frames >= max_steps:
                break
        
        return total_frames

    def run(self,
            cx_rate, 
            mut_rate,
            max_steps=None) -> List[float]:
        
        logging.info(f"Starting Genetic Algorithm with {self.pop_size} individuals for {self.n_generations} generations.")
        logging.info(f"Layer sizes: {self.layer_sizes}")

        best_history = []
        for gen in range(self.n_generations):
            
            fits = []
            for idx, (weights, biases) in enumerate(self.population):
                fitness = self.evaluate_individual(weights, biases,
                                                   max_steps, 
                                                   individual_idx=idx, generation=gen)
                fits.append(fitness)

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
