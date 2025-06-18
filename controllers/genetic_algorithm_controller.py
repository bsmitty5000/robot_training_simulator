import random
import numpy as np
import simulator.constants as constants
from typing import List, Tuple

from sim import core
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from smart_car.smart_car import SmartCar
from models.robots.two_wheel_TT import TwoWheelTT
from models.distance_sensors.sharp_ir import SharpIR
from courses.course1 import GridCoverageCourseA
import controllers.helpers as helpers
import logging
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"logs/ga_{timestamp}.log"

# Configure logging (you can adjust level and format as needed)
logging.basicConfig(
    filename=filename,  # Log to a file; use None or remove filename for console
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class GeneticAlgorithmController:
    def __init__(self, pop_size=20, n_generations=10, layer_sizes = None, initial_genotype : List[np.ndarray] = None):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.layer_sizes = layer_sizes if layer_sizes is not None else [3, 4, 2]  # Default layer sizes
        self.population: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
        if initial_genotype is not None:
            for _ in range(pop_size // 2):
                mutated = FFNetDecisionMaker.from_genotype(self.mutate(initial_genotype, rate=1.0), 
                                                           layer_sizes=self.layer_sizes)
                self.population.append(mutated)
        for _ in range(len(self.population), pop_size):
            self.population.append(self.random_individual())

    def random_individual(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        weights = [
            np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            for n_in, n_out in zip(self.layer_sizes, self.layer_sizes[1:])
        ]
        biases = [
            np.zeros(n_out)
            for n_out in self.layer_sizes[1:]
        ]
        return (weights, biases)
    
    def evaluate_individual(self, 
                            weights, 
                            biases, 
                            width, height, 
                            individual_idx=None, 
                            generation=None) -> float:
        # Set up sensors, robot, decision maker, smart car, and course
        sensors = [SharpIR(-45, 0.3), SharpIR(0, 0.3), SharpIR(45, 0.3)]
        robot_instance = TwoWheelTT(75, 75, distance_sensors=sensors)
        decision_maker = FFNetDecisionMaker(self.layer_sizes)
        decision_maker.set_weights(weights, biases)
        smart_car = SmartCar(robot_instance, decision_maker)
        course = GridCoverageCourseA(width, height)
        # Run simulation headless (no display) or with display as needed
        sim_time = self.run_simulation(smart_car, course) * constants.SIM_DT
        coverage = course.coverage_ratio()
        fitness = coverage * 1000.0 / sim_time
        logging.info(
            f"Gen {generation} Ind {individual_idx}: Coverage={coverage:.4f}: Time={sim_time:.4f}: Fitness={fitness:.4f} Genetics={smart_car.decision_maker.print_info()}"
        )
        return fitness

    def run_simulation(self, smart_car, course, max_steps=None):
        
        running = True
        total_frames = 0
        previous_coverage = 0.0
        coverage_stale_count = 0

        obstacles = course.make_course()

        while running:

            smart_car.update(constants.SIM_DT, obstacles)

            course.mark_visited(smart_car.robot.x_coordinate,
                                smart_car.robot.y_coordinate)
            
            current_coverage = course.coverage_ratio()
            if current_coverage > previous_coverage:
                coverage_stale_count = 0
            else:
                coverage_stale_count += 1

            previous_coverage = current_coverage

            if coverage_stale_count > constants.COVERAGE_ABORT_S * constants.FRAME_RATE:
                # If coverage hasn't improved for a while, restart the simulation
                break

            for obstacle in obstacles:
                if(core.circle_rect_collision(smart_car.robot.circle, obstacle)):
                    break

            total_frames += 1
            if max_steps is not None and total_frames >= max_steps:
                break
        
        return total_frames

    def evolve(self, width, height):
        for gen in range(self.n_generations):
            fitnesses = []
            for idx, (weights, biases) in enumerate(self.population):
                fitness = self.evaluate_individual(weights, biases, width, height, individual_idx=idx, generation=gen)
                fitnesses.append(fitness)

            best = max(fitnesses)
            avg = sum(fitnesses) / len(fitnesses)

            logging.info(f"Generation {gen}: Best fitness={best:.4f}, Avg fitness={avg:.4f}")
            # Select, crossover, mutate (implement your GA logic here)
            # For now, just keep the top N
            sorted_pop = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]
            self.population = sorted_pop[:self.pop_size]
            best_idx = fitnesses.index(best)


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

    def run(self,
            width, height,
            cx_rate, 
            mut_rate) -> List[float]:
        
        logging.info(f"Starting Genetic Algorithm with {self.pop_size} individuals for {self.n_generations} generations.")
        logging.info(f"Layer sizes: {self.layer_sizes}")

        best_history = []
        for gen in range(self.n_generations):
            
            fits = []
            for idx, (weights, biases) in enumerate(self.population):
                fitness = self.evaluate_individual(weights, biases,
                                                   width, height, 
                                                   individual_idx=idx, generation=gen)
                fits.append(fitness)

            best = max(fits)
            best_history.append(best)
            logging.info(f"Generation {gen}: Best fitness={best:.4f}")
            
            # Create new generation
            new_pop = []
            while len(new_pop) < len(self.population):
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
