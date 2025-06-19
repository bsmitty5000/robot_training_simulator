import re
import os

from courses.course1 import GridCoverageCourseA
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
from models.distance_sensors.sharp_ir import SharpIR
from models.robots.two_wheel_TT import TwoWheelTT
import simulator.constants as constants

from controllers.genetic_algorithm_controller import GeneticAlgorithmController
from controllers.optimized_controller import OptimizedGAController
from smart_car.smart_car import SmartCar

def parse_log_file(filename):
    fitness_line_pattern = re.compile(r'Fitness=([0-9.]+)\s+Genetics=genotype:\s*\[(.*)\]')
    float_pattern = re.compile(r'np\.float64\(([-+]?\d*\.\d+|\d+)\)')
    layer_sizes_pattern = re.compile(r'Layer sizes:\s*\[([0-9,\s]+)\]')
    best_fitness = float('-inf')
    best_genotype = None
    layer_sizes = [3, 4, 2]

    with open(filename, "r") as f:
        for line in f:
            match = layer_sizes_pattern.search(line)
            if match:
                layer_sizes_str = match.group(1)
                layer_sizes = [int(x) for x in layer_sizes_str.split(',')]

            # Match lines with Fitness and Genetics
            match = fitness_line_pattern.search(line)
            if match:
                fitness = float(match.group(1))
                genotype_str = match.group(2)
                # Extract all float values (ignore np.float64 wrappers)
                floats = [float(x) for x in float_pattern.findall(genotype_str)]
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genotype = floats
    
    return layer_sizes, best_fitness, best_genotype

def main():
    layer_sizes, best_fitness, best_genotype = [3, 4, 2], 0.0, None
    if os.path.exists(constants.LOG_FILE_TO_SEED):
        layer_sizes, best_fitness, best_genotype = parse_log_file(constants.LOG_FILE_TO_SEED)

        print(f"Seeding with previous genotype that had fitness of: {best_fitness}")

    # Set up sensors, robot, decision maker, smart car, and course
    course = GridCoverageCourseA(constants.WIDTH, constants.HEIGHT)
    sensors = [SharpIR(-45, 0.3), SharpIR(0, 0.3), SharpIR(45, 0.3)]
    robot_instance = TwoWheelTT(distance_sensors=sensors)
    decision_maker = FFNetDecisionMaker(layer_sizes)
    smart_car = SmartCar(robot_instance, decision_maker)
    
    ga = GeneticAlgorithmController(smart_car,
                                    course,
                                    pop_size=5, 
                                    n_generations=2,
                                    initial_genotype=best_genotype)
    
    # ga = OptimizedGAController(smart_car,
    #                                 course,
    #                                 pop_size=5, 
    #                                 n_generations=2,
    #                                 initial_genotype=best_genotype)
    
    if constants.DEMO_RUN:
        weights, biases = FFNetDecisionMaker.from_genotype(best_genotype, layer_sizes=ga.layer_sizes)
        ga.evaluate_individual(weights, biases,
                                constants.WIDTH, constants.HEIGHT, 
                                individual_idx=0, generation=0)
    else:
        #ga.evolve(screen, clock, constants.WIDTH, constants.HEIGHT)
        cx_rate = 0.7
        mut_rate = 0.03
        if constants.NO_RANDOM:
            cx_rate = 0.0
            mut_rate = 0.0
        fitness_history = ga.run(cx_rate=cx_rate,
                                  mut_rate=mut_rate,
                                  max_steps=5000)
        
        
        # plt.plot(fitness_history)
        # plt.title("Best Fitness over Generations")
        # plt.xlabel("Generation")
        # plt.ylabel("Fitness")
        # plt.ioff()
        # plt.show()
    
if __name__ == "__main__":
    main()