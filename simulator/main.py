import re
import os

from matplotlib import pyplot as plt
from ml_stuff.ff_net_decision_maker import FFNetDecisionMaker
import simulator.constants as constants

from controllers.genetic_algorithm_controller import GeneticAlgorithmController

def parse_log_file(filename):
    fitness_line_pattern = re.compile(r'Fitness=([0-9.]+)\s+Genetics=genotype:\s*\[(.*)\]')
    float_pattern = re.compile(r'np\.float64\(([-+]?\d*\.\d+|\d+)\)')
    layer_sizes_pattern = re.compile(r'Layer sizes:\s*\[([0-9,\s]+)\]')
    best_fitness = float('-inf')
    best_genotype = None
    layer_sizes = None

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
    layer_sizes, best_fitness, best_genotype = None, 0.0, None
    if os.path.exists(constants.LOG_FILE_TO_SEED):
        layer_sizes, best_fitness, best_genotype = parse_log_file(constants.LOG_FILE_TO_SEED)
        print(f"Seeding with previous genotype that had fitness of: {best_fitness}")

    ga = GeneticAlgorithmController(pop_size=10, 
                                    n_generations=5, 
                                    layer_sizes=layer_sizes, 
                                    initial_genotype=best_genotype)
    if constants.DEMO_RUN:
        weights, biases = FFNetDecisionMaker.from_genotype(best_genotype, layer_sizes=ga.layer_sizes)
        ga.evaluate_individual(weights, biases,
                                constants.WIDTH, constants.HEIGHT, 
                                individual_idx=0, generation=0)
    else:
        #ga.evolve(screen, clock, constants.WIDTH, constants.HEIGHT)
        fitness_history = ga.run(constants.WIDTH, constants.HEIGHT,
                cx_rate=0.7,
                mut_rate=0.03)
        
        
        # plt.plot(fitness_history)
        # plt.title("Best Fitness over Generations")
        # plt.xlabel("Generation")
        # plt.ylabel("Fitness")
        # plt.ioff()
        # plt.show()
    
if __name__ == "__main__":
    main()