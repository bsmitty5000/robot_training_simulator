import random
import matplotlib.pyplot as plt

def random_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def decode(individual):
    """Convert binary list to integer."""
    return sum(bit << idx for idx, bit in enumerate(reversed(individual)))

def fitness(individual):
    """Objective: maximize x^2 where x is the decoded integer."""
    x = decode(individual)
    return x * x

def tournament_selection(pop, fitnesses, k=3):
    """Select two parents via tournament selection."""
    selected = []
    for _ in range(2):
        aspirants = random.sample(list(zip(pop, fitnesses)), k)
        selected.append(max(aspirants, key=lambda af: af[1])[0])
    return selected

def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(individual, rate):
    return [bit ^ 1 if random.random() < rate else bit for bit in individual]

def genetic_algorithm(pop_size, 
                      chrom_length, 
                      generations, 
                      cx_rate, 
                      mut_rate):
    # Initialize population
    population = [random_individual(chrom_length) for _ in range(pop_size)]
    best_history = []

    for gen in range(generations):
        fits = [fitness(ind) for ind in population]
        best = max(fits)
        best_history.append(best)
        
        # Create new generation
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = tournament_selection(population, fits)
            if random.random() < cx_rate:
                o1, o2 = single_point_crossover(p1, p2)
            else:
                o1, o2 = p1[:], p2[:]
            new_pop.append(mutate(o1, mut_rate))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(o2, mut_rate))
        population = new_pop

    # Final results
    fits = [fitness(ind) for ind in population]
    best_idx = fits.index(max(fits))
    best_ind = population[best_idx]
    print(f"Best individual (binary): {best_ind}")
    print(f"Decoded value: {decode(best_ind)}")
    print(f"Best fitness: {fits[best_idx]}")

    # Plot fitness over generations
    plt.plot(best_history)
    plt.title("Best Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

# Run the algorithm with simple parameters
if __name__ == "__main__":
    genetic_algorithm(
        pop_size=10,
        chrom_length=10,
        generations=20,
        cx_rate=0.7,
        mut_rate=0.01
    )
