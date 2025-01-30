import tsplib95 as tslib
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ----------------
# Data Loading and Preprocessing
# ----------------

def read_tsp_with_pandas():
    # Opens a file dialog to select and read a TSP problem file.
    # Returns a DataFrame with node coordinates.
    root = tk.Tk()
    root.withdraw()  # Hide window
    file_path = filedialog.askopenfilename(
        title="Choose the file",
        filetypes=[("TSP Files", "*.tsp"), ("All Files", "*.*")]
    )
    
    if not file_path:
        raise ValueError("No file")
    
    problem = tslib.load(file_path)
    if problem.node_coords:
        coordinates = pd.DataFrame(
            [(node, x, y) for node, (x, y) in problem.node_coords.items()],
            columns=["NodeID", "X", "Y"]
        )
    else:
        coordinates = pd.DataFrame(columns=["NodeID", "X", "Y"])
    return coordinates

def precompute_distance_matrix(coordinates):
    # Creates a distance matrix for faster distance calculations.
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i, (_, x1, y1) in coordinates.iterrows():
        for j, (_, x2, y2) in coordinates.iterrows():
            distance_matrix[i, j] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance_matrix

def calculate_distance(node1, node2, distance_matrix):
    # Calculates distance between two nodes using the precomputed distance matrix.
    return distance_matrix[node1 - 1, node2 - 1]

def calculate_fitness(solution, coordinates, distance_matrix):
    # Calculates total distance (fitness) of a route.
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i+1], distance_matrix)
    total_distance += calculate_distance(solution[-1], solution[0], distance_matrix)
    return total_distance

# ----------------
# Basic Solution Generators
# ----------------

def create_random_solution(coordinates):
    # Creates a random route visiting all cities.
    city_ids = coordinates["NodeID"].tolist()
    random.shuffle(city_ids)
    return city_ids

def run_random_solutions(coordinates, num_runs=1000):
    # Runs random solutions and calculates statistics.
    distance_matrix = precompute_distance_matrix(coordinates)
    fitnesses = []  # storage list

    for _ in range(num_runs):
        random_solution = create_random_solution(coordinates)
        fitness = calculate_fitness(random_solution, coordinates, distance_matrix)
        fitnesses.append(fitness)

    best_fitness = min(fitnesses)
    avg_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    variance_fitness = np.var(fitnesses)

    print(f"\nRandom Solutions (1000 runs):")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Average Fitness: {avg_fitness:.2f}")
    print(f"Standard Deviation: {std_fitness:.2f}")
    print(f"Variance: {variance_fitness:.2f}")

    return best_fitness, avg_fitness, std_fitness, variance_fitness

def greedy_algorithm_with_matrix(start_node, coordinates, distance_matrix):
    # Implements a greedy nearest-neighbor algorithm for TSP.
    unvisited = set(coordinates["NodeID"].tolist())
    current_node = start_node
    solution = [current_node]
    unvisited.remove(current_node)

    while unvisited:
        next_node = min(
            unvisited,
            key=lambda node: distance_matrix[current_node - 1, node - 1]
        )
        solution.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return solution

# ----------------
# Genetic Algorithm Components
# ----------------

def swap_mutation(solution, mutation_prob=0.1):
    # Performs mutation by swapping random cities in the route.
    mutated = solution.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def ordered_crossover(parent1, parent2):
    # Performs ordered crossover between two parent solutions.
    size = len(parent1)
    child = [-1] * size
    
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    
    remaining = [x for x in parent2 if x not in child[start:end]]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    
    return child

def plot_solution(coordinates, solution, title="Best Solution"):
    # Plots the best solution found.
    x = [coordinates.loc[coordinates["NodeID"] == node, "X"].values[0] for node in solution]
    y = [coordinates.loc[coordinates["NodeID"] == node, "Y"].values[0] for node in solution]

    x.append(x[0])
    y.append(y[0])
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label="Route")
    plt.scatter(x, y, color='red')  # coordinates of cities
    for i, node in enumerate(solution):
        plt.text(x[i], y[i], f"{node}", fontsize=12, ha='right')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
'''
lambda is the key to use function only ONCE and fast without wasting time
2f for the number with 2 numbers after dot
'''

def run_genetic_algorithm_roulette(coordinates, population_size=100, num_epochs=1000, mutation_prob=0.05, tournament_size=5, verbose=False):
    # Main genetic algorithm implementation with tournament selection and elitism.
    distance_matrix = precompute_distance_matrix(coordinates)
    population = [create_random_solution(coordinates) for _ in range(population_size)]
    best_fitnesses, avg_fitnesses = [], []

    best_solution = min(population, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
    best_overall_fitness = calculate_fitness(best_solution, coordinates, distance_matrix)
    
    elite_size = population_size // 10
    
    for epoch in range(num_epochs):
        population.sort(key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
        elite = population[:elite_size]
        new_population = elite.copy()
        
        while len(new_population) < population_size:
            tournament = random.sample(population, tournament_size)
            parent1 = min(tournament, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
            tournament = random.sample(population, tournament_size)
            parent2 = min(tournament, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
            
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_prob)
            new_population.append(child)
        
        population = new_population
        
        epoch_best = min(population, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
        epoch_fitness = calculate_fitness(epoch_best, coordinates, distance_matrix)
        
        if epoch_fitness < best_overall_fitness:
            best_overall_fitness = epoch_fitness
            best_solution = epoch_best
        
        best_fitnesses.append(best_overall_fitness)
        avg_fitnesses.append(np.mean([calculate_fitness(sol, coordinates, distance_matrix) 
                                    for sol in population]))
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}: Best Fitness = {best_overall_fitness:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitnesses, label='Best Fitness', color='blue', linewidth=2)
    plt.plot(avg_fitnesses, label='Average Fitness', color='orange', alpha=0.7)
    plt.title('Genetic Algorithm Performance Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Fitness (Total Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    if len(coordinates) == 11:
        plot_solution(coordinates, best_solution, title="Best Solution for 11 Cities")
    
    return best_solution, best_overall_fitness, best_fitnesses, avg_fitnesses

# ----------------
# Analysis and Comparison Functions
# ----------------

def print_solution_info(solution, fitness):
    # Prints formatted information about a solution.
    print("Solution:", " -> ".join(map(str, solution)))
    print(f"Fitness (Total Distance): {fitness:.2f}")

def compare_greedy_and_random(coordinates, distance_matrix, num_random_solutions=100):
    # Compares performance of greedy and random solutions.
    start_node = 1
    greedy_solution = greedy_algorithm_with_matrix(start_node, coordinates, distance_matrix)
    greedy_fitness = calculate_fitness(greedy_solution, coordinates, distance_matrix)
    print("\nGreedy Solution:")
    print_solution_info(greedy_solution, greedy_fitness)

    random_solutions = [create_random_solution(coordinates) for _ in range(num_random_solutions)]
    random_fitnesses = [calculate_fitness(sol, coordinates, distance_matrix) for sol in random_solutions]

    best_random_solution = random_solutions[np.argmin(random_fitnesses)]
    best_random_fitness = min(random_fitnesses)
    print("\nBest Random Solution:")
    print_solution_info(best_random_solution, best_random_fitness)

    plt.hist(random_fitnesses, bins=20, alpha=0.7, label="Random Solutions")
    plt.axvline(greedy_fitness, color='red', label="Greedy Solution")
    plt.legend()
    plt.show()

def compare_mutation_rates_simple(coordinates, test_mutation_rates=False):
    # Compares different mutation rates.
    if not test_mutation_rates:
        return  # if tests not needed skip

    plt.figure(figsize=(10, 6))
    results = {}
    mutation_rates = [0.01, 0.5]

    for rate in mutation_rates:
        print(f"\nTesting mutation rate: {rate}")
        best_solution, fitness = run_genetic_algorithm_roulette(
            coordinates,
            population_size=100,
            num_epochs=1000,
            mutation_prob=rate
        )
        results[rate] = fitness

    plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen'])
    plt.title('Mutation Rates Comparison')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Best Fitness (Distance)')

    for i, v in results.items():
        plt.text(i, v, f'{v:.0f}', ha='center', va='bottom')

    plt.show()

    best_rate = min(results.items(), key=lambda x: x[1])
    print(f"\nBest mutation rate: {best_rate[0]} (fitness: {best_rate[1]:.2f})")

# ----------------
# New Comparison Functions
# ----------------

def compare_mutation_rates(coordinates):
    # Compares different mutation rates.
    mutation_rates = [0.1, 0.5, 0.05]
    results = {}

    for rate in mutation_rates:
        print(f"\nTesting mutation rate: {rate}")
        best_solution, best_fitness, best_fitnesses, avg_fitnesses = run_genetic_algorithm_roulette(
            coordinates,
            population_size=100,
            num_epochs=1000,
            mutation_prob=rate
        )
        results[rate] = (best_fitnesses, avg_fitnesses)

    plt.figure(figsize=(12, 6))
    for rate, (best_fitnesses, avg_fitnesses) in results.items():
        plt.plot(best_fitnesses, label=f'Mutation Rate {rate}')

    plt.title('Comparison of Mutation Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness (Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_population_sizes(coordinates):
    # Compares different population sizes.
    population_sizes = [50, 100, 200]
    results = {}

    for size in population_sizes:
        print(f"\nTesting population size: {size}")
        best_solution, best_fitness, best_fitnesses, avg_fitnesses = run_genetic_algorithm_roulette(
            coordinates,
            population_size=size,
            num_epochs=1000,
            mutation_prob=0.05
        )
        results[size] = (best_fitnesses, avg_fitnesses)

    plt.figure(figsize=(12, 6))
    for size, (best_fitnesses, avg_fitnesses) in results.items():
        plt.plot(best_fitnesses, label=f'Population Size {size}')

    plt.title('Comparison of Population Sizes')
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness (Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_tournament_sizes(coordinates):
    # Compares different tournament sizes.
    tournament_sizes = [3, 5, 10]
    results = {}

    for size in tournament_sizes:
        print(f"\nTesting tournament size: {size}")
        best_solution, best_fitness, best_fitnesses, avg_fitnesses = run_genetic_algorithm_roulette(
            coordinates,
            population_size=100,
            num_epochs=1000,
            mutation_prob=0.05,
            tournament_size=size
        )
        results[size] = (best_fitnesses, avg_fitnesses)

    plt.figure(figsize=(12, 6))
    for size, (best_fitnesses, avg_fitnesses) in results.items():
        plt.plot(best_fitnesses, label=f'Tournament Size {size}')

    plt.title('Comparison of Tournament Sizes')
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness (Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Load data
    coordinates = read_tsp_with_pandas()
    
    # Run genetic algorithm
    print("\nRunning Genetic Algorithm with Enhanced Parameters...")
    best_solution, best_fitness = run_genetic_algorithm_roulette(
        coordinates,
        population_size=100,
        num_epochs=1000,
        mutation_prob=0.05,
        verbose=False
    )
    print("\nBest Solution Found:")
    print_solution_info(best_solution, best_fitness)
    run_random_solutions(coordinates, num_runs=1000)
    
    # Compare mutation rates
    compare_mutation_rates_simple(coordinates)
    test_mutation = input("Test mutation rates 0.5 and 0.1? (y/n): ").strip().lower()
    compare_mutation_rates_simple(coordinates, test_mutation == 'y')
    
    # Compare with greedy solution
    distance_matrix = precompute_distance_matrix(coordinates)
    greedy_solution = greedy_algorithm_with_matrix(1, coordinates, distance_matrix)
    greedy_fitness = calculate_fitness(greedy_solution, coordinates, distance_matrix)
    print("\nGreedy Solution for Comparison:")
    print_solution_info(greedy_solution, greedy_fitness)
    print(f"\nImprovement over greedy: {((greedy_fitness - best_fitness) / greedy_fitness) * 100:.2f}%")
    
    # New comparison functions
    compare_mutation_rates(coordinates)
    compare_population_sizes(coordinates)
    compare_tournament_sizes(coordinates)