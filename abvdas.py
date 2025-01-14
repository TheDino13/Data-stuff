import tsplib95 as tslib
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def read_tsp_with_pandas():
    # Commands to choose file
    root = tk.Tk()
    root.withdraw()  # Hide window
    file_path = filedialog.askopenfilename(
        title="Choose the file",
        filetypes=[("TSP Files", "*.tsp"), ("All Files", "*.*")]
    )
    
    if not file_path:
        raise ValueError("No file")
    
    # Getting data from file
    problem = tslib.load(file_path)
    if problem.node_coords:
        coordinates = pd.DataFrame(
            [(node, x, y) for node, (x, y) in problem.node_coords.items()],
            columns=["NodeID", "X", "Y"]
        )
    else:
        coordinates = pd.DataFrame(columns=["NodeID", "X", "Y"])
    return coordinates

# Matrix to make it faster
def precompute_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i, (_, x1, y1) in coordinates.iterrows():
        for j, (_, x2, y2) in coordinates.iterrows():
            distance_matrix[i, j] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance_matrix

# Distance from the matrix
def calculate_distance(node1, node2, distance_matrix):
    return distance_matrix[node1 - 1, node2 - 1]

# "Fitness" calculation
def calculate_fitness(solution, coordinates, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i+1], distance_matrix)
    total_distance += calculate_distance(solution[-1], solution[0], distance_matrix)
    return total_distance

# Info show
def print_solution_info(solution, fitness):
    print("Solution:", " -> ".join(map(str, solution)))
    print(f"Fitness (Total Distance): {fitness:.2f}")

# Greedy algorithm
def greedy_algorithm_with_matrix(start_node, coordinates, distance_matrix):
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

# Random solutions
def create_random_solution(coordinates):
    city_ids = coordinates["NodeID"].tolist()
    random.shuffle(city_ids)
    return city_ids

def run_genetic_algorithm_roulette(coordinates, population_size=100, num_epochs=1000, mutation_prob=0.05):
    distance_matrix = precompute_distance_matrix(coordinates)
    # Increase initial population size and add elitism
    population = [create_random_solution(coordinates) for _ in range(population_size)]
    best_fitnesses, avg_fitnesses = [], []

    best_solution = min(population, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
    best_overall_fitness = calculate_fitness(best_solution, coordinates, distance_matrix)
    
    # Add elitism - keep best 10% of solutions
    elite_size = population_size // 10
    
    for epoch in range(num_epochs):
        # Sort population by fitness
        population.sort(key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
        elite = population[:elite_size]
        
        # Create new population with elitism
        new_population = elite.copy()
        
        while len(new_population) < population_size:
            # Tournament selection instead of roulette
            tournament_size = 5
            tournament = random.sample(population, tournament_size)
            parent1 = min(tournament, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
            tournament = random.sample(population, tournament_size)
            parent2 = min(tournament, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
            
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_prob)
            new_population.append(child)
        
        population = new_population
        
        # Update best solution
        epoch_best = min(population, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
        epoch_fitness = calculate_fitness(epoch_best, coordinates, distance_matrix)
        
        if epoch_fitness < best_overall_fitness:
            best_overall_fitness = epoch_fitness
            best_solution = epoch_best
        
        best_fitnesses.append(best_overall_fitness)
        avg_fitnesses.append(np.mean([calculate_fitness(sol, coordinates, distance_matrix) 
                                    for sol in population]))

        if epoch % 50 == 0:  # Print less frequently
            print(f"Epoch {epoch}: Best Fitness = {best_overall_fitness:.2f}")
    
    # Improved visualization
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitnesses, label='Best Fitness', color='blue', linewidth=2)
    plt.plot(avg_fitnesses, label='Average Fitness', color='orange', alpha=0.7)
    plt.title('Genetic Algorithm Performance Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Fitness (Total Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_solution, best_overall_fitness

# Comparison of greedy and random
def compare_greedy_and_random(coordinates, distance_matrix, num_random_solutions=100):
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

def swap_mutation(solution, mutation_prob=0.1):
    mutated = solution.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            j = random.randint(0, len(mutated) - 1)
            #Switching places
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    
    # Random segment
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    
    remaining = [x for x in parent2 if x not in child[start:end]]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    
    return child

if __name__ == "__main__":
    coordinates = read_tsp_with_pandas()
    print("\nRunning Genetic Algorithm with Enhanced Parameters...")
    best_solution, best_fitness = run_genetic_algorithm_roulette(
        coordinates,
        population_size=100,  # Increased population size
        num_epochs=1000,      # Increased epochs
        mutation_prob=0.05    # Adjusted mutation rate
    )
    print("\nBest Solution Found:")
    print_solution_info(best_solution, best_fitness)
    
    # Compare with greedy solution
    distance_matrix = precompute_distance_matrix(coordinates)
    greedy_solution = greedy_algorithm_with_matrix(1, coordinates, distance_matrix)
    greedy_fitness = calculate_fitness(greedy_solution, coordinates, distance_matrix)
    print("\nGreedy Solution for Comparison:")
    print_solution_info(greedy_solution, greedy_fitness)
    print(f"\nImprovement over greedy: {((greedy_fitness - best_fitness) / greedy_fitness) * 100:.2f}%")