import tsplib95 as tslib
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. Парсинг TSP файла
def read_tsp_with_pandas(file_path="E:\\xd\\docs\\berlin52.tsp"):
    """
    Загружает данные из TSP файла в pandas DataFrame.
    """
    problem = tslib.load(file_path)
    if problem.node_coords:
        coordinates = pd.DataFrame(
            [(node, x, y) for node, (x, y) in problem.node_coords.items()],
            columns=["NodeID", "X", "Y"]
        )
    else:
        coordinates = pd.DataFrame(columns=["NodeID", "X", "Y"])
    return coordinates

# 2. Предварительная матрица расстояний
def precompute_distance_matrix(coordinates):
    """
    Создаёт матрицу расстояний для ускорения вычислений.
    """
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i, (_, x1, y1) in coordinates.iterrows():
        for j, (_, x2, y2) in coordinates.iterrows():
            distance_matrix[i, j] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance_matrix

# 3. Вычисление расстояния из матрицы
def calculate_distance(node1, node2, distance_matrix):
    return distance_matrix[node1 - 1, node2 - 1]

# 4. Функция расчёта фитнеса
def calculate_fitness(solution, coordinates, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i+1], distance_matrix)
    total_distance += calculate_distance(solution[-1], solution[0], distance_matrix)
    return total_distance

# 5. Вывод информации о решении
def print_solution_info(solution, fitness):
    print("Solution:", " -> ".join(map(str, solution)))
    print(f"Fitness (Total Distance): {fitness:.2f}")

# 6. Жадный алгоритм с матрицей расстояний
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

# 7. Генерация случайных решений
def create_random_solution(coordinates):
    city_ids = coordinates["NodeID"].tolist()
    random.shuffle(city_ids)
    return city_ids

# 8. Генерация новой эпохи
def create_new_epoch_roulette(previous_population, coordinates, distance_matrix, population_size, mutation_prob=0.1):
    fitness_values = [calculate_fitness(ind, coordinates, distance_matrix) for ind in previous_population]
    new_population = []
    best_fitness = float('inf')
    best_solution = None

    while len(new_population) < population_size:
        # Селекция: рулетка
        parent1 = roulette_wheel_selection(previous_population, fitness_values)
        parent2 = roulette_wheel_selection(previous_population, fitness_values)
        
        # Кроссовер
        child = ordered_crossover(parent1, parent2)
        
        # Мутация
        child = swap_mutation(child, mutation_prob)
        
        # Фитнес
        fitness = calculate_fitness(child, coordinates, distance_matrix)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = child
            
        new_population.append(child)
    
    return new_population, best_solution, best_fitness

# 9. Генетический алгоритм
def run_genetic_algorithm_roulette(coordinates, population_size=50, num_epochs=100, mutation_prob=0.1):
    distance_matrix = precompute_distance_matrix(coordinates)
    population = [create_random_solution(coordinates) for _ in range(population_size)]
    best_fitnesses, avg_fitnesses = [], []

    best_solution = min(population, key=lambda x: calculate_fitness(x, coordinates, distance_matrix))
    best_overall_fitness = calculate_fitness(best_solution, coordinates, distance_matrix)
    
    for epoch in range(num_epochs):
        population, epoch_best, epoch_fitness = create_new_epoch_roulette(
            population, coordinates, distance_matrix, population_size, mutation_prob
        )
        if epoch_fitness < best_overall_fitness:
            best_overall_fitness = epoch_fitness
            best_solution = epoch_best
        
        best_fitnesses.append(best_overall_fitness)
        avg_fitness = np.mean([calculate_fitness(sol, coordinates, distance_matrix) for sol in population])
        avg_fitnesses.append(avg_fitness)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Best Fitness = {best_overall_fitness:.2f}")
    
    plt.plot(best_fitnesses, label='Best Fitness')
    plt.plot(avg_fitnesses, label='Average Fitness')
    plt.legend()
    plt.show()
    
    return best_solution, best_overall_fitness
# Функция рулетки
def roulette_wheel_selection(population, fitness_values):
    """
    Выбирает решение из популяции на основе рулетки (вероятность пропорциональна обратному фитнесу).
    """
    total_fitness = sum(1.0 / f for f in fitness_values)
    pick = random.uniform(0, total_fitness)
    current = 0
    
    for individual, fitness in zip(population, fitness_values):
        current += 1.0 / fitness
        if current >= pick:
            return individual
    return population[-1]  # В случае ошибки возврат последнего элемента

# 10. Сравнение случайных и жадных решений

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
    """
    Мутация решения путем обмена двух случайных городов с заданной вероятностью.
    """
    mutated = solution.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            j = random.randint(0, len(mutated) - 1)
            # Обмен местами
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def ordered_crossover(parent1, parent2):
    """
    Упорядоченный кроссовер (Ordered Crossover - OX).
    """
    size = len(parent1)
    child = [-1] * size
    
    # Выбираем случайный сегмент
    start, end = sorted(random.sample(range(size), 2))
    
    # Копируем сегмент из первого родителя
    child[start:end] = parent1[start:end]
    
    # Заполняем оставшиеся позиции из второго родителя
    remaining = [x for x in parent2 if x not in child[start:end]]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    
    return child

# Основной запуск
if __name__ == "__main__":
    coordinates = read_tsp_with_pandas()
    distance_matrix = precompute_distance_matrix(coordinates)
    compare_greedy_and_random(coordinates, distance_matrix)
    print("\nRunning Genetic Algorithm...")
    best_solution, best_fitness = run_genetic_algorithm_roulette(coordinates)
    print("\nBest Solution from Genetic Algorithm:")
    print_solution_info(best_solution, best_fitness)
