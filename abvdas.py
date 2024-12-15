import tsplib95 as tslib
import pandas as pd
import random
import numpy as np

# 1. Функция парсинга TSP файла
def read_tsp_with_pandas(file_path="E:\\xd\\docs\\kroA100.tsp"):
    """
    Загружает данные из TSP файла в pandas DataFrame.

    Args:
        file_path (str): Путь к файлу TSP.

    Returns:
        pd.DataFrame: Таблица с координатами узлов (NodeID, X, Y).
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

# Загрузка данных
coordinates = read_tsp_with_pandas()

# 2. Функция вычисления расстояния
def calculate_distance(node1, node2, coordinates):
    """
    Рассчитывает евклидово расстояние между двумя узлами.

    Args:
        node1 (int): ID первого узла.
        node2 (int): ID второго узла.
        coordinates (pd.DataFrame): Таблица координат.

    Returns:
        float: Расстояние между узлами.
    """
    node1_data = coordinates[coordinates["NodeID"] == node1].iloc[0]
    node2_data = coordinates[coordinates["NodeID"] == node2].iloc[0]
    x1, y1 = node1_data["X"], node1_data["Y"]
    x2, y2 = node2_data["X"], node2_data["Y"]
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance

# Пример использования
node1_id = 1
node2_id = 5
distance = calculate_distance(node1_id, node2_id, coordinates)
print(f"Distance between node {node1_id} and node {node2_id}: {distance:.2f}")

# 3. Хранение решения и создание случайного решения
def create_random_solution(coordinates):
    """
    Создаёт случайное решение (перестановка всех узлов).

    Args:
        coordinates (pd.DataFrame): Таблица координат.

    Returns:
        list: Случайная перестановка NodeID.
    """
    city_ids = coordinates["NodeID"].tolist()
    random.shuffle(city_ids)
    return city_ids

# Генерация случайного решения
random_solution = create_random_solution(coordinates)
print("Random Solution:", random_solution)

# 4. Функция расчёта "фитнеса" решения
def calculate_fitness(solution, coordinates):
    """
    Рассчитывает фитнес-значение (общую длину маршрута).

    Args:
        solution (list): Перестановка узлов (NodeID).
        coordinates (pd.DataFrame): Таблица координат.

    Returns:
        float: Общая длина маршрута.
    """
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i+1], coordinates)
    # Замыкаем маршрут
    total_distance += calculate_distance(solution[-1], solution[0], coordinates)
    return total_distance

# Пример расчёта фитнеса
fitness = calculate_fitness(random_solution, coordinates)
print(f"Fitness of random solution: {fitness:.2f}")

# 5. Вывод информации о решении
def print_solution_info(solution, fitness):
    """
    Выводит решение и его фитнес-значение.

    Args:
        solution (list): Перестановка узлов (NodeID).
        fitness (float): Фитнес-значение.
    """
    print("Solution:", " -> ".join(map(str, solution)))
    print(f"Fitness (Total Distance): {fitness:.2f}")

# Вывод информации о случайном решении
print_solution_info(random_solution, fitness)

# 6. Жадный алгоритм
def greedy_algorithm(start_node, coordinates):
    """
    Жадный алгоритм для нахождения решения TSP.

    Args:
        start_node (int): Начальный узел.
        coordinates (pd.DataFrame): Таблица координат.

    Returns:
        list: Решение (перестановка узлов).
    """
    unvisited = set(coordinates["NodeID"].tolist())
    current_node = start_node
    solution = [current_node]
    unvisited.remove(current_node)

    while unvisited:
        # Найти ближайший узел
        next_node = min(
            unvisited,
            key=lambda node: calculate_distance(current_node, node, coordinates)
        )
        solution.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return solution

# Пример запуска жадного алгоритма
start_node = 1
greedy_solution = greedy_algorithm(start_node, coordinates)
greedy_fitness = calculate_fitness(greedy_solution, coordinates)
print_solution_info(greedy_solution, greedy_fitness)

# 7. Генерация 100 случайных решений и сравнение с жадным
random_solutions = [create_random_solution(coordinates) for _ in range(100)]
random_fitnesses = [calculate_fitness(sol, coordinates) for sol in random_solutions]

# Находим лучшее случайное решение
best_random_solution = random_solutions[np.argmin(random_fitnesses)]
best_random_fitness = min(random_fitnesses)

print("\nBest Random Solution:")
print_solution_info(best_random_solution, best_random_fitness)

print("\nGreedy Solution:")
print_solution_info(greedy_solution, greedy_fitness)
