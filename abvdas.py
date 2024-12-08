import tsplib95 as tslib
import pandas as pd

import tsplib95
import pandas as pd

def read_tsp_with_pandas(file_path="E:\\xd\\docs\\kroA100.tsp"):
   # Загрузка TSP файла
    problem = tsplib95.load(file_path)
    # Преобразование координат узлов в DataFrame
    if problem.node_coords:
        coordinates = pd.DataFrame( [(node, x, y) for node, (x, y) in problem.node_coords.items()], columns=["NodeID", "X", "Y"])
    else:
        coordinates = pd.DataFrame(columns=["NodeID", "X", "Y"])  # Пустая таблица, если координат нет

    return coordinates
coordinates = read_tsp_with_pandas()


print("Координаты узлов:\n")
print(coordinates)

node_id = 5
node_data = coordinates[coordinates["NodeID"] == node_id]
if not node_data.empty:
    x, y = node_data.iloc[0]["X"], node_data.iloc[0]["Y"]
    print(f"Координаты для узла {node_id}: X={x}, Y={y}\n")
else:
    print(f"Узел {node_id} не найден.\n")

def calculate_distance(node1, node2, coordinates):
    node1_data = coordinates[coordinates["NodeID"] == node1].iloc[0]
    node2_data = coordinates[coordinates["NodeID"] == node2].iloc[0]
    x1, y1 = node1_data["X"], node1_data["Y"]
    x2, y2 = node2_data["X"], node2_data["Y"]
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance

# Example usage:
node1_id = 1
node2_id = 5
distance = calculate_distance(node1_id, node2_id, coordinates)
print(f"Distance between node {node1_id} and node {node2_id}: {distance:.2f}")