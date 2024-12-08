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
