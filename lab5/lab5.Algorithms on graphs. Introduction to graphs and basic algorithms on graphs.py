import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
# random.seed(0)

n, m = 100, 200
matrix = np.zeros((n, n), dtype=int)

edges = 0
while edges < m:
    a = random.randint(0, n - 1)
    b = random.randint(0, n - 1)
    if a == b or matrix[a, b] != 0:
        continue
    matrix[a, b] = 1
    matrix[b, a] = 1
    edges += 1

graph = nx.from_numpy_matrix(matrix)
plt.figure(figsize=(14, 10))
plt.title('Graph Visualization')
nx.draw_spring(graph, with_labels=True)
plt.savefig('graph.png')
plt.show()


adjacency_matrix = nx.adjacency_matrix(graph).todense()
print('\nThe first 5 rows of adjacency matrix:')
for i in range(6):
    print('{}: {}'.format(i, adjacency_matrix[i]))

dict_of_lists = nx.to_dict_of_lists(graph)
print('\nThe first 5 rows of adjacency list:')
for i in range(6):
    print('{}: {}'.format(i, dict_of_lists[i]))


depth_first_args = list(nx.dfs_preorder_nodes(graph, 0))
print('\nDepth First Search:', depth_first_args)

breadth_first_args = list(nx.bidirectional_shortest_path(graph, 0, len(graph)-1))
print('Breadth First Search:', breadth_first_args)
