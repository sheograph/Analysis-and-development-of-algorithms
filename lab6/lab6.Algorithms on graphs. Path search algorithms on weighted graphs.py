import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import timeit
import random
# random.seed(0)


def create_graph(n, m):
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
    return graph


n = 100     # Nodes
m = 500     # Edges
graph = create_graph(n, m)

for (u, v) in graph.edges():
    graph.edges[u, v]['weight'] = np.random.randint(0, 100)

adjacency_matrix = nx.adjacency_matrix(graph).todense()

for row in adjacency_matrix[:3]:
    print(row)


plt.figure(figsize=(15, 8))
plt.title('Graph Visualization')
nx.draw_spring(graph, with_labels=True)
# pos = nx.spring_layout(graph)
# nx.draw_networkx(graph, pos)
# labels = nx.get_edge_attributes(graph, 'weight')
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.savefig('graph.png')
plt.show()


def dijkstra_path(graph, source, target):
    return nx.dijkstra_path(G=graph, source=source, target=target)


def bellman_ford_path(graph, source, target):
    return nx.bellman_ford_path(G=graph, source=source, target=target)


def timestamp(algo, graph):
    ts = np.array([])
    ts_ = np.array([datetime.datetime.now().timestamp()])
    ap = algo(graph, 0, 10)
    ts_ = np.append(ts_, datetime.datetime.now().timestamp())
    ts = np.append(ts, ts_[1] - ts_[0])
    return ap, ts


runs = 10
ts_d = np.array([])
ts_bf = np.array([])
for i in range(0, runs):
    graph = create_graph(n, m)
    ap_d, ts_ = timestamp(dijkstra_path, graph)
    ts_d = np.append(ts_d, ts_)
    ap_bf, ts_ = timestamp(bellman_ford_path, graph)
    ts_bf = np.append(ts_bf, ts_)
ts_d = np.sum(ts_d)/runs
ts_bf = np.sum(ts_bf)/runs
print('Dijkstra Algorithm:', ap_d, '\n', ts_d, 'seconds')
print('Bellman-Ford Algorithm:', ap_bf, '\n', ts_bf, 'seconds')




x_grid = 20
y_grid = 10
obstacles_number = 30
graph = nx.grid_2d_graph(y_grid, x_grid)

removed = 0
while removed < obstacles_number:
    x = random.randint(0, x_grid - 1)
    y = random.randint(0, y_grid - 1)
    if (y, x) not in graph:
        continue
    graph.remove_node((y, x))
    removed += 1

plt.figure(figsize=(15, 8))
pos = {(x, y): (y, -x) for x, y in graph.nodes()}
nx.draw(graph, pos=pos, with_labels=True, node_size=1000, font_size=9)
plt.savefig('A*.png')
plt.show()

for i in range(5):
    a = random.choice(list(graph.nodes.keys()))
    b = random.choice(list(graph.nodes.keys()))
    print('\nA* Algorithm\nFrom', a, 'to', b)

    time_start = timeit.default_timer()
    path = nx.astar_path(graph, a, b)
    time_end = timeit.default_timer()
    print('Path:', path)
    print(time_end - time_start, 'seconds')

plt.figure(figsize=(15, 8))
pos = {(x, y): (y, -x) for x, y in graph.nodes()}
nx.draw(graph, pos=pos, with_labels=True, node_size=1000, font_size=9)
path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=3)
plt.savefig('A* path.png')
plt.show()