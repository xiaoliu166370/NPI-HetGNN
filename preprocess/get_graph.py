import numpy as np
import pickle
import dgl
import torch


in_file = open('0-1-0-1-0.adjlist', 'r')
adjlist12 = [line.strip() for line in in_file]
adjlist12 = adjlist12
in_file.close()

in_file = open('0-1-0-1-0_idx.pickle', 'rb')
idx10 = pickle.load(in_file)
in_file.close()
num = 0
offset = 118
l = []
edges = []

result_indices = []
for row in adjlist12:
    row_parsed = list(map(int, row.split(' ')))
    # print('start')
    # print(len(row_parsed[1:]))

    indices = idx10[row_parsed[0]]
    if len(row_parsed) > 1:
        neighbors = np.array(row_parsed[1:])
        edge_indices = indices

        edge_indices = np.stack((edge_indices[:, 0], edge_indices[:, 5 // 2], edge_indices[:, -1]), axis=1)
        edge_indices, idx = np.unique(edge_indices, return_index=True, axis=0)
        neighbors = neighbors[idx]

        unique, counts = np.unique(neighbors, return_counts=True)
        p = []
        for count in counts:
            p += [(count ** (3 / 4)) / count] * count
        p = np.array(p)
        p = p / p.sum()
        # print(p.sum())
        samples = min(100, len(neighbors))
        # print('sample', samples)
        sampled_idx = np.sort(np.random.choice(len(neighbors), samples, replace=False, p=p))
        neighbors = neighbors[sampled_idx]
        edge_indices = edge_indices[sampled_idx]

        result_indices.append(edge_indices)

    else:
        neighbors = [row_parsed[0]]

        edge_indices = np.array([[row_parsed[0]] * (5 - 2)])

        result_indices.append(edge_indices)
    for dst in neighbors:
        edges.append([row_parsed[0], dst])

result_indices = np.vstack(result_indices)


edges, idx = np.unique(edges, return_index=True, axis=0)
result_indices = result_indices[idx]
edge = [tuple(e) for e in edges]
edges = edge
g = dgl.graph([],num_nodes=449)
sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
result_indices = torch.LongTensor(result_indices[sorted_index])
g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
torch.save(g,'g20_0_1.pkl')
torch.save(result_indices,'edge20_0_1.pkl')

