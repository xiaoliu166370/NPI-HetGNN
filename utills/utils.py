import random
import torch
import dgl
import dgl.function as fn
import numpy as np
import torch.nn as nn
import pickle
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def split_train_test(data, ratio):
    n_samples = data.shape[0]
    n_test = int(n_samples * ratio)
    ridx = np.random.choice(n_samples, n_test, replace=False)

    train = np.delete(data, ridx, axis=0)

    n_train, p_train = np.unique(train[:, 0]), np.unique(train[:, 1])

    f_test_idx = []
    for i in ridx:
        line = data[i]
        if line[0] in n_train and line[1] in p_train:
            f_test_idx.append(i)
    f_test = data[f_test_idx]
    f_train = np.delete(data, f_test_idx, axis=0)

    return f_train, f_test









if __name__ == '__main__':
    f = open('20_r_p.txt', 'r')
    l = []
    data = f.readlines()
    for d in data:
        d = d.strip()
        d = d.split(' ')
        d = [eval(dd) for dd in d]
        l.append(d)
    print(len(l))
    l = np.array(l)
    n = l[:, 0]
    p = l[:, 1]
    train, test = split_train_test(np.array(l), 0.8)
    print(train)
    print(test.shape)
