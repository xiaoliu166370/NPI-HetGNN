import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import GATConv
from .meta_agg import MetapathAgg


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class MPALayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(MPALayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_size,
                out_size,
                layer_num_heads,
                dropout,
                dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            ))
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                MetapathAgg(out_size, layer_num_heads)
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g_list, feat, edge_idx):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g_list:
            self._cached_graph = g_list
            self._cached_coalesced_graph.clear()
            for idx, meta_path in enumerate(self.meta_paths):
                self._cached_coalesced_graph[
                    meta_path
                ] = g_list[idx]

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]

            semantic_embeddings.append(self.gat_layers[i](new_g, feat, edge_idx).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class MPA(nn.Module):
    def __init__(
            self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(MPA, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            MPALayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                MPALayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g_list, feat, edge_idx):
        for gnn in self.layers:
            feat = gnn(g_list, feat, edge_idx)

        return self.predict(feat)
