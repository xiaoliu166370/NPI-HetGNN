from .MPA import MPA
from utills.utils import *
from .engyge_constrained_attention import MutilAttention


class Model(nn.Module):
    def __init__(self, feat_dims, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout, device, meta_paths):
        super().__init__()

        self.trans_list = nn.ModuleList([nn.Linear(feat_dim, in_dim, bias=True) for feat_dim in feat_dims])
        self.map1 = MPA(meta_paths=meta_paths[0],
                        in_size=in_dim,
                        hidden_size=hidden_dim,
                        out_size=out_dim,
                        num_heads=[num_heads],
                        dropout=dropout
                        )
        self.map2 = MPA(meta_paths=meta_paths[1],
                        in_size=in_dim,
                        hidden_size=hidden_dim,
                        out_size=out_dim,
                        num_heads=[num_heads],
                        dropout=dropout
                        )

        self.act = nn.ELU()
        self.fc_list = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(len(feat_dims))])
        self.pred = HeteroDotProductPredictor()
        self.device = device
        if num_layers == 1:
            self.sq1 = MutilAttention(in_dim, out_dim, num_heads, dropout)
            self.sq2 = MutilAttention(in_dim, out_dim, num_heads, dropout)
        else:
            self.sq1 = nn.Sequential()
            self.sq2 = nn.Sequential()
            self.sq1.add_module(f'layer{1}', MutilAttention(in_dim, hidden_dim, num_heads, dropout))
            self.sq2.add_module(f'layer{1}', MutilAttention(in_dim, hidden_dim, num_heads, dropout))
            for i in range(1, num_layers):
                if i + 1 == num_layers:
                    self.sq1.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, out_dim, num_heads, dropout))
                    self.sq2.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, out_dim, num_heads, dropout))
                else:
                    self.sq1.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, hidden_dim, num_heads, dropout))
                    self.sq2.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, hidden_dim, num_heads, dropout))

    def forward(self, g_list, pos_g, neg_g, x, etype, edge_idx_list):
        pos_g = pos_g.to(self.device)
        neg_g = neg_g.to(self.device)
        x1 = x['protein'].to(self.device)
        x2 = x['rna'].to(self.device)
        g_p = g_list[0]
        g_r = g_list[1]
        g_p1 = []
        g_r1 = []
        for g in g_p:
            g = g.to(self.device)
            g_p1.append(g)
        for g in g_r:
            g = g.to(self.device)
            g_r1.append(g)
        neg_g = neg_g.to(self.device)

        # nodes transform
        x1 = self.trans_list[0](x1)
        x1 = self.act(x1)
        x2 = self.trans_list[1](x2)
        x2 = self.act(x2)

        # mpa
        edge_idx0 = edge_idx_list[0].to(self.device)
        edge_idx1 = edge_idx_list[1].to(self.device)
        h1 = self.map1(g_p1, x1, edge_idx0)
        h2 = self.map2(g_r1, x2, edge_idx1)

        # add attention
        a1 = self.sq1(x1)
        a2 = self.sq2(x2)

        # residual connection
        emb1 = x1 + h1 + a1
        emb2 = x2 + h2 + a2

        # linear projection
        emb1 = self.fc_list[0](emb1)
        emb2 = self.fc_list[1](emb2)

        h = {'protein': emb1, 'rna': emb2}

        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype), h
