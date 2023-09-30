import torch
from torch import nn


class DiffAttention(nn.Module):
    """扩散能量抑制的注意力"""

    def __init__(self, dropout):
        super(DiffAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # print('*'*50)
        # print('keys',keys.shape)
        d = torch.einsum("nhm,lhm->nlh", queries, keys)
        # print('d',d.shape)
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", queries, keys))
        all_ones = torch.ones([keys.shape[0]]).to(keys.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        # print('atten_norm1',attention_normalizer.shape)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, keys.shape[0], 1)
        # print('atten_norm2', attention_normalizer.shape)
        attention = attention_num / attention_normalizer
        # print('atten',attention.shape)

        attention = self.dropout(attention)
        attn_output = torch.einsum("nlh,lhd->nhd", attention, values).mean(1)
        # print('attn_output.shape',attn_output.shape)
        # print('*' * 50)
        return attn_output


class MutilAttention(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 dropout,
                 ):
        super(MutilAttention, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)

        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.dfatten = DiffAttention(dropout)

    def forward(self, query_input):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(query_input).reshape(-1, self.num_heads, self.out_channels)

        value = self.Wv(query_input).reshape(-1, self.num_heads, self.out_channels)

        attention_output = self.dfatten(query, key, value)

        return attention_output  # ,value1


if __name__ == '__main__':
    net = MutilAttention(20, 30, 8, 'sigmoid')
    x = torch.randn(2, 20)
    logits = net(x, x)
    print(logits.shape)
