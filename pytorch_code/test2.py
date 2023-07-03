import torch

from san import SelfAttention, PositionEmbedding

batch_size = 100
seq_len = 70
dim = 100
# x = torch.randn(batch_size, dim)
# adj = torch.rand(batch_size, batch_size)
# zeros = torch.zeros_like(adj)
# ones = torch.ones_like(adj)
# adj = torch.where(adj >= 0.5, ones, zeros)
# gat = GATv2Conv(in_channels=dim, out_channels=dim, concat=False, heads=3)
# edge_index, _ = dense_to_sparse(adj)
# a = gat(x, edge_index)
# print("a: ", a.shape)
x = torch.randn(seq_len, batch_size, dim)
x = x.transpose(0, 1)
print(x.shape)
positionEmbedding = PositionEmbedding(seq_len, dim)
x = positionEmbedding(x)
x = x.transpose(0, 1)
print(x.shape)
SAN = SelfAttention(num_layers=3, d_model=dim, nhead=1, dim_ff=dim * 4, dropout=0.5)
out = SAN(x)
