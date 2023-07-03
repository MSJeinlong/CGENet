import torch
from torch import nn

# adj_in = torch.randn(100, 50, 50)
# adj_out = torch.randn(100, 50, 50)
# x = torch.randn(100, 50, 100)
# y = torch.matmul(adj_in, x) + torch.matmul(adj_out, x)
# print("y shape: ", y.shape)
# x = torch.randn(694, 1)
# weight = torch.randn(1, 100)
# out = torch.matmul(x, weight)
# print(out.shape)

# target output size of 5
m = nn.AdaptiveMaxPool1d(2)
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
print(x1)
print(x2)
print(torch.max(x1, x2))
