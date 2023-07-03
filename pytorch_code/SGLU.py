from torch import nn
from torch.nn import functional as F


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        return self.spatial_proj(self.norm(x))


class SGLULayer(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super(SGLULayer, self).__init__()
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model, eps=1e-8)
        self.in_dense1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU()
        )
        self.in_dense2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU()
        )
        self.SGU = SpatialGatingUnit(d_model * 4, seq_len)
        self.out_dense = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        u = self.in_dense1(x)
        F.dropout(u, self.dropout, inplace=True)
        v = self.in_dense2(x)
        F.dropout(v, self.dropout, inplace=True)
        z = self.SGU(v)
        out = u * z
        out = self.out_dense(out)
        return out + residual


class SGUL(nn.Module):
    def __init__(self, d_model, seq_len, num_layers=2, dropout=0.):
        super(SGUL, self).__init__()
        self.net = nn.Sequential(
            *[SGLULayer(d_model, seq_len, dropout) for _ in range(num_layers)]
        )
        self.last_norm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, x):
        x = self.net(x)
        return self.last_norm(x)
