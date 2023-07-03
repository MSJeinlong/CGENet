from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class PositionAttention(nn.Module):
    def __init__(self, d_model, d_attn=64):
        super().__init__()
        self.qk_project = nn.Linear(d_model, d_attn * 2)
        self.pos_scale = 1. / sqrt(2 * d_attn)

    def forward(self, x):
        qk = self.qk_project(x)
        q, k = qk.chunk(2, dim=-1)
        # q dot k
        pos_attn = torch.einsum("bnd,bmd->nm", q, k)
        pos_attn = pos_attn * self.pos_scale
        return pos_attn


class TinyAttention(nn.Module):
    def __init__(self, d_model, d_attn=64, dropout=0.1, mask_flag=True):
        super(TinyAttention, self).__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.qkv_project = nn.Linear(d_model, d_attn * 3)
        self.out_project = nn.Linear(d_attn, d_model // 2)
        self.dropout = dropout
        self.scale = 1. / sqrt(d_attn)
        self.mask_flag = mask_flag

    def forward(self, x):
        seq_len = x.shape[1]
        qkv = self.qkv_project(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_weights = torch.einsum("bnd, bmd -> bnm", q, k)
        attn_weights = attn_weights * self.scale

        if self.mask_flag:
            attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device))
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        attn = torch.softmax(attn_weights, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("bnm, bmd -> bnd", attn, v)
        out = out.contiguous()
        out = self.out_project(out)
        return out


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, norm_eps=1e-8):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn, eps=norm_eps)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class SGUWithAttention(nn.Module):
    def __init__(self, d_ffn, d_attn, seq_len, dropout, layer_norm_eps=1e-8):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn, eps=layer_norm_eps)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        self.tiny_attn = TinyAttention(d_model=d_ffn * 2, d_attn=d_attn, dropout=dropout)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        y = self.tiny_attn(x)
        out = u * (v + y)
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, dropout, norm_eps=1e-8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, d_ffn * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            SpatialGatingUnit(d_ffn, seq_len, norm_eps=norm_eps),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6, dropout=0.2, norm_eps=1e-8):
        super().__init__()
        self.encoder = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, dropout, norm_eps=norm_eps) for _ in range(num_layers)]
        )
        self.last_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        x = self.encoder(x)
        return self.last_norm(x)


class aMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, d_attn, seq_len, dropout, layer_norm_eps=1e-8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, d_ffn * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            SGUWithAttention(d_ffn, d_attn, seq_len, dropout, layer_norm_eps=layer_norm_eps),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x


class aMLP(nn.Module):

    def __init__(self, d_model=256, d_ffn=512, d_attn=64, seq_len=256, num_layers=6, dropout=0.1, layer_norm_eps=1e-8):
        super(aMLP, self).__init__()
        self.encoder = nn.Sequential(
            *[aMLPBlock(d_model, d_ffn, d_attn, seq_len, dropout, layer_norm_eps=layer_norm_eps)
              for _ in range(num_layers)]
        )
        self.last_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x):
        x = self.encoder(x)
        return self.last_norm(x)
