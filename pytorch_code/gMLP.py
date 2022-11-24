#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on June, 2022

@author: JunlongChi
"""


from torch import nn


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
