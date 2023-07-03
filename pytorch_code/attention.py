from math import sqrt

import torch
from torch import nn


class PositionAttention(nn.Module):
    def __init__(self, d_model, d_attn=128):
        super().__init__()
        self.qk_project = nn.Linear(d_model, d_attn * 2)
        # self.pos_scale = scale or 1. / sqrt(2 * d_attn)

    def forward(self, x):
        qk = self.qk_project(x)
        q, k = qk.chunk(2, dim=-1)
        # q dot k
        pos_attn = torch.einsum("bnd,bmd->nm", q, k)
        pos_attn = pos_attn
        return pos_attn


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_attn=128, dropout=0.1, mask_flag=True):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.qkv_project = nn.Linear(d_model, d_attn * 3)
        self.out_project = nn.Linear(d_attn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1. / sqrt(d_attn)
        self.mask_flag = mask_flag

    def forward(self, x, attn_bias=None, attn_mask=None, key_mask=None):
        seq_len = x.shape[1]
        qkv = self.qkv_project(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_weights = torch.einsum("bnd, bmd -> bnm", q, k)
        attn_weights = attn_weights

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device))
                attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        if attn_bias is not None:
            attn_weights += attn_bias
            self.scale = 1. / sqrt(2 * self.d_attn)
        attn_weights = attn_weights * self.scale

        attn = torch.softmax(attn_weights, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bnm, bmd -> bnd", attn, v)
        out = out.contiguous()
        out = out * key_mask
        out = self.dropout(self.out_project(out))
        return out


# 前馈网络
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, pre_ln=False):

        super(PointWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.coreNet = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),
            nn.Dropout(p=dropout)
        )

        # 归一化层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        # pre_ln 决定是否先执行归一化
        self.pre_lnorm = pre_ln

    def forward(self, inputs):

        # 如果pre_lnorm=True，则先对inputs执行归一化，再输入到FFN（即原始SASRec的做法）
        if self.pre_lnorm:
            lnorm_out = self.layer_norm(inputs)
            outputs = self.coreNet(lnorm_out.transpose(-1, -2))
            # as Conv1D requires (N, C, Length)
            outputs = outputs.transpose(-1, -2)
            # 残差连接
            outputs = inputs + outputs
        else:
            # 修改后的SASRec（与原始SASRec稍有不同）
            # 先输入到FFN
            outputs = self.coreNet(inputs.transpose(-1, -2))
            outputs = outputs.transpose(-1, -2)
            # 再残差连接 + 层归一化
            outputs = self.layer_norm(outputs + inputs)

        return outputs


class TinyformerEncoder(nn.Module):
    def __init__(self, num_layer, d_model, d_attn, d_ffn, dropout_rate, mask_flag=True, pre_ln=True):
        super(TinyformerEncoder, self).__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.d_ffn = d_ffn
        self.num_layer = num_layer
        self.pre_ln = pre_ln

        self.attention = nn.ModuleList()
        self.attn_ln = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.pos_attention = PositionAttention(d_model, d_attn=d_attn)
        for i in range(num_layer):
            # attention layer
            new_attention = SelfAttention(d_model, d_attn=d_attn, dropout=dropout_rate, mask_flag=mask_flag)
            self.attention.append(new_attention)
            new_ln = nn.LayerNorm(d_model, eps=1e-6)
            self.attn_ln.append(new_ln)

            # ffn layer
            new_ffn = PointWiseFeedForward(d_model=d_ffn, dropout=dropout_rate, pre_ln=pre_ln)
            self.ffn.append(new_ffn)

        # last layer_norm
        self.last_ln = nn.LayerNorm(d_model, eps=1e-6) if pre_ln else None

    def forward(self, x, pos_emb, attn_mask=None, key_mask=None):
        pos_attn = self.pos_attention(pos_emb)
        for i in range(self.num_layer):
            if self.pre_ln:
                x = self.attn_ln[i](x)
            residual = x
            x = self.attention[i](x, attn_mask=attn_mask, attn_bias=pos_attn, key_mask=key_mask)
            if not self.pre_ln:
                x = self.attn_ln[i](x + residual)
            else:
                x += residual
            x = self.ffn[i](x)
        if self.last_ln is not None:
            x = self.last_ln(x)
        return x
