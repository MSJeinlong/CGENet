#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
import copy
import datetime
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from gMLP import gMLP
from san import SelfAttention, PositionEmbedding


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class DenseGINConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GINConv`.

    :rtype: :class:`Tensor`
    """

    def __init__(self, nn, eps=0, train_eps=False):
        super(DenseGINConv, self).__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        adj_in = adj[:, :, :N]
        adj_out = adj[:, :, N:]

        out = torch.matmul(adj_in, x) + torch.matmul(adj_out, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class HighwayGate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HighwayGate, self).__init__()
        self.W = nn.Linear(2 * in_dim, out_dim, bias=False)

    def forward(self, h0, hl):
        g = torch.sigmoid(self.W(torch.cat([h0, hl], dim=-1)))
        out = g * h0 + (1 - g) * hl
        return out


class MLP4GIN(nn.Module):
    def __init__(self, in_dim, out_dim, norm_eps=1e-8):
        super(MLP4GIN, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim, eps=norm_eps)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.linear2(x)
        return x


class BasicGNN(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim, act=nn.ReLU(inplace=True), dropout=0.0, norm=None):
        super(BasicGNN, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.gnn = nn.ModuleList()

        self.norms = None
        if norm is not None:
            self.norms = nn.ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

    def forward(self, x, adj, mask):
        for i in range(self.num_layers):
            x = self.gnn[i](x, adj, mask)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GIN(BasicGNN):
    def __init__(self, num_layers, in_dim, out_dim, act=nn.ReLU(inplace=True), dropout=0.0, norm=None, norm_eps=1e-8):
        super(GIN, self).__init__(num_layers, in_dim, out_dim, act, dropout, norm)

        for _ in range(num_layers):
            self.gnn.append(DenseGINConv(MLP4GIN(in_dim, out_dim, norm_eps=norm_eps)))


class GateUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GateUnit, self).__init__()
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W_2 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x1, x2):
        g = torch.sigmoid(self.W_1(x1) + self.W_2(x2))
        out = g * x1 + (1 - g) * x2
        return out


class SessionGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0., k=1):
        super(SessionGAT, self).__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                             dropout=dropout)
        self.k = k

    def forward(self, x):
        B, D = x.shape
        scale = 1.0 / math.sqrt(D)
        attn = torch.einsum("bd, sd -> bs", x, x)
        attn = torch.softmax(attn * scale, dim=1)
        # Select K neighbor session nodes with the smallest cosine distance for each session node,
        # that is, these neighbor nodes have edge connections with the current node
        _, indices = torch.topk(attn, dim=1, k=self.k)
        adj = torch.zeros_like(attn)
        adj.scatter_(1, indices.long(), 1)
        edge_index, _ = dense_to_sparse(adj)
        out = self.gat(x, edge_index)
        return out


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.dropout = opt.dropout
        self.nonhybrid = opt.nonhybrid
        self.no_hn = opt.no_hn
        self.no_gmlp = opt.no_gmlp
        self.no_sca = opt.no_sca
        self.use_san = opt.use_san
        self.aggregation = opt.aggregation
        if self.use_san:
            self.pos_emb = PositionEmbedding(opt.max_len, opt.hiddenSize)
            self.SAN = SelfAttention(num_layers=opt.gmlp_layers, d_model=self.hidden_size, nhead=1,
                                     dim_ff=self.hidden_size * 4, dropout=opt.dropout)
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.gnn = GNN(self.hidden_size, step=opt.step)
        self.gnn = GIN(num_layers=opt.gnn_layers, in_dim=self.hidden_size, out_dim=self.hidden_size,
                       dropout=opt.dropout)
        self.highway_gate = HighwayGate(in_dim=self.hidden_size, out_dim=self.hidden_size)
        self.gmlp = gMLP(d_model=self.hidden_size, d_ffn=self.hidden_size * 2, seq_len=opt.max_len,
                         num_layers=opt.gmlp_layers, dropout=opt.dropout, norm_eps=opt.layer_norm_eps)
        self.gate = GateUnit(self.hidden_size, self.hidden_size)
        self.sessionGAT = SessionGAT(self.hidden_size, self.hidden_size, heads=1, concat=True,
                                     dropout=opt.dropout, k=opt.k)
        self.s_ln = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.b_ln = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.aggr_gate = GateUnit(self.hidden_size, self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        s_local = torch.sum(hidden * mask, 1) / torch.sum(mask, 1)

        z = hidden
        if not self.no_gmlp and not self.use_san:  # use gmlp
            z = self.gmlp(z)
        elif self.use_san:
            z = self.SAN(self.pos_emb(z))

        s_global = torch.sum(mask * z, dim=1)
        s_final = self.gate(s_global, s_local)

        if not self.no_sca:  # use sca-gat
            # cross session aware attention
            s_cross = self.sessionGAT(s_global)
            if self.aggregation == "sum":
                s_final += s_cross
            elif self.aggregation == "max":
                s_final = torch.max(s_final, s_cross)
            elif self.aggregation == "concat":
                s_final = self.linear_transform(torch.cat([s_final, s_cross], dim=-1))
            elif self.aggregation == "gate":
                s_final = self.aggr_gate(s_final, s_cross)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        s_final = self.s_ln(s_final)
        b = self.b_ln(b)
        scores = torch.matmul(s_final, b.transpose(1, 0))
        return scores

    def forward(self, items, adj, mask):
        items_emb = self.embedding(items)
        h0 = F.dropout(items_emb, p=self.dropout, training=self.training)
        hl = self.gnn(h0, adj, mask)
        if not self.no_hn:
            hl = self.highway_gate(h0, hl)
        return hl


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()

    hidden = model(items, adj, mask)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    j = 0
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 1000 == 0:
            print('\tLoss:\t%.3f' % loss)
        j += 1
    print('\tTotal loss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    hit_10, mrr_10 = [], []
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores_10 = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        targets = targets.numpy()
        for score, score_10, target, mask in zip(sub_scores, sub_scores_10, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            hit_10.append(np.isin(target - 1, score_10))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
            if len(np.where(score_10 == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score_10 == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(mrr_10) * 100)
    return result
