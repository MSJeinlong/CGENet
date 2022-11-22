import torch
import torch.nn.functional as F
from torch import nn


class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings  # the length of sequence data
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )


class SelfAttention(nn.Module):

    def __init__(self, num_layers, d_model, nhead, dim_ff, dropout):
        super(SelfAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, "gelu", 1e-8, True)
        self.trn_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, src, attn_mask=None):
        seq_len = src.shape[1]
        if attn_mask is None:
            attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=src.device))
        return self.trn_encoder(src, attn_mask)
