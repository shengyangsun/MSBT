#Some codes are grabbed from https://github.com/JustinYuu/MACIL_SD/blob/main/Transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa


class CrossAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(CrossAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, m_a, m_b):
        m_a_cma = self.layer(m_a, m_b, m_b)
        m_b_cma = self.layer(m_b, m_a, m_a)
        return m_a_cma, m_b_cma


class MultilayerTransformer(nn.Module):
    def __init__(self, Transformer_layer, n_layers):
        super(MultilayerTransformer, self).__init__()
        self.layer = clones(Transformer_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, feat):
        for layer_i in range(self.n_layers):
            feat = self.layer[layer_i](feat, feat, feat)
        return feat


class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = nn.LayerNorm(size)

    def forward(self, q, k, v):
        q, k, v = self.norm(q), self.norm(k), self.norm(v)
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        return self.sublayer[1](q, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


def attention(query, key, value, masksize, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if masksize != 1:
        masksize = int(masksize / 2)
        mask = torch.ones(scores.size()).cuda()
        for i in range(mask.shape[2]):
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, self.masksize, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output