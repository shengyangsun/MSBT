import torch
from Transformer import *

class MultiScale_Bottleneck_Transformer(nn.Module):
    def __init__(self, hid_dim, n_head, dropout, n_bottleneck=8, bottleneck_std=0.15):
        super(MultiScale_Bottleneck_Transformer, self).__init__()
        self.n_layers = int(math.log2(n_bottleneck)) + 1
        self.sma = nn.ModuleList([
            TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim), PositionwiseFeedForward(hid_dim, hid_dim), dropout=dropout)
            for _ in range(self.n_layers)])
        self.decoder = TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim), PositionwiseFeedForward(hid_dim, hid_dim), dropout=dropout)
        self.bottleneck_list = nn.ParameterList([
            nn.Parameter(nn.init.normal_(torch.zeros(1, int(n_bottleneck / (2 ** layer_i)), hid_dim).cuda(), std=bottleneck_std))
            for layer_i in range(self.n_layers)])

    def forward(self, m_a, m_b):
        n_batch = m_a.shape[0]
        n_modality = m_a.shape[1]
        bottleneck = self.bottleneck_list[0]
        bottleneck = bottleneck.repeat([n_batch, 1, 1])
        m_a_in, m_b_in = m_a, m_b
        for layer_i in range(self.n_layers):
            m_a_cat = torch.cat([m_a_in, bottleneck], dim=1)
            m_a_cat = self.sma[layer_i](m_a_cat, m_a_cat, m_a_cat)
            m_a_in = m_a_cat[:, :n_modality, :]
            m_a_bottleneck = m_a_cat[:, n_modality:, :]
            if layer_i < self.n_layers - 1:
                next_bottleneck = self.bottleneck_list[layer_i + 1]
                next_bottleneck = next_bottleneck.repeat([n_batch, 1, 1])
                bottleneck = self.decoder(next_bottleneck, m_a_bottleneck, m_a_bottleneck)
            m_b_cat = torch.cat([m_b_in, m_a_bottleneck], dim=1)
            m_b_cat = self.sma[layer_i](m_b_cat, m_b_cat, m_b_cat)
            m_b_in = m_b_cat[:, :n_modality, :]

        return m_b_in, m_a_bottleneck