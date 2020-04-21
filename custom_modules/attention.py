import torch
from torch import nn

import torch.nn.functional as F


class MultiHeadCondAttention(nn.Module):
    """ Multi-Head Conditioned Self Attention module """

    def __init__(self, n_head, input_feature_dim, out_dim, dim_k, dim_v, dropout=0.1):
        super(MultiHeadCondAttention, self).__init__()

        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.w_qs = nn.Linear(input_feature_dim, n_head * dim_k, bias=False)
        self.w_ks = nn.Linear(input_feature_dim, n_head * dim_k, bias=False)
        self.w_vs = nn.Linear(input_feature_dim, n_head * dim_v, bias=False)
        self.fc = nn.Linear(n_head * dim_v, out_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_feature_dim, eps=1e-6)

    def forward(self, inp, actions, mask=None):
        b, seq_length = inp.size(0), inp.size(1)

        residual = inp

        # Same action on all nodes
        action_vec = actions.repeat(1, seq_length)
        action_vec = action_vec.view(b, seq_length, actions.size(1))

        inp = torch.cat([inp, action_vec], dim=-1)

        inp = self.layer_norm(inp)

        # Pass through the pre-attention projection: b x len_q x (n_head*dv)
        # Separate different heads: [b, seq_length, n_head, dim_v]
        q = self.w_qs(inp).view(b, seq_length, self.n_head, self.dim_k)
        k = self.w_ks(inp).view(b, seq_length, self.n_head, self.dim_k)
        v = self.w_vs(inp).view(b, seq_length, self.n_head, self.dim_v)

        # Transpose for attention dot product: b x n_head x lq x dim_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        scores, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        scores = scores.transpose(1, 2).contiguous().view(b, seq_length, -1)
        scores = self.dropout(self.fc(scores))
        scores += residual

        return scores


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
