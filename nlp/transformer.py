# references:
# https://arxiv.org/abs/1706.03762
# https://peterbloem.nl/blog/transformers
# https://github.com/karpathy/minGPT
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        assert d_model % 2 == 0
        super().__init__()
        pos = torch.arange(max_len).float()
        i = torch.arange(d_model // 2)
        den = 10_000 ** (2 * i / d_model)
        p_i = pos.unsqueeze(1) / den
        enc = torch.empty(max_len, d_model)
        enc[:, 0::2] = torch.sin(p_i)
        enc[:, 1::2] = torch.cos(p_i)
        self.register_buffer('enc', enc, persistent=False)

    def forward(self, x):
        return self.enc[:x.size(-2)]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.seq(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, max_len, d_model, n_heads, causal=False, dropout=0.1):
        assert d_model % n_heads == 0
        super().__init__()
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        causal_mask = None
        if causal:
            causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, x, pad_mask=None):
        b, n, d = x.size()
        h = self.n_heads
        s = d // h

        mask = self._merge_masks(self.causal_mask, pad_mask, n)  # bnn

        q, k, v = self.qkv_proj(x).split(d, dim=2)  # bnd
        q, k, v = [z.view(b, n, h, s).transpose(1, 2) for z in (q, k, v)]  # bhns

        # pre-scale q for memory efficiency
        q = q / math.sqrt(s)

        a = q @ k.transpose(2, 3)  # bhnn
        if mask is not None:
            mask = mask.unsqueeze(1)  # b1nn
            a = a.masked_fill(mask, float('-inf'))
        a = a.softmax(dim=3)  # bhnn
        a = self.dropout(a)

        y = a @ v  # bhns
        y = y.transpose(1, 2).reshape(b, n, d)  # bnd
        y = self.out_proj(y)  # bnd

        return y

    @staticmethod
    def _merge_masks(causal_mask, pad_mask, seq_len):
        # causal_mask: NN
        # pad_mask: bn
        if causal_mask is None and pad_mask is None:
            return None
        if causal_mask is not None:
            causal_mask = causal_mask[:seq_len, :seq_len].unsqueeze(0)  # 1nn
            if pad_mask is None:
                return causal_mask
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1)  # b1n
            if causal_mask is None:
                return pad_mask
        return causal_mask | pad_mask


class TransformerBlock(nn.Module):
    def __init__(self, max_len, d_model, n_heads, d_ff, causal=False, dropout=0.1, attn_drop=True, ff_drop=True):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(max_len, d_model, n_heads, causal=causal, dropout=(dropout * attn_drop))
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=(dropout * ff_drop))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        skip = x
        x = self.mhsa(x, pad_mask=pad_mask)
        x = self.dropout1(x)
        x = x + skip
        x = self.norm1(x)

        skip = x
        x = self.ff(x)
        x = self.dropout2(x)
        x = x + skip
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, max_len=128, d_model=128, n_blocks=2, n_heads=2, d_ff=256, causal=False, dropout=0.1,
            attn_drop=True, ff_drop=True):
        super().__init__()
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        kwargs = dict(causal=causal, dropout=dropout, attn_drop=attn_drop, ff_drop=ff_drop)
        self.blocks = nn.ModuleList([TransformerBlock(max_len, d_model, n_heads, d_ff, **kwargs)
            for _ in range(n_blocks)])

    def forward(self, x, pad_mask=None):
        x = x + self.pos_enc(x).unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        return x


class TextTransformer(nn.Module):
    def __init__(self, n_vocab, max_len, d_model, pad_index=None, *xf_args, **xf_kwargs):
        super().__init__()
        self.pad_index = pad_index
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.transformer = Transformer(max_len, d_model, *xf_args, **xf_kwargs)
        self.linear = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        pad_mask = (x == self.pad_index) if self.pad_index is not None else None
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.transformer(x, pad_mask=pad_mask)
        y = self.linear(x)
        return y

    def predict_probas(self, x):
        scores = self(x)
        probas = scores.softmax(2)
        return probas
