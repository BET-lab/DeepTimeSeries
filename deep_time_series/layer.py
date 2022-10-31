import math
import torch
import torch.nn as nn


class Unsqueesze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class LeftPadding1D(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        # x: (B, C, L).
        B = x.size(0)
        C = x.size(1)

        padding = torch.zeros(B, C, self.padding_size).to(x.device)

        y = torch.cat([padding, x], axis=2)

        return y


# Modified from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
# Modifications
# 1. Dropout functionality is removed.
# 2. Changed to batch_first format.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) \
            * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Change to batch_first format.
        pe = pe.permute((1, 0, 2))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, F).
        x = x + self.pe[:, :x.size(1), :]
        return x