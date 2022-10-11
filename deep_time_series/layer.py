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