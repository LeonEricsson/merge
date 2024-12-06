import torch
import torch.nn as nn
import torch.nn.functional as F

from merge.modules.activations import ACT2FN


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation_fn: str = "relu",
        dropout: float | None = None,
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * int(2 * hidden_dim / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation_fn = ACT2FN[activation_fn]
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.up(x)
        x = self.activation_fn(x)
        x = self.down(x)
        x = self.dropout(x)
        return x


class MLPSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float | None = None,
        bias: bool = False,
    ):

        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * int(2 * hidden_dim / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = F.silu(nn.Linear(dim, hidden_dim, bias=bias))
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.up(x) * self.gate(x)  # SwiGLU
        x = self.down(x)
        x = self.dropout(x)
        return x
