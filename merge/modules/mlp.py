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
        """
        MLP - Transformer specific with 2 linear transformations.
        Args:
            dim (int): Input dimension
            hidden_dim (int, optional): Hidden dimension. If None, computed as 4 * (2/3 * hidden_dim)
            activation_fn (str): Activation function name. Defaults to "relu"
            dropout (float, optional): Dropout probability. Defaults to None
            bias (bool): Whether to use bias. Defaults to True
            dim (int): The input dimension.
        """

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
        """
        SwiGLU Feed Forward Network - Transformer specific (2 layers, 1 of which is gated).
        Args:
            dim (int): The input dimension.
            hidden_dim (int, optional): Hidden dimension. If None, computed as 4 * (2/3 * hidden_dim)
            dropout (float, optional): Dropout probability. Defaults to None
            bias (bool): Whether to use bias. Defaults to False
        References:
            - "Gated Linear Units" (https://arxiv.org/pdf/2002.05202v1)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * int(2 * hidden_dim / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.up(x) * F.silu(self.gate(x))  # SwiGLU
        x = self.down(x)
        x = self.dropout(x)
        return x


MLP_MAP = {
    "mlp": MLP,
    "mlp_swiglu": MLPSwiGLU,
}
