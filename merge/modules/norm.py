import torch
import torch.nn as nn
from jaxtyping import Float

from merge.modules.config import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        eps: Float = 1e-5,
    ):
        """
        Root Mean Square Normalization Layer.
        The mean accumulation is done in 32-bit precision, using the fast inverse square root.

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension to normalize over (Layer dimension)
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.

        References:
            - "Root Mean Square Layer Normalization" (https://arxiv.org/pdf/1910.07467)
        """
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(config.d_model))

    def _irms(self, x):
        return torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        irms = self._irms(x.float()).type_as(x)
        return self.gamma * x * irms


class LayerNorm(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        eps: Float = 1e-5,
    ):
        """
        Layer Normalization.
        The mean and variance accumulation is done in 32-bit precision.

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension to normalize over (Layer dimension)
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.

        References:
            - "Layer Normalization" (https://arxiv.org/abs/1607.06450)
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))

    def _norm_scaling(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x):
        normalized = self._norm_scaling(x.float()).type_as(x)
        return self.gamma * normalized + self.beta


NORM_MAP = {
    "rmsnorm": RMSNorm,
    "layernorm": LayerNorm,
    "none": nn.Identity,
}
