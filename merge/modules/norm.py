import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps: float = 1e-5,
    ):
        """
        Root Mean Square Normalization Layer.
        The mean accumulation is done in 32-bit precision, using the fast inverse square root.
        Args:
            dim (int): Input dimension to normalize over (Layer dimension)
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        References:
            - "Root Mean Square Layer Normalization" (https://arxiv.org/pdf/1910.07467)
        """
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _irms(self, x):
        return torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        irms = self._irms(x.float()).type_as(x)
        return self.gamma * x * irms


NORM_MAP = {
    "rmsnorm": RMSNorm,
    "layernorm": nn.LayerNorm,
    "none": nn.Identity,
}
