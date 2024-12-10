import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Complex
from torch import Tensor


class Block(nn.Module):
    def __init__(self, attention: nn.Module, mlp: nn.Module, norm1: nn.Module, norm2: nn.Module):
        super().__init__()

        self.norm1 = norm1
        self.attn = attention
        self.norm2 = norm2
        self.mlp = mlp

    def forward(self, x: Float[Tensor, "batch seq d_model"], mask: Float[Tensor, "1 1 seq seq"], freq_cis: Complex[Tensor, "seq half_head_dim"]):
        
        # Self-attention block
        attn_out = self.attn(self.norm1(x), mask, freq_cis)
        x = x + attn_out # add to residual stream

        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out # add to residual stream
        
        return x
