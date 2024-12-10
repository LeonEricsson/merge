import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from merge.modules.attention import ATTN_MAP
from merge.modules.config import TransformerConfig
from merge.modules.mlp import MLP_MAP
from merge.modules.norm import NORM_MAP


class Block(nn.Module):
    def __init__(self, attention: nn.Module, mlp: nn.Module, norm: nn.Module):
        super().__init__()

        self.norm1 = norm
        self.attn = attention
        self.norm2 = norm
        self.mlp = mlp
