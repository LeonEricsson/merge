from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

head_dim = 64
freq_cis = torch.pow(10000, -torch.arange(0, head_dim, 2) / head_dim)


def precompute_freqs_cis(dim: Int, end: Int, theta: Float = 10000.0) -> Tensor:

    pass


def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    # Applying the rotation
    pass
