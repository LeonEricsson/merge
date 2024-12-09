from typing import Tuple, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Complex
from torch import Tensor

MHATensor: TypeAlias = Float[Tensor, 'batch seq heads head_dim']

def precompute_freqs_cis(head_dim: Int, max_seq_length: Int, theta: Float = 10000.0) -> Tensor:
    base = torch.pow(theta, -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    positions = torch.arange(0, max_seq_length)
    rotations = torch.outer(positions, base)
    freqs_cis = torch.polar(torch.ones_like(rotations), rotations)
    return freqs_cis


def apply_rotary_emb(
        q: MHATensor, 
        k: MHATensor, 
        freqs_cis: Complex[Tensor, 'seq dim_half']
    ) -> Tuple[MHATensor, MHATensor]:
    batch, seq_length, n_heads, head_dim = q.size()
    
    complex_q = torch.complex(q[:, :, :, ::2], q[:, :, :, 1::2])
    complex_k = torch.complex(k[:, :, :, ::2], k[:, :, :, 1::2])

    rotated_q = complex_q * freqs_cis[None, None, :, :] # wrong shapes q is seq heads not heads seq so doesnt match freqs cis
    rotated_k = complex_k * freqs_cis[None, None, :, :]

    rotated_q = torch.stack((rotated_q.real, rotated_q.imag), dim=4).view(batch, seq_length, n_heads, head_dim)
    rotated_k = torch.stack((rotated_k.real, rotated_q.imag), dim=4).view(batch, seq_length, n_heads, head_dim)
    
    return rotated_q, rotated_k
