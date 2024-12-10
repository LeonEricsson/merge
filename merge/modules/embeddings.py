from typing import Tuple
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

MHATensor: TypeAlias = Float[Tensor, "batch heads seq head_dim"]


def precompute_freqs_cis(
    head_dim: Int, max_seq_length: Int, theta: Float = 10000.0
) -> Tensor:
    """
    For each position in the sequence and each pair of dimensions in the head, computes
    a rotation in the complex plane of the form cos(m*θᵢ) + i*sin(m*θᵢ), where:
    - m is the position in the sequence
    - θᵢ = 1/θ^(2i/d) is the base frequency for the i-th dimension pair

    Args:
        head_dim: Size of attention head dimension (must be even)
        max_seq_length: Maximum sequence length to precompute rotations for
        theta: Base frequency scaling factor (default: 10000.0)
    """
    assert head_dim % 2 == 0

    freqs = 1.0 / torch.pow(
        theta, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    print(freqs)
    positions = torch.arange(0, max_seq_length)
    rotations = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(rotations), rotations)
    return freqs_cis


def apply_rope(
    q: MHATensor, k: MHATensor, freqs_cis: Complex[Tensor, "seq dim_half"]
) -> Tuple[MHATensor, MHATensor]:
    """
    Applies rotary position embeddings to query and key tensors in float32.

    Rotates pairs of dimensions (x,y) in q/k by position-dependent angles. For each pair:
    1. Convert to complex number: z = x + iy
    2. Multiply by rotation e^(im*θ) = cos(m*θ) + i*sin(m*θ)
    3. Convert result back to real pairs
    """
    batch, n_heads, seq_length, head_dim = q.size()
    q_f = q.float()
    k_f = k.float()
    complex_q = torch.complex(q_f[:, :, :, ::2], q_f[:, :, :, 1::2])
    complex_k = torch.complex(k_f[:, :, :, ::2], k_f[:, :, :, 1::2])

    rotated_q = complex_q * freqs_cis[None, None, :, :]
    rotated_k = complex_k * freqs_cis[None, None, :, :]

    # interleave real/imaginary parts and reshape to original shape
    real_q = torch.stack((rotated_q.real, rotated_q.imag), dim=4).view(
        batch, n_heads, seq_length, head_dim
    )
    real_k = torch.stack((rotated_k.real, rotated_k.imag), dim=4).view(
        batch, n_heads, seq_length, head_dim
    )

    return real_q.type_as(q), real_k.type_as(k)
