from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor

from omni.modules.norm import LayerNorm
from omni.modules.pos_embeddings import apply_rope_real


class DifferentialAttention(nn.Module):
    def __init__(self, config, layer_idx):
        """
        Combined differential and grouped query attention

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - head_dim: Dimension of each attention head
            - num_heads: Number of attention heads
            - num_kv_heads: Number of key/value attention heads (must divide num_heads)
            - d_model: Model dimension
            - bias: Whether to use bias in linear layers
            - dropout: Dropout probability for attention and residual connections
        layer_idx: The layer index in the transformer ∈ [0, L]

        References:
            - "Differential Attention": https://arxiv.org/abs/2410.05258
        """
        super().__init__()
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        if self.n_heads is None:
            assert self.head_dim is not None
            self.n_heads = config.d_model // (2 * self.head_dim)
            self.n_kv_heads = self.n_heads

        if self.head_dim is None:
            self.head_dim = config.d_model // (2 * self.n_heads)

        assert config.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads * 2 * self.head_dim == config.d_model

        self.kv_groups = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        self.W_Q = nn.Linear(
            config.d_model,
            (self.head_dim * self.n_heads * 2),
            bias=config.attention_bias,
        )

        self.W_KV = nn.Linear(
            config.d_model,
            2 * (self.head_dim * self.n_kv_heads * 2),
            bias=config.attention_bias,
        )

        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self.group_norms = nn.ModuleList(
            [LayerNorm(dim=2 * self.head_dim) for _ in range(self.n_heads)]
        )

        self.register_buffer(
            "lambda_init", 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * layer_idx))
        )

        self.lambda_q1 = nn.Parameter(torch.ones(2 * self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.ones(2 * self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.ones(2 * self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.ones(2 * self.head_dim))

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.res_dropout = nn.Dropout(config.attention_dropout)

        self.flash_attn: bool = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        self.pos_encoding_type = config.pos_encoding_type

    def flash_attention(self, q, k, v, mask):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            attn_mask=mask,
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
        kv_cache,
    ):
        batch_size, seq_length, d_model = x.size()

        q = self.W_Q(x).unflatten(-1, (self.n_heads, 2 * self.head_dim))
        kv = self.W_KV(x).unflatten(-1, (2, self.n_kv_heads, 2 * self.head_dim))

        k, v = kv[:, :, 0], kv[:, :, 1]  # (batch, seq, n_kv_heads, 2*head_dim)

        q = q.transpose(1, 2)  # (batch, n_heads, seq, 2*head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.forward(k, v)

        k = torch.repeat_interleave(k, self.kv_groups, dim=1)
        v = torch.repeat_interleave(v, self.kv_groups, dim=1)

        q1, q2 = q.split(self.head_dim, dim=-1)
        k1, k2 = k.split(self.head_dim, dim=-1)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q1, k1 = apply_rope_real(q1, k1, freq_cis)
            q2, k2 = apply_rope_real(q2, k2, freq_cis)

        _lambda = (
            torch.exp(self.lambda_q1 * self.lambda_k1)
            - torch.exp(self.lambda_q2 * self.lambda_k2)
            + self.lambda_init
        )

        # to support single step inference
        start = k.shape[2] - q.shape[2]
        end = k.shape[2]
        mask = mask[:, :, start:end, : k.shape[2]]

        if self.flash_attn:
            A1 = self.flash_attention(q1, k1, v, mask)
            A2 = self.flash_attention(q2, k2, v, mask)
            output = A1 - _lambda * A2  # (batch, n_heads, seq, 2*head_dim)
        else:
            A1 = (q1 @ k1.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
            A2 = (q2 @ k2.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
            A = A1 - _lambda * A2
            A = self.attn_dropout(F.softmax(A + mask, dim=-1))
            output = A @ v  # (batch, n_heads, seq, 2*head_dim)

        output = torch.cat(
            [norm(output[:, i : i + 1]) for i, norm in enumerate(self.group_norms)],
            dim=1,
        )

        output = output * (1 - self.lambda_init)

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output
