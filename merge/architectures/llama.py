from dataclasses import dataclass
from dataclasses import field

import torch
import torch.nn as nn
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from merge.modules.transformer import Transformer


@dataclass
class LlamaConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int
    rope_theta: Float
    norm_eps: Float
    activation_fn: str = "silu"  # relu, gelu, silu, tanh
    mlp_bias: Bool = False
    mlp_dropout: Float = False
    attention_bias: Bool = False
    attention_drouput: Float = 0.0
    pos_encoding_type: str = "rope"
    mlp: str = "mlp_swiglu"
    normalization: str = "rmsnorm"
    attention: str = "gqa"


class Llama(Transformer):
    def __init__(self, config: LlamaConfig):
        if not isinstance(config, LlamaConfig):
            raise TypeError("Llama model requires a LlamaConfig instance")
        super().__init__(config)

    def forward(
        self, x: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq vocab_size"]:
        return super().forward(x)
