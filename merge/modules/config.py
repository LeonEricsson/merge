from dataclasses import dataclass

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


@dataclass
class TransformerConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int
    mlp_bias: Bool
    mlp_dropout: Float
    attention_bias: Bool
    attention_dropout: Float
    rope_theta: Float
    norm_eps: Float
    pos_encoding_type: str  # rope, absolute
    activation_fn: str  # relu, gelu, silu, tanh
    mlp: str  # mlp, mlp_swiglu
    normalization: str  # rmsnorm, layernorm, none
    attention: str  # mha
