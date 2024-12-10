from dataclasses import dataclass

from jaxtyping import Float
from jaxtyping import Int


@dataclass
class TransformerConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_layers: Int
    dropout: Float
    bias: bool
    activation_fn: str
    mlp: str
    normalization: str
    attention: str
