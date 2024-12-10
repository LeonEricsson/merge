import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Complex
from torch import Tensor

from merge.modules.config import TransformerConfig
from merge.modules.mlp import MLP_MAP
from merge.modules.attention import ATTN_MAP
from merge.modules.norm import NORM_MAP
from merge.modules.block import Block
from merge.modules.pos_embeddings import precompute_freqs_cis

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        attention = ATTN_MAP[config.attention] # needs input on init?
        mlp = MLP_MAP[config.mlp] # needs input on init
        norm1 = NORM_MAP[config.normalization] # need input param on init
        norm2 = NORM_MAP[config.normalization]
        
        self.blocks = nn.ModuleList([Block(attention, mlp, norm1, norm2) for _ in config.num_layers])

        self.norm_out = NORM_MAP[config.normalization]

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.freq_cis = precompute_freqs_cis(config.d_model // config.num_heads, config.seq_len, config.rope_theta)
