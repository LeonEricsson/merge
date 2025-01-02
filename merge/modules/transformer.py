import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from merge.modules.attention import causal_attention_mask
from merge.modules.block import Block
from merge.modules.config import TransformerConfig
from merge.modules.norm import NORM_MAP
from merge.modules.pos_embeddings import PositionEncoding


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.pos_encoding = PositionEncoding(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm_out = NORM_MAP[config.normalization](config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.causal_mask = self.register_buffer(
            "causal_mask", causal_attention_mask(config.seq_len)
        )

    def forward(
        self, x: Int[Tensor, "batch seq"], pad_mask: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask & pad_mask[:, None, None, :]

        x = self.token_emb(x)

        x, pos_info = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, mask, pos_info)

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x
