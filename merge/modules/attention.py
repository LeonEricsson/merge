import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from merge.modules.embeddings import apply_rope


def causal_attention_mask(sequence_length):
    mask = torch.tril(torch.ones((1, 1, sequence_length, sequence_length)))
    return torch.where(mask == 1.0, 1.0, -10000.0)


class MHA(nn.Module):

    def __init__(
        self,
        d_model: Int,
        n_heads: Int,
        attn_droput: Float = 0.0,
        res_dropout: Float = 0.0,
        bias: Bool = True,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.W_QKV = nn.Linear(d_model, d_model * 3, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(attn_droput)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq dim"],
        mask: Float[Tensor, "1 1 seq seq"],
        freqs_cis: Tensor,
    ):
        batch_size, seq_length, d_model = x.size()

        x = self.W_QKV(x)

        q, k, v = x.chunk(3, dim=-1)

        q = q.reshape(batch_size, seq_length, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cis)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            qk = (q @ k.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
            qk = qk + mask

            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output


ATTN_MAP = {
    "mha": MHA,
}
