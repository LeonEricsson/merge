from typing import List
from typing import Optional
from typing import Union

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer as HFAutoTokenizer


class BPETokenizer:
    """Custom BPE tokenizer implementation"""

    def __init__(self):
        super().__init__()

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Int[Tensor, "batch seq"]:
        # Custom BPE encoding implementation
        pass

    def decode(
        self,
        token_ids: Int[Tensor, "batch seq"],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        # Custom BPE decoding implementation
        pass


class HFTokenizerWrapper:
    """Wrapper for HuggingFace tokenizers that implements our interface"""

    def __init__(self, name: str):
        super().__init__()
        self.tokenizer = HFAutoTokenizer.from_pretrained(name)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


class AutoTokenizer:
    """Factory class for creating tokenizers"""

    _CUSTOM_TOKENIZERS = {
        "bpe": BPETokenizer,
    }

    @classmethod
    def from_pretrained(cls, name: str):
        """Create a tokenizer instance based on name"""
        if name in cls._CUSTOM_TOKENIZERS:
            return cls._CUSTOM_TOKENIZERS[name]()

        # If not a custom tokenizer, try loading from HuggingFace
        return HFTokenizerWrapper(name)
