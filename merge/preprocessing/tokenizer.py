from collections import Counter
from typing import List
from typing import Optional
from typing import Union

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer as HFAutoTokenizer


class BPETokenizer:
    """Custom BPE tokenizer implementation"""

    def __init__(self, dataset: List[str], vocab_size: Int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}

        self._build_vocabulary(dataset, vocab_size)

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Int[Tensor, "batch seq"]:
        if isinstance(text, str):
            text = [text]

        encoded_sequences = []

        for sequence in text:
            tokens = " ".join(sequence)

            # Apply merges in order
            for pair in self.merges:
                old = " ".join(pair)
                new = "".join(pair)
                tokens = tokens.replace(old, new)

            token_ids = []
            for token in tokens.split():
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])

            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]

            encoded_sequences.append(token_ids)

        if padding and max_length is not None:
            max_len = max_length
        elif padding:
            max_len = max(len(seq) for seq in encoded_sequences)
        else:
            max_len = max(len(seq) for seq in encoded_sequences)

        if padding:
            for seq in encoded_sequences:
                while len(seq) < max_len:
                    seq.append(0)  # Using 0 as padding token

        return torch.tensor(encoded_sequences)

    def decode(
        self,
        token_ids: Int[Tensor, "batch seq"],
    ) -> Union[str, List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not isinstance(token_ids[0], list):
            token_ids = [token_ids]

        decoded_sequences = []

        for sequence in token_ids:
            tokens = []
            for token_id in sequence:
                if token_id != 0:  # Skip padding tokens
                    tokens.append(self.id_to_token[token_id])

            decoded_text = "".join(tokens)
            decoded_sequences.append(decoded_text)

        return (
            decoded_sequences[0] if len(decoded_sequences) == 1 else decoded_sequences
        )

    def _build_vocabulary(self, dataset: List[str], vocab_size: Int):
        split_words = [" ".join(word) for word in dataset]
        chars = set("".join(dataset))
        self.token_to_id = {char: idx for idx, char in enumerate(chars)}
        self.id_to_token = {idx: char for idx, char in enumerate(chars)}
        current_vocab_size = len(chars)

        while current_vocab_size < vocab_size:
            pair_frequencies = Counter()
            for word in split_words:
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_frequencies[pair] += 1

            # Find most frequent pair
            if not pair_frequencies:
                break  # No more pairs to merge

            best_pair = max(pair_frequencies.items(), key=lambda x: x[1])[0]

            # Merge the pair in all words
            new_token = "".join(best_pair)
            self.merges.append(best_pair)

            # Add new token to vocabulary
            self.token_to_id[new_token] = current_vocab_size
            self.id_to_token[current_vocab_size] = new_token
            current_vocab_size += 1

            # Update the dataset with merged tokens
            new_split_words = []
            for word in split_words:
                tokens = word.split()
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_split_words.append(" ".join(new_tokens))
            split_words = new_split_words


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
    def create(
        cls,
        name: str,
        dataset: Optional[List[str]] = None,
        vocab_size: Optional[Int] = None,
    ):
        """Create a tokenizer instance based on name"""
        if name in cls._CUSTOM_TOKENIZERS:
            return cls._CUSTOM_TOKENIZERS[name](dataset, vocab_size)

        # If not a custom tokenizer, try loading from HuggingFace
        return HFTokenizerWrapper(name)
