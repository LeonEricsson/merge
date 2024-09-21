from pathlib import Path

import torch.nn.functional as F
from datasets import Dataset
from datasets import DownloadMode
from datasets import load_dataset
from jaxtyping import Int
from torch import Tensor

from merge.preprocessing.tokenizer import AutoTokenizer


def prepare_dataset(
    dataset_name: str,
    tokenizer_name: str,
    min_seq_length: Int,
    max_seq_length: Int,
    padding_token: Int,
    name: str,
    split: str | None = None,
    revision: str | None = None,
    output_dir: Path = Path("./data"),
    cache_dir: Path = Path("./hf_cache"),
    push: bool = False,
    hf_username: str | None = None,
) -> None:

    dataset = _download_dataset(
        dataset_name=dataset_name,
        name=name,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
    )

    dataset = _tokenize(dataset, tokenizer_name)

    dataset = _process_sequences(dataset, min_seq_length, max_seq_length, padding_token)

    _save(dataset, output_dir)

    if push:
        assert hf_username is not None
        id = f"{hf_username}/pretokenized_{name}_{tokenizer_name}"
        dataset.push_to_hub(id, split=split)


def _download_dataset(
    dataset_name: str,
    name: str,
    split: str | None = None,
    revision: str | None = None,
    cache_dir: Path = Path("./hf_cache"),
) -> Dataset:

    dataset = load_dataset(
        dataset_name,
        name=name,
        split=split,
        revision=revision,
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        cache_dir=cache_dir,
        num_proc=4,
    )

    return dataset.select_columns("text")


def _tokenize(dataset: Dataset, tokenizer_name: str) -> Dataset:
    tokenizer = AutoTokenizer.create(tokenizer_name)
    dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset


def _tokenize_truncate(dataset: Dataset, tokenizer_name: str) -> Dataset:
    tokenizer = AutoTokenizer.create(tokenizer_name)
    dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset


def _process_sequences(
    dataset: Dataset,
    min_seq_length: Int,
    max_seq_length: Int,
    padding_token: Int,
) -> Dataset:
    dataset = dataset.filter(lambda x: len(x["input_ids"]) >= min_seq_length)

    def split_and_pad(example):
        """
        - Sequences shorter than `max_seq_length` are padded to `max_seq_length`.
        - Sequences longer than `max_seq_length` are split into chunks of `max_seq_length` (padded).
        """
        input_ids: Int[Tensor, "batch len"] = example["input_ids"]
        attention_mask: Int[Tensor, "batch len"] = example["attention_mask"]
        total_length = input_ids.shape[1]

        if total_length <= max_seq_length:
            padding_length = max_seq_length - total_length
            return {
                "input_ids": F.pad(input_ids, (0, padding_length), value=padding_token),
                "attention_mask": F.pad(
                    attention_mask, (0, padding_length), value=padding_token
                ),
            }

        chunks = {"input_ids": [], "attention_mask": []}
        for start in range(0, total_length, max_seq_length):
            end = start + max_seq_length
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]

            if len(chunk_ids) >= min_seq_length:
                if len(chunk_ids) < max_seq_length:
                    padding_length = max_seq_length - len(chunk_ids)
                    chunk_ids = chunk_ids + [0] * padding_length
                    chunk_mask = chunk_mask + [0] * padding_length
                chunks["input_ids"].append(chunk_ids)
                chunks["attention_mask"].append(chunk_mask)
        print(chunks)
        return chunks

    return dataset.map(
        split_and_pad,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
    )


def _save(dataset: Dataset, output_dir: Path) -> None:
    dataset.save_to_disk(output_dir)
