import json
import os

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader

from merge.architectures.llama import Llama
from merge.architectures.llama import LlamaConfig
from merge.utils.lr_schedule import CosineWarmupScheduler
from merge.utils.tools import auto_device

model_config = LlamaConfig(
    vocab_size=50257,
    seq_len=512,
    d_model=256,
    hidden_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=6,
    rope_theta=0.1,
    norm_eps=1e-6,
    activation_fn="silu",
    mlp_bias=False,
    mlp_dropout=0.0,
    attention_bias=False,
    attention_dropout=0.0,
    pos_encoding_type="rope",
    mlp="mlp_swiglu",
    normalization="rmsnorm",
    attention="gqa",
)

dataset_dir = "data/pretokenized_roneneldan_TinyStories"

batch_size = 32
learning_rate = 5e-5
num_epochs = 10

device = auto_device()
compile_model = True


def validate():
    pass


def train():
    pass


def extract_preprocessed_max_seq_length():
    metadata_path = os.path.join(dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    preprocessing_params = metadata.get("preprocessing_params", {})
    max_seq_length = preprocessing_params.get("max_seq_length")
    return max_seq_length


if __name__ == "__main__":
    max_seq_length = extract_preprocessed_max_seq_length()
    assert max_seq_length == model_config.seq_len

    train_dataset = load_from_disk(dataset_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Llama(model_config).to(device)

    if compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=1000, total_steps=10000)

    #
