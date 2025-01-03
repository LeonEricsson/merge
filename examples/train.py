import json
import os
from pathlib import Path
from typing import Any
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from jaxtyping import Int
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader

from merge.architectures.llama import Llama
from merge.architectures.llama import LlamaConfig
from merge.utils.lr_schedule import CosineWarmupScheduler
from merge.utils.tools import auto_device
from merge.utils.tools import get_gpu_memory
from merge.utils.tools import get_system_stats

console = Console()

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

training_config = {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "eval_every": 100,
    "warmup_steps": 1000,
    "total_steps": 10000,
    "gradient_clip_norm": 1.0,
    "seed": 42,
}

dataset_dir = Path(
    "data/pretokenized_roneneldan_TinyStories"
)  # pretokenized - run preprocess.py first


device = auto_device()
console.print(f"[blue]Using device: {device}")


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb.init(
        project="Llama",
        config=config,
        notes="LLaMA architecture training on TinyStories",
        tags=["llama", "tinystories", "pre-training"],
    )


def validate(
    test_dataloader: DataLoader,
    model: nn.Module,
    epoch: int,
    device: torch.device,
    ignore_index: int = -1,
) -> None:

    model.eval()
    total_loss = 0
    total_tokens = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        validate_task = progress.add_task(
            "[yellow]Validating...", total=len(test_dataloader)
        )

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"][:, :-1].to(device)
                target_ids = batch["input_ids"][:, 1:].to(device)
                attention_mask = batch["attention_mask"][:, :-1].to(device)

                logits = model(input_ids, attention_mask)
                batch_size, seq_len, vocab_size = logits.size()

                logits = logits.reshape(batch_size * seq_len, vocab_size)
                target_ids = target_ids.reshape(batch_size * seq_len)

                loss = F.cross_entropy(
                    logits, target_ids, ignore_index=ignore_index, reduction="sum"
                )

                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()

                progress.advance(validate_task)

    avg_loss = total_loss / total_tokens
    clipped_loss = min(avg_loss, 100)
    perplexity = torch.exp(torch.tensor(clipped_loss))

    metrics = {
        "val/loss": avg_loss,
        "val/perplexity": perplexity.item(),
        **get_gpu_memory(),
        **get_system_stats(),
    }

    wandb.log(metrics, step=epoch)

    table = Table(title=f"Validation Metrics - Epoch {epoch}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    gradient_clip_norm: float,
    eval_every: int = 100,
    ignore_index: int = -1,
) -> None:

    model.train()
    total_steps = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        for epoch in range(num_epochs):
            epoch_task = progress.add_task(
                f"[green]Epoch {epoch + 1}/{num_epochs}", total=len(train_dataloader)
            )

            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                input_ids: Int[Tensor, "batch seq"] = batch["input_ids"][:, :-1].to(
                    device
                )
                target_ids: Int[Tensor, "batch seq"] = batch["input_ids"][:, 1:].to(
                    device
                )
                attention_mask: Int[Tensor, "batch seq"] = batch["attention_mask"][
                    :, :-1
                ].to(device)

                logits = model(input_ids, attention_mask)

                batch, seq, vocab_size = logits.size()
                logits = logits.reshape(batch * seq, vocab_size)
                target_ids = target_ids.reshape(batch * seq)
                loss = F.cross_entropy(logits, target_ids, ignore_index=ignore_index)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip_norm
                )

                optimizer.step()
                scheduler.step()

                total_steps += 1
                batch_tokens = attention_mask.sum().item()
                total_tokens += batch_tokens

                if total_steps % eval_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    metrics = {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/step": total_steps,
                        "train/total_tokens": total_tokens,
                        **get_gpu_memory(),
                        **get_system_stats(),
                    }
                    wandb.log(metrics, step=total_tokens)  # Use tokens as step

                    # Log to console
                    console.print(f"\nTokens processed: {total_tokens:,}")
                    console.print(f"Loss: {loss.item():.4f}")
                    console.print(f"Learning rate: {current_lr:.2e}")

                    # Run validation
                    model.eval()
                    validate(val_dataloader, model, total_tokens, device, ignore_index)
                    model.train()

                progress.advance(epoch_task)

            validate()


def extract_metadata(dataset_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    torch.manual_seed(training_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config["seed"])

    setup_wandb({**model_config.__dict__, **training_config})

    # load data
    data_metadata = extract_metadata(str(dataset_dir))
    max_seq_length = data_metadata["preprocessing_params"]["max_seq_length"]
    pad_token_id = data_metadata["pad_token_id"]
    assert max_seq_length >= model_config.seq_len

    # create dataloaders
    dataset = load_from_disk(str(dataset_dir))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = Llama(model_config).to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=training_config["warmup_steps"],
        total_steps=training_config["total_steps"],
    )

    console.print("[green]Starting training...")
    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=training_config["num_epochs"],
        eval_every=training_config["eval_every"],
        ignore_index=pad_token_id,
    )

    wandb.finish()
