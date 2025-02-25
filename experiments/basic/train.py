"""
A training example for a 20M parameter Llama style transformer on the TinyStories dataset.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from jaxtyping import Float, Int
from torch.utils.data import DataLoader

import wandb
from experiments.logger import TrainingLogger
from omni.architectures.llama import LlamaConfig
from omni.modules.transformer import Transformer
from omni.utils.lr_schedule import CosineWarmupScheduler
from omni.utils.setup import parse_args, validate_model_initialization
from omni.utils.system import auto_device

model_config = LlamaConfig(
    vocab_size=50258,
    seq_len=512,
    d_model=256,
    hidden_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=6,
    activation_fn="silu",
    mlp_bias=True,
    mlp_dropout=0.0,
    attention_bias=True,
    attention_dropout=0.0,
    pos_encoding_type="alibi",
    mlp="mlp_swiglu",
    normalization="rmsnorm",
    attention="gqa",
)

training_config = {
    "batch_size": 1,
    "learning_rate": 5e-4,
    "min_lr": 1e-7,
    "num_epochs": 10,
    "num_workers": 4,
    "eval_every": 500,
    "warmup_steps": 1000,
    "tot_steps": 50000,
    "gradient_clip_norm": 1.0,
    "seed": 42,
    "device": "cpu",
}

dataset_dir = Path(
    "data/pretokenized_roneneldan_TinyStories"
)  # pretokenized - run preprocess.py first


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb.init(
        project="Llama",
        config=config,
        notes="LLaMA architecture training on TinyStories",
        tags=["llama", "tinystories", "pre-training"],
        mode="disabled",
    )


def validate(
    test_dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    logger: TrainingLogger,
    ignore_index: Int = -1,
) -> Dict[str, float]:
    """
    Validate the model, logging loss and perplexity.

    Args:
        val_dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be validated.
        total_tokens (int): Total number of tokens processed.
        device (torch.device): Device to run the validation on.
        logger (TrainingLogger): Logger for tracking validation metrics.
        ignore_index (int): Index to ignore in the loss calculation.

    Returns:
        Dict[str, Any]: Validation metrics.
    """
    model.eval()
    total_loss = 0
    val_tokens = 0

    logger.start_validation(len(test_dataloader))

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"][:, :-1].to(device)
            target_ids = batch["input_ids"][:, 1:].to(device)
            attention_mask = batch["attention_mask"][:, :-1].to(device)

            with torch.autocast(device.type, enabled=amp_available):
                logits = model(input_ids, attention_mask)
                loss = cross_entropy_loss(
                    logits, target_ids, ignore_index, reduction="sum"
                )

            total_loss += loss.item()
            val_tokens += attention_mask.sum().item()

            logger.advance_validation()

    avg_loss = total_loss / val_tokens
    clipped_loss = min(avg_loss, 100)
    perplexity = torch.exp(torch.tensor(clipped_loss))

    metrics = {
        "val/loss": avg_loss,
        "val/perplexity": perplexity.item(),
    }

    return metrics


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: Int,
    gradient_clip_norm: Float,
    gradient_acc_steps: Int = 1,
    eval_every: Int = 100,
    ignore_index: Int = -1,
    amp_available: bool = False,
):
    """
    Trains the model for 'num_epochs'.

    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        gradient_clip_norm (float): Maximum norm for gradient clipping.
        eval_every (int, optional): Frequency of evaluation during training. Defaults to 100.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to -1.
    """
    model.train()
    total_steps = 0
    total_tokens = 0

    start_time = time.perf_counter()

    optimizer.zero_grad()
    scaler = torch.amp.GradScaler()

    with TrainingLogger(num_epochs, len(train_dataloader)) as logger:
        logger.start_training(device)
        for epoch in range(num_epochs):
            logger.start_epoch(epoch, total_steps)

            for step, batch in enumerate(train_dataloader, start=1):
                input_ids = batch["input_ids"][:, :-1].to(device)
                target_ids = batch["input_ids"][:, 1:].to(device)
                attention_mask = batch["attention_mask"][:, :-1].to(device)

                with torch.autocast(device.type, enabled=amp_available):
                    logits = model(input_ids, attention_mask)
                    loss = cross_entropy_loss(logits, target_ids, ignore_index, "mean")

                scaler.scale(loss).backward()

                # gradient accumulation
                if step % gradient_acc_steps == 0 or step == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=gradient_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                total_steps += 1
                batch_tokens = attention_mask.sum().item()
                total_tokens += batch_tokens

                if total_steps % eval_every == 0:
                    elapsed_time = time.perf_counter() - start_time
                    tokens_per_second = total_tokens / elapsed_time
                    train_metrics = {
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "utils/tokens_per_second": tokens_per_second,
                    }
                    logger.log_training_step(train_metrics, total_steps)

                    # Run validation
                    model.eval()
                    validation_metrics = validate(
                        val_dataloader,
                        model,
                        device,
                        logger,
                        ignore_index,
                    )
                    logger.log_validation_step(validation_metrics, total_steps)
                    model.train()

                train_metrics = {
                    "train/loss": loss.item(),
                    "Tokens": total_tokens,
                }
                logger.log_training_step(train_metrics, total_steps)
                logger.advance_train()

            logger.end_epoch(epoch, total_steps)
        logger.end_training()


def cross_entropy_loss(logits, target_ids, ignore_index, reduction):
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.reshape(batch_size * seq_len, vocab_size)
    target_ids = target_ids.reshape(batch_size * seq_len)
    return F.cross_entropy(
        logits, target_ids, ignore_index=ignore_index, reduction=reduction
    )


def extract_metadata(dataset_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


def main():
    cli_args = parse_args(training_config)
    training_config.update(cli_args)

    device = auto_device(training_config["device"])
    amp_available = torch.amp.autocast_mode.is_autocast_available(device.type)

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
    train_size = int(0.975 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    def init_dataloader(dataset):
        return DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=True,  # avoid torch model recompilation
        )

    train_dataloader = init_dataloader(train_dataset)
    val_dataloader = init_dataloader(val_dataset)

    model = Transformer(model_config).to(device)
    if device.type == "cuda":
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
        total_steps=training_config["tot_steps"],
        min_lr=training_config["min_lr"],
    )

    validate_model_initialization(dataset, model, device, ignore_index=pad_token_id)

    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=training_config["gradient_clip_norm"],
        num_epochs=training_config["num_epochs"],
        eval_every=training_config["eval_every"],
        ignore_index=pad_token_id,
        amp_available=amp_available,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
