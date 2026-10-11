"""Training entry point for Zyvanta.

Single-GPU training by design: device selection is explicit, no DistributedData
Parallel is used, and both Tesla T4s are left available for separate runs.
"""

import argparse
import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from zyvanta.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from zyvanta.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_STEPS,
    SAVE_INTERVAL,
    SEED,
    TOKENIZER_SAMPLE_LIMIT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from zyvanta.data import SyntheticTextGenerator, load_text_files, make_loader
from zyvanta.model import ZyvantaLM
from zyvanta.tokenizer import ByteBPETokenizer


def get_device():
    """Pick the best available device: CUDA, then Apple Silicon, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer_and_scheduler(model, steps: int):
    """AdamW with a linear warmup then cosine decay schedule."""
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step: int):
        if step < WARMUP_STEPS:
            return (step + 1) / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train(
    steps: int = MAX_STEPS,
    resume: bool = True,
    device=None,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    data_dir=None,
):
    """Run the training loop with checkpoint save/resume.

    ``data_dir`` points at a directory of plain .txt corpus files. When it is
    provided, the tokenizer is fit on that real corpus and training runs on
    it. When omitted, synthetic text is used -- this is the smoke-test path
    only, not a substitute for real training data.
    """
    torch.manual_seed(SEED)
    device = device or get_device()

    # 1. Data: real corpus or synthetic fallback.
    if data_dir is not None:
        texts = load_text_files(data_dir)
    else:
        texts = list(SyntheticTextGenerator(seed=SEED))

    # 2. Model + device.
    model = ZyvantaLM().to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model, steps)

    # 3. Tokenizer: restore from checkpoint when resuming, otherwise fit fresh.
    # Fitting is skipped on resume so the checkpoint's exact tokenizer is used
    # rather than silently re-fitting and then overwriting it. If the checkpoint
    # predates tokenizer persistence (no merges stored), fit so the tokenizer
    # is usable instead of empty.
    tokenizer = ByteBPETokenizer()
    start_step = 0
    if resume:
        latest = latest_checkpoint(CHECKPOINT_DIR)
        if latest is not None:
            start_step, _ = load_checkpoint(model, optimizer, latest, tokenizer=tokenizer)
            print(f"Resumed from {latest} at step {start_step}")
            if not tokenizer.merges:
                tokenizer.fit(texts, sample_limit=TOKENIZER_SAMPLE_LIMIT)
        else:
            tokenizer.fit(texts, sample_limit=TOKENIZER_SAMPLE_LIMIT)
    else:
        tokenizer.fit(texts, sample_limit=TOKENIZER_SAMPLE_LIMIT)

    # 4. Data + loop.
    loader = make_loader(texts, tokenizer, BATCH_SIZE)
    loader_iter = iter(loader)

    model.train()
    for step in range(start_step, steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(batch["input_ids"], targets=batch["targets"])
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % log_interval == 0:
            print(f"step {step} loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

        if (step + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, step + 1, loss=loss.item(), tokenizer=tokenizer)

    save_checkpoint(model, optimizer, steps, loss=loss.item(), tokenizer=tokenizer)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Zyvanta from scratch.")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory of .txt corpus files. Omit for synthetic smoke-test text.")
    args = parser.parse_args()
    train(steps=args.steps, resume=not args.no_resume, data_dir=args.data_dir)