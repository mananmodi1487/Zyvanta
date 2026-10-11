"""Checkpoint save and resume for Zyvanta training.

Saves the model state, optimizer state, and training step so a run can be
paused and resumed. Checkpoints are plain PyTorch ``.pt`` files; no external
service is involved.
"""

from pathlib import Path
from typing import Optional

import torch

from zyvanta.config import CHECKPOINT_DIR
from zyvanta.tokenizer import ByteBPETokenizer


def save_checkpoint(model, optimizer, step: int, path=None, loss: Optional[float] = None, tokenizer: Optional["ByteBPETokenizer"] = None):
    """Save model + optimizer + step + tokenizer to ``path`` (a directory)."""
    directory = Path(path) if path is not None else Path(CHECKPOINT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if tokenizer is not None:
        payload["tokenizer_merges"] = [list(m) for m in tokenizer.merges]
        payload["tokenizer_vocab"] = tokenizer.vocab
    if loss is not None:
        payload["loss"] = float(loss)
    out = directory / f"step_{step}.pt"
    torch.save(payload, out)
    return out


def load_checkpoint(model, optimizer, path, tokenizer: Optional["ByteBPETokenizer"] = None):
    """Restore model + optimizer state + tokenizer from ``path`` and return (step, loss)."""
    ckpt = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt.get("step", 0))
    loss = ckpt.get("loss", None)
    if tokenizer is not None and "tokenizer_merges" in ckpt:
        tokenizer.merges = [tuple(m) for m in ckpt["tokenizer_merges"]]
        tokenizer.vocab = ckpt["tokenizer_vocab"]
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.pad_id = tokenizer.vocab["<pad>"]
        tokenizer.bos_id = tokenizer.vocab["<bos>"]
        tokenizer.eos_id = tokenizer.vocab["<eos>"]
    return step, loss


def latest_checkpoint(directory=CHECKPOINT_DIR) -> Optional[Path]:
    """Return the most recent checkpoint in ``directory``, or None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = list(directory.glob("step_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.stem.split("_")[1]))