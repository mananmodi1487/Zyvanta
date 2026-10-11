"""Data pipeline for Zyvanta training.

Provides a streaming text dataset that tokenizes lazily and packs fixed-length
blocks, plus a tiny synthetic-text generator used by the smoke test so the full
pipeline can be exercised without downloading any corpus.
"""

import random
from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import Dataset, IterableDataset

from zyvanta.config import BOS_ID, EOS_ID, MAX_SEQ_LEN, PAD_ID
from zyvanta.tokenizer import ByteBPETokenizer


class TextDataset(Dataset):
    """Packs a list of texts into fixed-length token blocks.

    Each sample is a block of ``block_len`` tokens; the final partial block is
    dropped so every batch has a uniform shape. Texts are concatenated with a
    BOS token between them so the model sees document boundaries.
    """

    def __init__(self, texts: List[str], tokenizer: ByteBPETokenizer, block_len: int = MAX_SEQ_LEN):
        self.block_len = block_len
        self.tokenizer = tokenizer
        self.blocks: List[List[int]] = []
        buffer: List[int] = []
        for text in texts:
            ids = tokenizer.encode(text)
            if not ids:
                continue
            buffer.extend([BOS_ID])
            buffer.extend(ids)
            buffer.extend([EOS_ID])
            while len(buffer) >= block_len:
                self.blocks.append(buffer[:block_len])
                buffer = buffer[block_len:]
        # Pad the final partial block so every sample has a uniform shape.
        # Without this, a short corpus yields no full blocks and the
        # DataLoader returns nothing (drop_last=True drops the partial block).
        if buffer:
            pad = [PAD_ID] * (block_len - len(buffer))
            self.blocks.append(buffer + pad)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.blocks[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": x, "targets": y}


class SyntheticTextGenerator:
    """Produces varied synthetic text in-process for smoke testing only.

    The text is not linguistically meaningful -- it exists solely to verify the
    tokenizer, dataloader, training loop, checkpointing, and generation all
    work together. It is not a substitute for real training data.
    """

    VOCAB = (
        "the quick brown fox jumps over the lazy dog\n"
        "python is a programming language\n"
        "def train(model, data):\n"
        "    return model(data)\n"
        "hello world\n"
        "mera naam hai zyvanta\n"
        "yeh ek project hai\n"
        "kya aapko samajh aaya\n"
        "machine learning is fun\n"
        "import torch\n"
        "torch.manual_seed(42)\n"
        "for i in range(10):\n"
        "    print(i)\n"
    )

    def __init__(self, seed: int = 0, length: int = 4096):
        self.rng = random.Random(seed)
        self.length = length

    def __iter__(self) -> Iterator[str]:
        words = self.VOCAB.split()
        for _ in range(self.length):
            yield self.rng.choice(words)


def load_text_files(directory) -> List[str]:
    """Load every ``.txt`` file in ``directory`` recursively as a list of strings.

    Raises a clear ``FileNotFoundError`` when the directory is missing or
    contains no ``.txt`` files, so a misconfigured data path fails fast
    instead of silently training on nothing. Subdirectories are traversed,
    so structures like ``data/english/`` and ``data/hindi/`` are preserved.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Data directory '{directory}' does not exist or is not a directory"
        )
    files = sorted(directory.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"Data directory '{directory}' contains no .txt files"
        )
    texts: List[str] = []
    for path in files:
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def make_loader(texts: List[str], tokenizer: ByteBPETokenizer, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Build a standard DataLoader over fixed-length token blocks."""
    dataset = TextDataset(texts, tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )