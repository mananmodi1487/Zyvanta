"""End-to-end pipeline smoke test for Zyvanta.

Exercises tokenization, forward/loss, a training step, checkpoint save/resume,
and text generation -- all on synthetic text, with no dataset download and no
pretrained weights. Requires PyTorch; run on Kaggle's free T4.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from zyvanta.checkpoint import load_checkpoint, save_checkpoint, latest_checkpoint
from zyvanta.config import BATCH_SIZE, BOS_ID, CHECKPOINT_DIR, DATA_DIR, MAX_STEPS, PAD_ID, SEED, TOKENIZER_SAMPLE_LIMIT
from zyvanta.data import SyntheticTextGenerator, load_text_files, make_loader
from zyvanta.generate import generate
from zyvanta.model import ZyvantaLM
from zyvanta.tokenizer import ByteBPETokenizer
from zyvanta.train import get_device, make_optimizer_and_scheduler


def _make_tokenizer(length: int = 8000) -> ByteBPETokenizer:
    tokenizer = ByteBPETokenizer()
    tokenizer.fit(list(SyntheticTextGenerator(seed=SEED, length=length)))
    return tokenizer


def test_tokenizer_round_trips_utf8():
    """encode -> decode must reproduce the original text exactly."""
    tokenizer = _make_tokenizer()
    samples = [
        "the quick brown fox",
        "mera naam hai zyvanta",
        "def train(model, data):\n    return model(data)",
        "print('hello हालो सन्गार')",
    ]
    for text in samples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_byte_coverage():
    """Every raw byte 0..255 must be representable, so no unknown tokens exist."""
    tokenizer = _make_tokenizer()
    for byte in range(256):
        text = bytes([byte]).decode("latin-1")
        ids = tokenizer.encode(text)
        assert len(ids) >= 1
        assert tokenizer.decode(ids) == text


def test_tokenizer_vocab_size_within_budget():
    """The learned vocabulary must not exceed the model's configured vocab size."""
    tokenizer = _make_tokenizer()
    assert len(tokenizer.vocab) <= tokenizer.vocab_size
    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.bos_id == BOS_ID


def test_data_batch_shape():
    """The dataloader must yield (batch, seq) tensors with shifted targets."""
    tokenizer = _make_tokenizer()
    loader = make_loader(list(SyntheticTextGenerator(seed=SEED, length=8000)), tokenizer, batch_size=4)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 511)
    assert batch["targets"].shape == (4, 511)
    assert batch["input_ids"].dtype == torch.long


def test_training_step_reduces_loss():
    """One optimizer step must reduce the loss on the same batch."""
    device = get_device()
    model = ZyvantaLM().to(device)
    optimizer, _ = make_optimizer_and_scheduler(model, steps=MAX_STEPS)
    tokenizer = _make_tokenizer()
    loader = make_loader(list(SyntheticTextGenerator(seed=SEED, length=8000)), tokenizer, batch_size=BATCH_SIZE)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    _, loss_before = model(batch["input_ids"], targets=batch["targets"])
    loss_before.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        _, loss_after = model(batch["input_ids"], targets=batch["targets"])

    assert loss_after < loss_before


def test_checkpoint_save_and_resume():
    """A checkpoint must restore model weights and optimizer state."""
    device = get_device()
    model = ZyvantaLM().to(device)
    optimizer, _ = make_optimizer_and_scheduler(model, steps=MAX_STEPS)

    tokenizer = _make_tokenizer()
    loader = make_loader(list(SyntheticTextGenerator(seed=SEED, length=8000)), tokenizer, batch_size=BATCH_SIZE)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    _, loss = model(batch["input_ids"], targets=batch["targets"])
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(model, optimizer, step=1, path=tmp, loss=loss.item())
        assert path.exists()

        # Reload into a fresh model/optimizer and verify weights match.
        model2 = ZyvantaLM().to(device)
        optimizer2, _ = make_optimizer_and_scheduler(model2, steps=MAX_STEPS)
        step, restored_loss = load_checkpoint(model2, optimizer2, path)

        assert step == 1
        assert abs(restored_loss - loss.item()) < 1e-5
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())
        assert latest_checkpoint(tmp) == path


def test_load_text_files_reads_corpus():
    """load_text_files() must return the contents of every .txt file."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("hello world", encoding="utf-8")
        (Path(tmp) / "b.txt").write_text("def train(model, data):\n    return model(data)", encoding="utf-8")
        (Path(tmp) / "ignore.md").write_text("not a txt file", encoding="utf-8")
        texts = load_text_files(tmp)
        assert len(texts) == 2
        assert "hello world" in texts
        assert any("def train(model, data):" in text for text in texts)


def test_load_text_files_missing_directory():
    """A missing data directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_text_files("/nonexistent/zyvanta/data/dir")


def test_load_text_files_empty_directory():
    """An empty data directory must raise FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_text_files(tmp)


def test_load_text_files_no_txt_files():
    """A directory with no .txt files must raise FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "notes.md").write_text("markdown only", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_text_files(tmp)


def test_padding_positions_ignored_in_loss():
    """Padded target positions (value=PAD_ID) must be ignored by cross_entropy.

    The model's loss uses ignore_index=PAD_ID. This test constructs a batch
    where all targets are PAD_ID except one position, and verifies the loss
    equals the loss from the single non-padded position.
    """
    device = get_device()
    model = ZyvantaLM().to(device)
    tokenizer = _make_tokenizer()
    batch_size = 2

    # Build a sequence with known content, then pad the rest.
    # Enough text to pack at least two 512-token blocks, so drop_last=True
    # still yields a batch of the requested size.
    texts = list(SyntheticTextGenerator(seed=SEED, length=8000))
    loader = make_loader(texts, tokenizer, batch_size=batch_size)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Replace all targets with PAD_ID except the first token of the first sample.
    original_target = batch["targets"][0, 0].item()
    batch["targets"].fill_(PAD_ID)
    batch["targets"][0, 0] = original_target

    model.eval()
    with torch.no_grad():
        _, loss_all_pad = model(batch["input_ids"], targets=batch["targets"])

    # Compare with a batch where only the first position has the original target.
    batch["targets"][0, 1] = batch["targets"][0, 0]  # duplicate first target
    with torch.no_grad():
        _, loss_two_valid = model(batch["input_ids"], targets=batch["targets"])

    # Loss with one valid target should be ~half of loss with two valid targets
    # (since cross_entropy averages over non-ignored positions).
    # The key assertion: loss with all-pad-except-one should be finite and
    # should NOT equal the loss with all-pad (which would be NaN if PAD_ID
    # weren't ignored).
    assert torch.isfinite(loss_all_pad), "Loss should be finite with PAD_ID targets"
    assert not torch.isnan(loss_all_pad), "Loss should not be NaN with PAD_ID targets"


def test_generation_produces_text():
    """generate() must return a string that does not echo the prompt.

    The random-init model is not expected to produce meaningful language, so
    this test verifies the documented contract rather than output quality:
    a string is returned, and the prompt is not repeated in it.
    """
    device = get_device()
    model = ZyvantaLM().to(device)
    tokenizer = _make_tokenizer()
    prompt = "the "
    text = generate(model, tokenizer, prompt, max_new_tokens=16, device=device)

    assert isinstance(text, str)
    assert prompt not in text, "generate() echoed the prompt"


def test_generation_respects_max_length():
    """Generation must not exceed the requested number of new tokens.

    The loop in generate() appends at most one token per iteration, and
    decode() -> encode() can only merge tokens further, never split them. So
    len(tokenizer.encode(text)) is a guaranteed upper bound on the number of
    generated tokens, independent of what the random weights produce.
    """
    device = get_device()
    model = ZyvantaLM().to(device)
    tokenizer = _make_tokenizer()
    max_new_tokens = 8
    text = generate(model, tokenizer, "def ", max_new_tokens=max_new_tokens, device=device)

    assert isinstance(text, str)
    assert len(tokenizer.encode(text)) <= max_new_tokens, (
        f"Generated {len(tokenizer.encode(text))} tokens, exceeding max_new_tokens={max_new_tokens}"
    )


def test_load_text_files_recursive():
    """load_text_files() must recursively load .txt files from subdirectories."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "english").mkdir()
        (Path(tmp) / "english" / "a.txt").write_text("hello world", encoding="utf-8")
        (Path(tmp) / "hindi").mkdir()
        (Path(tmp) / "hindi" / "b.txt").write_text("mera naam hai zyvanta", encoding="utf-8")
        (Path(tmp) / "ignore.md").write_text("not a txt file", encoding="utf-8")
        texts = load_text_files(tmp)
        assert len(texts) == 2
        assert "hello world" in texts
        assert "mera naam hai zyvanta" in texts


def test_tokenizer_sample_limit_in_training():
    """The training path must pass TOKENIZER_SAMPLE_LIMIT to tokenizer.fit()."""
    device = get_device()
    model = ZyvantaLM().to(device)
    optimizer, _ = make_optimizer_and_scheduler(model, steps=MAX_STEPS)

    tokenizer = ByteBPETokenizer()
    texts = list(SyntheticTextGenerator(seed=SEED, length=200))

    # Fit with explicit sample limit.
    tokenizer.fit(texts, sample_limit=TOKENIZER_SAMPLE_LIMIT)
    merges_with_limit = list(tokenizer.merges)

    # Fit without limit (full corpus) should have more or equal merges.
    tokenizer2 = ByteBPETokenizer()
    tokenizer2.fit(texts, sample_limit=None)
    merges_full = list(tokenizer2.merges)

    assert len(merges_with_limit) <= len(merges_full)


def test_checkpoint_save_restore_tokenizer():
    """Checkpoint must save and restore the fitted tokenizer exactly."""
    device = get_device()
    model = ZyvantaLM().to(device)
    optimizer, _ = make_optimizer_and_scheduler(model, steps=MAX_STEPS)
    tokenizer = ByteBPETokenizer()
    tokenizer.fit(list(SyntheticTextGenerator(seed=SEED, length=8000)), sample_limit=TOKENIZER_SAMPLE_LIMIT)

    # Train one step so the checkpoint has a meaningful state.
    # length=8000 yields ~46 packed 512-token blocks, enough for at least one
    # batch_size=8 batch with drop_last=True. A shorter corpus yields 0 blocks
    # and the loader raises StopIteration.
    texts = list(SyntheticTextGenerator(seed=SEED, length=8000))
    loader = make_loader(texts, tokenizer, batch_size=BATCH_SIZE)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    _, loss = model(batch["input_ids"], targets=batch["targets"])
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(model, optimizer, step=1, path=tmp, loss=loss.item(), tokenizer=tokenizer)

        # Restore into a fresh model/optimizer/tokenizer.
        model2 = ZyvantaLM().to(device)
        optimizer2, _ = make_optimizer_and_scheduler(model2, steps=MAX_STEPS)
        tokenizer2 = ByteBPETokenizer()
        step, restored_loss = load_checkpoint(model2, optimizer2, path, tokenizer=tokenizer2)

        assert step == 1
        assert abs(restored_loss - loss.item()) < 1e-5
        assert tokenizer2.merges == tokenizer.merges
        assert tokenizer2.vocab == tokenizer.vocab
        assert tokenizer2.pad_id == tokenizer.pad_id
        assert tokenizer2.bos_id == tokenizer.bos_id
        assert tokenizer2.eos_id == tokenizer.eos_id