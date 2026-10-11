"""Tests for scratch-initialized Zyvanta Transformer.

These tests verify that the model is genuinely randomly initialized and
behaves correctly on a forward pass. No pretrained weights or external files
are used anywhere.
"""

import pytest
import torch
import torch.nn as nn

from zyvanta.model import ZyvantaLM


def _make_model(seed: int = 0, **kwargs) -> ZyvantaLM:
    """Instantiate a model with a fixed RNG seed for reproducible tests."""
    torch.manual_seed(seed)
    return ZyvantaLM(**kwargs)


def test_same_seed_is_deterministic() -> None:
    """Two models with the same seed must have identical parameters."""
    m1 = _make_model(seed=42)
    m2 = _make_model(seed=42)

    p1 = list(m1.parameters())
    p2 = list(m2.parameters())
    assert len(p1) == len(p2)

    for a, b in zip(p1, p2):
        assert a.shape == b.shape
        assert torch.equal(a, b), "Parameters differ for the same seed"


def test_different_seed_is_distinct() -> None:
    """Two models with different seeds must NOT have identical parameters."""
    m1 = _make_model(seed=1)
    m2 = _make_model(seed=2)

    any_diff = False
    for a, b in zip(m1.parameters(), m2.parameters()):
        if not torch.equal(a, b):
            any_diff = True
            break
    assert any_diff, "Different seeds produced identical parameters"


def test_no_all_zero_or_constant_parameters() -> None:
    """Random init must not leave non-LayerNorm parameters at zero or a constant.

    LayerNorm parameters are intentionally initialized to weight=1 and bias=0
    by PyTorch's default initializer, so they are excluded here. All other
    parameters (embeddings, linear projections, FFN weights) must be genuinely
    random: not all zeros and not a single constant value.
    """
    model = _make_model(seed=7)

    # Map each parameter name to its owning module so LayerNorm params can be
    # identified by type rather than by brittle name matching.
    param_owner = {}
    for mname, module in model.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            param_owner[f"{mname}.{pname}" if mname else pname] = module

    for name, p in model.named_parameters():
        assert p.numel() > 0
        if isinstance(param_owner.get(name), nn.LayerNorm):
            continue
        assert torch.any(p != 0.0), f"{name} is all zeros"
        assert torch.any(p != p.flatten()[0]), f"{name} is constant"


def test_output_shape() -> None:
    """Forward pass must return logits of shape (b, s, vocab_size)."""
    model = _make_model(seed=0)
    b, s, vocab = 4, 16, model.vocab_size
    input_ids = torch.randint(0, vocab, (b, s))
    targets = torch.randint(0, vocab, (b, s))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (b, s, vocab)
    assert loss is not None
    assert loss.shape == ()


def test_outputs_are_finite() -> None:
    """Logits and loss must contain no NaN or Inf values."""
    model = _make_model(seed=3)
    b, s = 2, 8
    input_ids = torch.randint(0, model.vocab_size, (b, s))
    targets = torch.randint(0, model.vocab_size, (b, s))

    logits, loss = model(input_ids, targets=targets)

    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
    assert torch.isfinite(loss), "Loss is NaN/Inf"


def test_causal_masking() -> None:
    """Position i must not attend to positions j > i.

    Changing the token at a later position must leave earlier positions' logits
    unchanged. If attention were non-causal, earlier positions would be
    influenced by the later change.
    """
    model = _make_model(seed=5)
    s = 6
    ids = torch.randint(0, model.vocab_size, (1, s))
    ids_modified = ids.clone()
    ids_modified[0, -1] = (ids_modified[0, -1] + 1) % model.vocab_size

    logits_before, _ = model(ids)
    logits_after, _ = model(ids_modified)

    # torch.equal (exact, not allclose) is appropriate here: weights are frozen
    # between the two calls, dropout is disabled, and the forward pass is
    # deterministic on both CPU and CUDA for a fixed input. An exact match is
    # therefore a meaningful guarantee, and it still fails if causal masking
    # is removed (earlier positions would then be influenced by the change).
    # Running the comparison on CPU makes the determinism portable.
    before = logits_before[0].cpu()
    after = logits_after[0].cpu()

    # Earlier positions (0..s-2) must be identical; only the last may change.
    assert torch.equal(before[:-1], after[:-1]), (
        "Earlier positions changed when a later token was modified"
    )
    assert not torch.equal(before[-1], after[-1]), (
        "Modified token's logits did not change"
    )


def test_d_model_must_be_divisible_by_n_heads() -> None:
    """The constructor must reject d_model values not divisible by n_heads.

    Uses the actual constructor arguments so the test exercises the real
    validation path in CausalSelfAttention rather than a hardcoded number.
    """
    with pytest.raises(ValueError):
        _make_model(seed=0, d_model=321, n_heads=5)


def test_parameter_count_in_target_band() -> None:
    """Parameter count must match an independent architectural calculation
    and fall within the 10M-30M target band.

    The expected count is derived here from the model's own constructor
    arguments rather than copied from a magic number, so a silent architecture
    change is caught instead of merely assumed.
    """
    model = _make_model(seed=0)
    count = model.num_parameters()

    vocab, d_model, n_layers, d_ff, max_seq_len = (
        model.vocab_size,
        model.d_model,
        model.n_layers,
        model.d_ff,
        model.max_seq_len,
    )
    per_block = (
        4 * d_model              # two LayerNorms (weight + bias each)
        + 3 * d_model * d_model  # qkv projection
        + d_model * d_model      # output projection
        + 2 * d_model * d_ff     # feed-forward up and down
    )
    expected = (
        vocab * d_model          # token embedding
        + max_seq_len * d_model  # positional embedding
        + n_layers * per_block   # transformer blocks
        + 2 * d_model            # final LayerNorm
        # lm_head is tied to token_emb, so it contributes zero
    )

    assert count == expected, (
        f"Parameter count {count} does not match the architectural "
        f"expectation {expected}"
    )
    assert 10_000_000 <= count <= 30_000_000, (
        f"Parameter count {count} is outside the 10M-30M target band"
    )


def test_tied_lm_head_reduces_parameters() -> None:
    """LM head weight must be tied to the token embedding."""
    model = _make_model(seed=0)
    assert model.lm_head.weight is model.token_emb.weight


def test_tied_weight_is_random_and_shared() -> None:
    """The tied LM-head weight must be the same tensor as the token embedding
    and must carry a seed-dependent random value, not a constant.

    This guards against the shared tensor being silently overwritten or left
    at a default value during initialization.
    """
    m1 = _make_model(seed=11)
    m2 = _make_model(seed=12)

    assert m1.lm_head.weight is m1.token_emb.weight
    assert m2.lm_head.weight is m2.token_emb.weight

    assert not torch.equal(m1.token_emb.weight, m2.token_emb.weight), (
        "Tied weight did not change between seeds"
    )


def test_sequence_length_guard() -> None:
    """Sequences longer than max_seq_len must raise a ValueError."""
    model = _make_model(seed=0)
    too_long = torch.randint(0, model.vocab_size, (1, model.max_seq_len + 1))
    with pytest.raises(ValueError):
        model(too_long)
