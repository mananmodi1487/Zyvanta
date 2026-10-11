"""Scratch-initialized decoder-only Transformer for Zyvanta.

Every parameter is created from scratch by PyTorch's default initializers.
No pretrained weights, no external model files, and no ``torch.load`` calls
appear anywhere in this module.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zyvanta.config import PAD_ID


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        # Single combined projection for query, key, value.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Lower-triangular causal mask, stored as a buffer (not a parameter).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv_proj(x)  # (b, s, 3*d)
        q, k, v = qkv.chunk(3, dim=-1)

        # (b, s, h, head_dim) -> (b, h, s, head_dim)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:s, :s].unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (b, h, s, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two-layer feed-forward network with a GELU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: attention and feed-forward with residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ZyvantaLM(nn.Module):
    """Decoder-only Transformer language model, initialized from scratch.

    All dimensions are configurable through the constructor. Parameters are
    never loaded from disk; they are produced by PyTorch's default
    initializers via :meth:`apply`.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 320,
        n_layers: int = 6,
        n_heads: int = 5,
        d_ff: int = 1280,
        max_seq_len: int = 512,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie the LM head to the token embedding so the parameter count stays
        # low and the model is auditable. This is a design choice, not a
        # pretrained weight.
        #
        # This assignment happens AFTER self.apply(_init_weights) so the shared
        # tensor receives exactly one intentional random initialization (from
        # the Embedding branch) rather than being overwritten by the Linear
        # branch's traversal. The tied identity
        # ``lm_head.weight is token_emb.weight`` remains true.
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize Linear/Embedding weights from scratch (mean 0, std 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ):
        b, s = input_ids.shape
        if s > self.max_seq_len:
            raise ValueError(
                f"Sequence length {s} exceeds max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(s, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss

    def num_parameters(self) -> int:
        """Total trainable parameter count (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())
