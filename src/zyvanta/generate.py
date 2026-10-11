"""Text generation from a trained Zyvanta model.

Greedy and temperature sampling are supported. Generation is single-process
and CPU-friendly by default; pass ``device="cuda"`` for GPU inference.
"""

from typing import List, Optional

import torch

from zyvanta.config import BOS_ID, EOS_ID, GENERATE_MAX_NEW_TOKEN_TOKENS, GENERATE_TEMPERATURE
from zyvanta.model import ZyvantaLM
from zyvanta.tokenizer import ByteBPETokenizer


@torch.no_grad()
def generate(
    model: ZyvantaLM,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int = GENERATE_MAX_NEW_TOKEN_TOKENS,
    temperature: float = GENERATE_TEMPERATURE,
    device=None,
) -> str:
    """Generate text autoregressively from ``prompt``."""
    device = device or model.token_emb.weight.device
    model.eval()

    ids = [BOS_ID] + tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]
        logits, _ = model(input_ids)
        next_logits = logits[0, -1] / max(temperature, 1e-8)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == EOS_ID:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    generated_ids = input_ids[0].tolist()[len(ids):]
    return tokenizer.decode(generated_ids)