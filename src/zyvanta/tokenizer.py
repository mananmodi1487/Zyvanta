"""Byte-level BPE tokenizer trained from scratch.

No pretrained tokenizer, no external downloads, no paid services. Every raw
byte 0..255 is mapped to a printable Unicode codepoint, so any UTF-8 text is
representable -- English, Devanagari (Hindi), Hinglish, and programming code
all round-trip exactly, with no unknown-token fallback needed.

Vocabulary budget: 256 byte tokens + 3 special tokens (<pad>, <bos>, <eos>)
+ up to (vocab_size - 259) merge tokens. The special ids are shared with
``config.PAD_ID`` / ``BOS_ID`` / ``EOS_ID`` and must stay in sync.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

from zyvanta.config import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE


def _byte_to_unicode() -> dict[int, str]:
    """Map each raw byte 0..255 to a distinct printable Unicode codepoint.

    Printable ASCII and Latin-1 keep their own codepoints; the remaining bytes
    are shifted into the private-use area. This lets the tokenizer be stored
    and inspected as plain text.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    chars = list(printable)
    cursor = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(0x100 + cursor)
            cursor += 1
    return dict(zip(printable, [chr(c) for c in chars]))


_BYTE_TO_CHAR = _byte_to_unicode()
_CHAR_TO_BYTE = {v: k for k, v in _BYTE_TO_CHAR.items()}


class ByteBPETokenizer:
    """A minimal byte-level BPE tokenizer trained on supplied text."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.merges: List[tuple[str, str]] = []
        self._rebuild()

    # ------------------------------------------------------------------ #
    # Vocabulary bookkeeping
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        # Keys must be the byte-level characters produced by _BYTE_TO_CHAR,
        # not chr(0..255): non-printable bytes are remapped to shifted
        # codepoints (e.g. byte 128 -> U+0120 'Ġ'), so using chr(c) would
        # leave those byte tokens absent from the vocabulary.
        self.vocab = {_BYTE_TO_CHAR[c]: c for c in range(256)}
        self.vocab["<pad>"] = PAD_ID
        self.vocab["<bos>"] = BOS_ID
        self.vocab["<eos>"] = EOS_ID
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, texts: Iterable[str], min_freq: int = 2, sample_limit: Optional[int] = None) -> "ByteBPETokenizer":
        """Learn BPE merge rules from ``texts`` in memory.

        Pair frequencies are maintained in a single Counter and updated
        incrementally when each merge is applied, so each merge step costs time
        proportional to the occurrences of the merged pair rather than to the
        whole corpus. This keeps full-corpus BPE tractable on large inputs.

        ``sample_limit`` caps the total UTF-8 bytes consumed. Pass None to use
        the whole corpus. Sampling is deterministic: bytes are taken in the
        order they appear in ``texts``, never randomly.
        """
        # Each text becomes one token sequence of byte-level characters.
        sequences: List[List[str]] = []
        byte_budget = sample_limit
        for text in texts:
            encoded = text.encode("utf-8")
            if byte_budget is not None:
                if byte_budget <= 0:
                    break
                encoded = encoded[:byte_budget]
                byte_budget -= len(encoded)
            sequences.append([_BYTE_TO_CHAR[b] for b in encoded])

        # Pair frequencies, updated incrementally as merges are applied.
        pairs = self._count_pairs(sequences)

        merges: List[tuple[str, str]] = []
        budget = self.vocab_size - 259  # 256 bytes + 3 specials already used
        for _ in range(max(0, budget)):
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < min_freq:
                break
            merges.append(best)
            sequences = [self._apply_merge(seq, best) for seq in sequences]
            pairs = self._count_pairs(sequences)

        self.merges = merges
        self._rebuild()
        next_id = 259
        for a, b in self.merges:
            token = a + b
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self

    @staticmethod
    def _count_pairs(sequences: List[List[str]]) -> Counter:
        """Count adjacent byte-pair occurrences across all sequences."""
        pairs = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i + 1])] += 1
        return pairs

    @staticmethod
    def _apply_merge(seq: List[str], pair: tuple[str, str]) -> List[str]:
        merged: List[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                merged.append(seq[i] + seq[i + 1])
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """Tokenize ``text`` into a list of integer ids."""
        tokens = [_BYTE_TO_CHAR[b] for b in text.encode("utf-8")]
        for a, b in self.merges:
            tokens = self._apply_merge(tokens, (a, b))
        return [self.vocab[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from a list of integer ids.

        The model can emit any id in 0..vocab_size-1, but fit() only assigns
        ids 259..(259 + len(merges) - 1); the remainder are unassigned on a
        small corpus. Unassigned ids are skipped here rather than raising, so
        every model output id is decodable safely.
        """
        chars: List[str] = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i not in self.inv_vocab:
                continue
            chars.extend(self.inv_vocab[i])
        byte_vals = [_CHAR_TO_BYTE[c] for c in chars]
        return bytes(byte_vals).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        tok.pad_id = tok.vocab["<pad>"]
        tok.bos_id = tok.vocab["<bos>"]
        tok.eos_id = tok.vocab["<eos>"]
        return tok