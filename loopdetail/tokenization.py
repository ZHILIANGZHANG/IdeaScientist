"""Thin tokenizer interface so stimuli/scoring code does not depend on HF details.

`TextTokenizer` is the protocol; `HFTokenizer` wraps a transformers tokenizer.
The fake word-level tokenizer used by tests implements the same protocol.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence


class TextTokenizer(Protocol):
    name: str

    def encode(self, text: str, add_special: bool = True) -> list[int]: ...

    def encode_offsets(self, text: str, add_special: bool = True) -> tuple[list[int], list[tuple[int, int]]]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def chat_prompt(self, user_text: str) -> str: ...

    @property
    def stop_ids(self) -> list[int]: ...

    @property
    def has_chat_template(self) -> bool: ...


class HFTokenizer:
    """Wrap a HF (fast) tokenizer. Offsets come from `return_offsets_mapping`."""

    def __init__(self, tok: Any, name: str):
        self.tok = tok
        self.name = name
        if not getattr(tok, "is_fast", False):
            raise RuntimeError(
                f"tokenizer for {name} is not a fast tokenizer; character offsets are required "
                "to locate fact positions. Install `tokenizers` / pass use_fast=True."
            )

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        return list(self.tok(text, add_special_tokens=add_special)["input_ids"])

    def encode_offsets(self, text: str, add_special: bool = True) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=add_special, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None))

    def chat_prompt(self, user_text: str) -> str:
        """Chat-formatted prompt with the generation prompt appended; thinking disabled
        where the template supports `enable_thinking` (Qwen3-style)."""
        msgs = [{"role": "user", "content": user_text}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @property
    def stop_ids(self) -> list[int]:
        ids = set()
        for attr in ("eos_token_id",):
            v = getattr(self.tok, attr, None)
            if isinstance(v, int):
                ids.add(v)
        for special in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"):
            tid = self.tok.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid != getattr(self.tok, "unk_token_id", None) and tid >= 0:
                ids.add(tid)
        return sorted(ids)


def token_index_covering(offsets: Sequence[tuple[int, int]], char_pos: int) -> int:
    """Index of the token whose character span covers `char_pos` (leading-space tokens
    such as ' apple' cover the space too)."""
    for i, (s, e) in enumerate(offsets):
        if s <= char_pos < e and e > s:
            return i
    raise ValueError(f"no token covers character {char_pos}")


def token_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> list[int]:
    """All token indices overlapping the character interval [start, end)."""
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not idx:
        raise ValueError(f"no token overlaps characters [{start},{end})")
    return idx
