"""Forced-choice scoring of candidate continuations from next-token logits.

Single-token candidates: one forward, log-softmax at the last prompt position.
Multi-token candidates: one forward per candidate (prompt ids + candidate ids, the
candidate tokenised on its own without special tokens), summed log-prob of its tokens.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from .models.base import LoopAdapter
from .tokenization import TextTokenizer

LogitsFn = Callable[[torch.Tensor, list[int]], torch.Tensor]  # (ids [1,L], positions) -> [P, V]


def candidate_token_ids(tok: TextTokenizer, candidates: Sequence[str]) -> list[list[int]]:
    out = [tok.encode(c, add_special=False) for c in candidates]
    if any(len(c) == 0 for c in out):
        raise ValueError(f"empty candidate tokenisation in {candidates}")
    return out


def _positions(n_prompt: int, cand: list[int]) -> list[int]:
    return list(range(n_prompt - 1, n_prompt - 1 + len(cand)))


def _sum_logp(logits: torch.Tensor, cand: list[int]) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(sum(lp[j, c] for j, c in enumerate(cand)))


def score_with(logits_fn: LogitsFn, prompt_ids: list[int], cand_ids: list[list[int]],
               device: torch.device) -> np.ndarray:
    """Candidate log-probs [n_cands] under an arbitrary logits function."""
    if all(len(c) == 1 for c in cand_ids):
        ids = torch.tensor([prompt_ids], device=device)
        lp = torch.log_softmax(logits_fn(ids, [len(prompt_ids) - 1])[0].float(), dim=-1)
        return np.array([float(lp[c[0]]) for c in cand_ids])
    out = []
    for cand in cand_ids:
        ids = torch.tensor([prompt_ids + cand], device=device)
        out.append(_sum_logp(logits_fn(ids, _positions(len(prompt_ids), cand)), cand))
    return np.array(out)


def score_loops(adapter: LoopAdapter, prompt_ids: list[int], cand_ids: list[list[int]],
                loops: Sequence[int], method: str = "capture") -> dict[int, np.ndarray]:
    """{t: candidate log-probs} for every loop count t. method='capture' runs ONE forward
    with max(loops) and reads out every loop; 'set' runs one forward per t."""
    loops = sorted(set(int(t) for t in loops))

    def per_input(ids: torch.Tensor, positions: list[int]) -> dict[int, torch.Tensor]:
        if method == "capture":
            return adapter.readout_loops(ids, loops, positions)
        if method == "set":
            return {t: adapter.readout_logits(ids, t, positions) for t in loops}
        raise ValueError(method)

    dev = adapter.device
    if all(len(c) == 1 for c in cand_ids):
        ids = torch.tensor([prompt_ids], device=dev)
        res = per_input(ids, [len(prompt_ids) - 1])
        out = {}
        for t, lg in res.items():
            lp = torch.log_softmax(lg[0].float(), dim=-1)
            out[t] = np.array([float(lp[c[0]]) for c in cand_ids])
        return out
    acc: dict[int, list[float]] = {t: [] for t in loops}
    for cand in cand_ids:
        ids = torch.tensor([prompt_ids + cand], device=dev)
        res = per_input(ids, _positions(len(prompt_ids), cand))
        for t, lg in res.items():
            acc[t].append(_sum_logp(lg, cand))
    return {t: np.array(v) for t, v in acc.items()}


def choice_stats(logp: np.ndarray, answer: int, logp_ref: np.ndarray | None = None) -> dict:
    """correct (argmax == answer) and margin = logp[answer] - max(other logp); with
    `logp_ref` (no-context log-probs) also the calibrated versions."""
    def margin(x: np.ndarray) -> float:
        others = np.delete(x, answer)
        return float(x[answer] - others.max())

    out = {"correct": int(np.argmax(logp) == answer), "margin": margin(logp), "pred": int(np.argmax(logp))}
    if logp_ref is not None:
        cal = logp - logp_ref
        out.update({"correct_cal": int(np.argmax(cal) == answer), "margin_cal": margin(cal)})
    return out
