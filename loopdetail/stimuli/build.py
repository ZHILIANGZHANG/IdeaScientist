"""Build, validate and persist stimulus sets for one tokenizer.

Output: <out>/stimuli/pool.json, meta.json, {family}_{level}.jsonl (git-ignored).
Validation per instance (raises on failure, so a bad tokenizer interaction is loud):
  * every fact value is tokenised as its own token(s) inside the context,
  * box families: prompt+candidate tokenises as prompt ids + [candidate id].
Levels whose prompt would not fit the model's context window are skipped (recorded in
meta.json under `skipped_levels`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tokenization import TextTokenizer, token_span
from ..utils import read_json, read_jsonl, run_metadata, write_json, write_jsonl
from .generate import base_family, generate
from .pools import Pool, resolve_pool
from .render import INFO_TYPES, render

CONTEXT_MARGIN = 64  # tokens kept free for the answer / candidates


def check_instance(tok: TextTokenizer, inst: dict, chat: bool) -> int:
    """Validate tokenisation; returns the longest prompt length (tokens)."""
    chat_tok = tok if chat else None
    longest = 0
    for info in INFO_TYPES:
        pr = render(inst, info, chat_tok=chat_tok)
        ids, offs = tok.encode_offsets(pr.text, add_special=not chat)
        longest = max(longest, len(ids))
        for (s, e), val in zip(pr.fact_spans, inst["values"]):
            text = tok.decode([ids[i] for i in token_span(offs, s, e)]).strip()
            if val not in text:
                raise AssertionError(f"{inst['id']}: fact '{val}' tokenised as '{text}'")
        if base_family(inst["family"]) == "box":
            for c in pr.candidates:
                full = tok.encode(pr.text + c, add_special=not chat)
                if full[:-1] != ids or len(full) != len(ids) + 1:
                    raise AssertionError(f"{inst['id']}: candidate {c!r} is not a single appended token")
    return longest


def build_stimuli(tok: TextTokenizer, out_dir: Path, families: list[str], levels: list[dict[str, Any]],
                  n_contexts: int, seed: int, chat: bool = False, n_categories: int = 8,
                  max_context: int | None = None) -> dict[tuple[str, str], list[dict]]:
    """Generate (or load, if already built with the same settings) every (family, level)."""
    stim_dir = out_dir / "stimuli"
    settings = {"families": families, "levels": levels, "n_contexts": n_contexts, "seed": seed,
                "chat": chat, "tokenizer": tok.name, "n_categories": n_categories, "max_context": max_context}
    meta_path = stim_dir / "meta.json"
    if meta_path.exists() and read_json(meta_path)["config"] == settings:
        meta = read_json(meta_path)
        return {(f, lv): read_jsonl(stim_dir / f"{f}_{lv}.jsonl") for f, lv in meta["built"]}
    pool = resolve_pool(tok, n_categories=n_categories)
    write_json(stim_dir / "pool.json", pool.to_json())
    out: dict[tuple[str, str], list[dict]] = {}
    skipped, lengths = [], {}
    for fam in families:
        for level in levels:
            insts = generate(fam, pool, level, n_contexts, seed)
            longest = check_instance(tok, insts[0], chat)
            if max_context is not None and longest + CONTEXT_MARGIN > max_context:
                skipped.append({"family": fam, "level": level["name"], "prompt_tokens": longest,
                                "max_context": max_context})
                continue
            for inst in insts[1:]:
                longest = max(longest, check_instance(tok, inst, chat))
            lengths[f"{fam}_{level['name']}"] = longest
            write_jsonl(stim_dir / f"{fam}_{level['name']}.jsonl", insts)
            out[(fam, level["name"])] = insts
    meta = run_metadata(settings)
    meta.update({"built": sorted(out), "skipped_levels": skipped, "max_prompt_tokens": lengths})
    write_json(meta_path, meta)
    return out


def load_pool(out_dir: Path) -> Pool:
    return Pool.from_json(read_json(out_dir / "stimuli" / "pool.json"))
