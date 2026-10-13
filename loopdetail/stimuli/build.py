"""Build, validate and persist stimulus sets for one tokenizer.

Output layout: <out>/stimuli/pool.json, <out>/stimuli/{family}_N{N}.jsonl (+ meta.json).
Validation per instance (raises on failure, so a bad tokenizer interaction is loud):
  * every fact value is tokenised as its own token(s) inside the context,
  * for single-token families, prompt+candidate tokenises as prompt ids + [candidate id].
"""
from __future__ import annotations

from pathlib import Path

from ..tokenization import TextTokenizer, token_span
from ..utils import read_json, read_jsonl, run_metadata, write_json, write_jsonl
from .generate import generate
from .pools import Pool, resolve_pool
from .render import INFO_TYPES, render


def check_instance(tok: TextTokenizer, inst: dict, chat: bool) -> None:
    chat_tok = tok if chat else None
    for info in INFO_TYPES:
        pr = render(inst, info, chat_tok=chat_tok)
        ids, offs = tok.encode_offsets(pr.text, add_special=not chat)
        for (s, e), val in zip(pr.fact_spans, inst["values"]):
            span = token_span(offs, s, e)
            text = tok.decode([ids[i] for i in span]).strip()
            if val not in text:
                raise AssertionError(f"{inst['id']}: fact '{val}' tokenised as '{text}'")
        if inst["family"] == "box":
            for c in pr.candidates:
                full = tok.encode(pr.text + c, add_special=not chat)
                if full[:-1] != ids or len(full) != len(ids) + 1:
                    raise AssertionError(
                        f"{inst['id']}: candidate {c!r} is not a single appended token"
                    )


def build_stimuli(tok: TextTokenizer, out_dir: Path, families: list[str], n_values: list[int],
                  n_contexts: int, seed: int, chat: bool = False,
                  n_categories: int = 8) -> dict[tuple[str, int], list[dict]]:
    """Generate (or load, if already built with the same settings) all cells."""
    stim_dir = out_dir / "stimuli"
    settings = {"families": families, "n_values": n_values, "n_contexts": n_contexts,
                "seed": seed, "chat": chat, "tokenizer": tok.name, "n_categories": n_categories}
    meta_path = stim_dir / "meta.json"
    if meta_path.exists() and read_json(meta_path)["config"] == settings:
        return {(f, N): read_jsonl(stim_dir / f"{f}_N{N}.jsonl") for f in families for N in n_values}
    pool = resolve_pool(tok, n_categories=n_categories)
    write_json(stim_dir / "pool.json", pool.to_json())
    out: dict[tuple[str, int], list[dict]] = {}
    for fam in families:
        for N in n_values:
            insts = generate(fam, pool, N, n_contexts, seed)
            for inst in insts:
                check_instance(tok, inst, chat)
            write_jsonl(stim_dir / f"{fam}_N{N}.jsonl", insts)
            out[(fam, N)] = insts
    write_json(meta_path, run_metadata(settings))
    return out


def load_pool(out_dir: Path) -> Pool:
    return Pool.from_json(read_json(out_dir / "stimuli" / "pool.json"))
