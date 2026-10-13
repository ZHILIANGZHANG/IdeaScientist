"""Seeded, balanced stimulus generation.

Families (same generator per base family; they differ only in how facts are RENDERED):
  box_explicit   (primary)   "Box KQ contains the zebra, a kind of animal."  - the category
                             is stated, so detail AND gist queries are pure retrieval
  box            (secondary) "Box KQ contains the zebra."  - gist needs item->category
                             world knowledge (see the knowledge control in ExpB)
  codes_explicit (primary)   "The code of Ann is 4821, in the 4000s."
  codes          (secondary) "The code of Ann is 4821."  - gist needs numeric comparison

Difficulty LEVELS (configs/experiments.yaml): N facts, `fillers` neutral sentences after
each fact (longer fact -> query distance), optional `quartiles` restricting where the
queried fact sits (quartile 0 = first quarter = farthest from the query), and `labels`
(default | confusable: box -> 3-digit labels, codes -> 4 shared first names) controlling
label interference.

Every instance carries BOTH queries (paired), three minimal-pair edits (detail-swap,
gist-swap, unrelated-swap) and a different-input control context with identical token
positions (labels, fillers and category words of the same length).
Balancing per (family, level): the factorial (detail slot x gist slot x query quartile)
is cycled in shuffled order; the queried value cycles through all values.
"""
from __future__ import annotations

import itertools
import zlib
from typing import Any

import numpy as np

from .pools import Pool
from .vocab import FILLERS, SURNAMES

FAMILIES = ("box_explicit", "codes_explicit", "box", "codes")


def base_family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    return family.split("_")[0]


def is_explicit(family: str) -> bool:
    return family.endswith("_explicit")


def level_seed(seed: int, family: str, level: dict[str, Any]) -> list[int]:
    return [seed, zlib.crc32(base_family(family).encode()), zlib.crc32(level["name"].encode())]


def _balanced_cells(rng: np.random.Generator, n: int, quartiles: list[int]) -> list[tuple[int, int, int]]:
    cells = list(itertools.product(range(4), range(4), quartiles))
    out: list[tuple[int, int, int]] = []
    while len(out) < n:
        out += [cells[i] for i in rng.permutation(len(cells))]
    return out[:n]


def _query_position(rng: np.random.Generator, N: int, quartile: int) -> int:
    lo = (N * quartile) // 4
    hi = max((N * (quartile + 1)) // 4, lo + 1)
    return int(rng.integers(lo, hi))


def _place(rng: np.random.Generator, true_value: str, foils: list[str], slot: int) -> list[str]:
    foils = [str(f) for f in foils]
    rng.shuffle(foils)
    return foils[:slot] + [str(true_value)] + foils[slot:]


def _sample(rng: np.random.Generator, items: list[str], k: int) -> list[str]:
    """k items, without replacement when the list is long enough (large N may repeat
    NON-queried values; the queried value is always unique in its context)."""
    return [str(x) for x in rng.choice(items, size=k, replace=k > len(items))]


def _label_mode(level: dict) -> str:
    mode = level.get("labels", "default")
    if mode not in ("default", "confusable"):
        raise ValueError(f"level labels must be 'default' or 'confusable', got {mode!r}")
    return mode


def _box_labels(rng: np.random.Generator, pool: Pool, level: dict, N: int) -> list[str]:
    """'default': two-letter labels ('KQ'); 'confusable': three-digit labels ('418'), which
    share digits with each other and are therefore harder to tell apart."""
    mode = _label_mode(level)
    if mode == "default":
        return [str(x) for x in rng.choice(pool.labels, size=N, replace=False)]
    return [str(x) for x in rng.choice(np.arange(100, 1000), size=N, replace=False)]


def _code_labels(rng: np.random.Generator, pool: Pool, level: dict, N: int) -> list[str]:
    """'default': distinct 'First Last'; 'confusable': only 4 first names, so many people
    share a first name and differ only by surname (interference)."""
    mode = _label_mode(level)
    firsts = pool.names if mode == "default" else list(rng.choice(pool.names, size=4, replace=False))
    full = [f"{f} {l}" for f in firsts for l in SURNAMES]
    if N > len(full):
        raise ValueError(f"N={N} exceeds the {len(full)} available '{mode}' labels")
    return [str(x) for x in rng.choice(full, size=N, replace=False)]


def _fillers(rng: np.random.Generator, N: int, k: int) -> list[list[int]]:
    return [[int(i) for i in rng.choice(len(FILLERS), size=k, replace=False)] if k else [] for _ in range(N)]


def _common(level: dict, i: int, family: str, N: int, q: int, quart: int, rng) -> dict:
    return {"family": family, "id": f"{family}-{level['name']}-{i:05d}", "level": level["name"],
            "N": N, "fillers_per_fact": int(level.get("fillers", 0)), "index": i,
            "query_idx": q, "distance": N - 1 - q, "query_quartile": quart,
            "fillers": _fillers(rng, N, int(level.get("fillers", 0)))}


def generate_box(pool: Pool, family: str, level: dict, n: int, seed: int) -> list[dict]:
    N = int(level["N"])
    rng = np.random.default_rng(level_seed(seed, family, level))
    cat_of = pool.item_category()
    all_items = sorted(cat_of)
    cats = list(pool.categories)
    cells = _balanced_cells(rng, n, list(level.get("quartiles", [0, 1, 2, 3])))
    query_cycle: list[str] = []
    out = []
    for i in range(n):
        d_slot, g_slot, quart = cells[i]
        if not query_cycle:
            query_cycle = [all_items[j] for j in rng.permutation(len(all_items))]
        q_item = query_cycle.pop()
        q_cat = cat_of[q_item]
        q = _query_position(rng, N, quart)
        ctx = _sample(rng, [it for it in all_items if it != q_item], N - 1)
        values = ctx[:q] + [q_item] + ctx[q:]
        labels = _box_labels(rng, pool, level, N)
        in_ctx = set(values)
        same_cat = [it for it in pool.categories[q_cat] if it != q_item]
        d_foils = list(rng.choice(same_cat, size=3, replace=False))
        g_foils = list(rng.choice([c for c in cats if c != q_cat], size=3, replace=False))
        other_cat = [it for it in all_items if cat_of[it] != q_cat]
        free_same = [it for it in same_cat if it not in in_ctx] or same_cat
        free_other = [it for it in other_cat if it not in in_ctx] or other_cat
        j = int(rng.choice([k for k in range(N) if k != q]))
        unrel = [it for it in all_items if cat_of[it] != cat_of[values[j]] and it != q_item]
        free_unrel = [it for it in unrel if it not in in_ctx] or unrel
        ctrl_q = str(rng.choice(other_cat))
        control_values = _sample(rng, [it for it in all_items if it not in (ctrl_q, q_item)], N - 1)
        control_values = control_values[:q] + [ctrl_q] + control_values[q:]
        for k in range(N):  # never keep the original value at any box
            if control_values[k] == values[k]:
                control_values[k] = str(rng.choice([it for it in all_items if it not in (values[k], ctrl_q)]))
        variants = {"detail_swap": {"idx": q, "value": str(rng.choice(free_same))},
                    "gist_swap": {"idx": q, "value": str(rng.choice(free_other))},
                    "unrelated_swap": {"idx": j, "value": str(rng.choice(free_unrel))}}
        used = set(values) | set(control_values) | {v["value"] for v in variants.values()} | set(d_foils)
        out.append({
            **_common(level, i, family, N, q, quart, rng),
            "labels": labels, "values": values, "query_category": q_cat,
            "category_of": {it: cat_of[it] for it in sorted(used)},
            "detail": {"candidates": _place(rng, q_item, d_foils, d_slot), "answer": d_slot,
                       "foils_in_context": int(sum(f in in_ctx for f in d_foils))},
            "gist": {"candidates": _place(rng, q_cat, g_foils, g_slot), "answer": g_slot},
            "variants": variants, "control_values": control_values,
        })
    return out


def _digit_variant(rng: np.random.Generator, code: str, positions: list[int], forbidden: set[str]) -> str:
    for _ in range(1000):
        p = int(rng.choice(positions))
        d = str(int(rng.integers(1 if p == 0 else 0, 10)))
        new = code[:p] + d + code[p + 1:]
        if d != code[p] and new not in forbidden:
            return new
    raise RuntimeError("could not find a code variant")


def generate_codes(pool: Pool, family: str, level: dict, n: int, seed: int) -> list[dict]:
    N = int(level["N"])
    rng = np.random.default_rng(level_seed(seed, family, level))
    cells = _balanced_cells(rng, n, list(level.get("quartiles", [0, 1, 2, 3])))
    lead_cycle: list[int] = []
    out = []
    for i in range(n):
        d_slot, g_slot, quart = cells[i]
        if not lead_cycle:
            lead_cycle = [int(x) for x in rng.permutation(np.arange(1, 10))]
        lead = lead_cycle.pop()
        q = _query_position(rng, N, quart)
        q_code = f"{lead}{int(rng.integers(0, 1000)):03d}"
        codes: set[str] = {q_code}
        while len(codes) < N:
            codes.add(str(int(rng.integers(1000, 10000))))
        others = sorted(codes - {q_code})
        rng.shuffle(others)
        values = others[:q] + [q_code] + others[q:]
        labels = _code_labels(rng, pool, level, N)
        in_ctx = set(values)
        d_foils: list[str] = []
        while len(d_foils) < 3:  # one NON-leading digit differs -> same thousand range
            d_foils.append(_digit_variant(rng, q_code, [1, 2, 3], {q_code, *d_foils}))
        g_foils = [f"{d}000s" for d in rng.choice([d for d in range(1, 10) if d != lead], size=3, replace=False)]
        j = int(rng.choice([k for k in range(N) if k != q]))
        ctrl: list[str] = []
        for k in range(N):
            ctrl.append(_digit_variant(rng, values[k], [0] if k == q else [0, 1, 2, 3], in_ctx | set(ctrl)))
        out.append({
            **_common(level, i, family, N, q, quart, rng),
            "labels": labels, "values": values, "query_category": f"{lead}000s",
            "detail": {"candidates": _place(rng, q_code, d_foils, d_slot), "answer": d_slot,
                       "foils_in_context": int(sum(f in in_ctx for f in d_foils))},
            "gist": {"candidates": _place(rng, f"{lead}000s", g_foils, g_slot), "answer": g_slot},
            "variants": {
                "detail_swap": {"idx": q, "value": _digit_variant(rng, q_code, [1, 2, 3], in_ctx)},
                "gist_swap": {"idx": q, "value": _digit_variant(rng, q_code, [0], in_ctx)},
                "unrelated_swap": {"idx": j, "value": _digit_variant(rng, values[j], [0], in_ctx)},
            },
            "control_values": ctrl,
        })
    return out


def generate(family: str, pool: Pool, level: dict, n: int, seed: int) -> list[dict]:
    """Instances for one (family, level). Explicit and implicit variants of a base family
    share the same seed, hence identical facts and queries (paired across families)."""
    if base_family(family) == "box":
        return generate_box(pool, family, level, n, seed)
    return generate_codes(pool, family, level, n, seed)


def category_of(inst: dict, value: str) -> str:
    if base_family(inst["family"]) == "box":
        return inst["category_of"][value]
    return f"{value[0]}000s"


def variant_values(inst: dict, variant: str) -> list[str]:
    """Fact values for a minimal-pair variant ('base' returns the original)."""
    values = list(inst["values"])
    if variant == "base":
        return values
    if variant == "control":
        return list(inst["control_values"])
    v = inst["variants"][variant]
    values[v["idx"]] = v["value"]
    return values
