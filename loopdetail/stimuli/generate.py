"""Seeded, balanced stimulus generation for the two task families.

Every context instance carries BOTH queries (detail + gist, paired), three minimal-pair
edits (detail-swap / gist-swap / unrelated-swap) and a different-input control context
with identical labels (so token positions match; used by ExpH3).

Balancing (per (family, N) cell): the factorial of (detail answer slot x gist answer slot x
query-position quartile) is cycled in a shuffled order, and the queried value cycles
through a shuffled list of all values (equal answer frequency).
"""
from __future__ import annotations

import itertools

import numpy as np

from .pools import Pool

FAMILIES = ("box", "codes")


def _balanced_cells(rng: np.random.Generator, n: int) -> list[tuple[int, int, int]]:
    cells = list(itertools.product(range(4), range(4), range(4)))
    out: list[tuple[int, int, int]] = []
    while len(out) < n:
        perm = rng.permutation(len(cells))
        out += [cells[i] for i in perm]
    return out[:n]


def _query_position(rng: np.random.Generator, N: int, quartile: int) -> int:
    lo, hi = (N * quartile) // 4, max((N * (quartile + 1)) // 4, (N * quartile) // 4 + 1)
    return int(rng.integers(lo, hi))


def _place(rng: np.random.Generator, true_value: str, foils: list[str], slot: int) -> list[str]:
    foils = list(foils)
    rng.shuffle(foils)
    return foils[:slot] + [true_value] + foils[slot:]


def generate_box(pool: Pool, N: int, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng([seed, N, 11])
    cat_of = pool.item_category()
    all_items = sorted(cat_of)
    cats = list(pool.categories)
    cells = _balanced_cells(rng, n)
    query_cycle: list[str] = []
    out = []
    for i in range(n):
        d_slot, g_slot, quart = cells[i]
        if not query_cycle:
            query_cycle = [all_items[j] for j in rng.permutation(len(all_items))]
        q_item = query_cycle.pop()
        q_cat = cat_of[q_item]
        q = _query_position(rng, N, quart)
        others = [it for it in all_items if it != q_item]
        ctx = list(rng.choice(others, size=N - 1, replace=False))
        values = ctx[:q] + [q_item] + ctx[q:]
        labels = list(rng.choice(pool.labels, size=N, replace=False))
        in_ctx = set(values)
        same_cat = [it for it in pool.categories[q_cat] if it != q_item]
        d_foils = list(rng.choice(same_cat, size=3, replace=False))
        g_foils = list(rng.choice([c for c in cats if c != q_cat], size=3, replace=False))
        free_same = [it for it in same_cat if it not in in_ctx]
        free_other = [it for it in all_items if cat_of[it] != q_cat and it not in in_ctx]
        j = int(rng.choice([k for k in range(N) if k != q])) if N > 1 else q
        free_unrel = [it for it in all_items if cat_of[it] != cat_of[values[j]] and it not in in_ctx]
        control = list(rng.choice([it for it in all_items if cat_of[it] != q_cat], size=1))
        rest = [it for it in all_items if it != control[0]]
        ctrl_others = list(rng.choice(rest, size=N - 1, replace=False))
        control_values = ctrl_others[:q] + control + ctrl_others[q:]
        for k in range(N):  # never keep the original value at any box
            if control_values[k] == values[k]:
                swap = next(it for it in rest if it not in control_values and it != values[k])
                control_values[k] = swap
        out.append({
            "family": "box", "id": f"box-N{N}-{i:05d}", "N": N, "index": i,
            "labels": labels, "values": values, "query_idx": q, "distance": N - 1 - q,
            "query_quartile": quart, "query_category": q_cat,
            "categories": [cat_of[v] for v in values],
            "detail": {"candidates": _place(rng, q_item, d_foils, d_slot), "answer": d_slot,
                       "foils_in_context": int(sum(f in in_ctx for f in d_foils))},
            "gist": {"candidates": _place(rng, q_cat, g_foils, g_slot), "answer": g_slot},
            "variants": {
                "detail_swap": {"idx": q, "value": str(rng.choice(free_same))},
                "gist_swap": {"idx": q, "value": str(rng.choice(free_other))},
                "unrelated_swap": {"idx": j, "value": str(rng.choice(free_unrel))},
            },
            "control_values": control_values,
        })
    return out


def _digit_variant(rng: np.random.Generator, code: str, positions: list[int],
                   forbidden: set[str]) -> str:
    for _ in range(1000):
        p = int(rng.choice(positions))
        lo = 1 if p == 0 else 0
        d = str(int(rng.integers(lo, 10)))
        if d == code[p]:
            continue
        new = code[:p] + d + code[p + 1:]
        if new not in forbidden:
            return new
    raise RuntimeError("could not find a code variant")


def generate_codes(pool: Pool, N: int, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng([seed, N, 22])
    cells = _balanced_cells(rng, n)
    lead_cycle: list[int] = []
    out = []
    for i in range(n):
        d_slot, g_slot, quart = cells[i]
        if not lead_cycle:
            lead_cycle = list(rng.permutation(np.arange(1, 10)))
        lead = int(lead_cycle.pop())
        q = _query_position(rng, N, quart)
        q_code = f"{lead}{int(rng.integers(0, 1000)):03d}"
        codes: set[str] = {q_code}
        while len(codes) < N:
            codes.add(str(int(rng.integers(1000, 10000))))
        others = sorted(codes - {q_code})
        rng.shuffle(others)
        values = others[:q] + [q_code] + others[q:]
        labels = list(rng.choice(pool.names[:max(N, 32)], size=N, replace=False))
        in_ctx = set(values)
        d_foils: list[str] = []
        while len(d_foils) < 3:  # differ in exactly one NON-leading digit -> same thousand range
            d_foils.append(_digit_variant(rng, q_code, [1, 2, 3], {q_code, *d_foils}))
        g_foils = [f"{d}000s" for d in rng.choice([d for d in range(1, 10) if d != lead], size=3,
                                                  replace=False)]
        j = int(rng.choice([k for k in range(N) if k != q])) if N > 1 else q
        ctrl = []
        used: set[str] = set()
        for k in range(N):
            positions = [0] if k == q else [0, 1, 2, 3]
            c = _digit_variant(rng, values[k], positions, in_ctx | used)
            ctrl.append(c)
            used.add(c)
        out.append({
            "family": "codes", "id": f"codes-N{N}-{i:05d}", "N": N, "index": i,
            "labels": labels, "values": values, "query_idx": q, "distance": N - 1 - q,
            "query_quartile": quart, "query_category": f"{lead}000s",
            "categories": [f"{v[0]}000s" for v in values],
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


def generate(family: str, pool: Pool, N: int, n: int, seed: int) -> list[dict]:
    if family == "box":
        return generate_box(pool, N, n, seed)
    if family == "codes":
        return generate_codes(pool, N, n, seed)
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


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
