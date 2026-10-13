"""ExpA2: counterfactual sensitivity with minimal pairs.

For each instance and query type, the base prompt and three single-fact edits
(detail-swap: same-category / same-thousand value; gist-swap: different category /
thousand; unrelated-swap: a NON-queried fact changes category / thousand) are run with
identical token positions. Per (representation, loop, layer, site) we record the
distance between base and edited representation at the queried FACT position and at
the QUERY's last position:
  cos  = 1 - cosine similarity
  l2ln = L2 distance after per-vector layer normalisation, divided by sqrt(dim)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..stimuli.generate import variant_values
from ..stimuli.render import INFO_TYPES, render
from ..utils import iter_jsonl_glob, mark_done, parse_layer_selector, shard_done, write_jsonl
from .common import RunContext

VARIANTS = ("detail_swap", "gist_swap", "unrelated_swap")
SITES = ("fact", "query")


def _ln(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)


def distances(a: torch.Tensor, b: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """a, b: [..., dim] -> (1 - cos, normalised L2 after layernorm) with shape [...]."""
    a, b = a.float(), b.float()
    cos = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
    l2 = (_ln(a) - _ln(b)).norm(dim=-1) / a.shape[-1] ** 0.5
    return cos.numpy(), l2.numpy()


def instance_rows(ctx: RunContext, inst: dict, loops: list[int], layers: list[int],
                  reps: list[str]) -> list[dict]:
    chat_tok = ctx.adapter.tok if ctx.chat else None
    T = max(loops)
    rows = []
    for info in INFO_TYPES:
        caps = {}
        base_len = None
        for variant in ("base",) + VARIANTS:
            prep = ctx.prepare(render(inst, info, values=variant_values(inst, variant), chat_tok=chat_tok))
            if base_len is None:
                base_len = len(prep.ids)
            elif len(prep.ids) != base_len:
                ctx.logger.warning("A2 %s %s: token length changed; skipping instance", inst["id"], variant)
                return []
            positions = [prep.fact_pos[inst["query_idx"]], len(prep.ids) - 1]
            caps[variant] = ctx.adapter.capture(ctx.tensor(prep.ids), T, positions, what=reps,
                                                loops=loops, layers=layers, store_dtype=torch.float32)
        for variant in VARIANTS:
            for rep in reps:
                base, edit = caps["base"].get(rep), caps[variant].get(rep)
                base = base.reshape(*base.shape[:3], -1)
                edit = edit.reshape(*edit.shape[:3], -1)
                cos, l2 = distances(base, edit)  # [loops, layers, 2]
                for si, site in enumerate(SITES):
                    rows.append({"item_id": inst["id"], "family": inst["family"], "N": inst["N"],
                                 "info_type": info, "variant": variant, "rep": rep, "site": site,
                                 "loops": loops, "layers": layers,
                                 "cos": cos[:, :, si].round(6).tolist(), "l2ln": l2[:, :, si].round(6).tolist()})
    return rows


def _ratio(a: np.ndarray, b: np.ndarray) -> list:
    """Elementwise a / b; NaN (null in JSON) where b ~ 0 (e.g. layer 0 at the query position,
    which no edit can reach yet)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 1e-6, a / np.where(b > 1e-6, b, 1.0), np.nan)
    return [[None if np.isnan(x) else round(float(x), 6) for x in row] for row in r]


def summarize_a2(rows: list[dict]) -> dict:
    acc: dict[tuple, list[np.ndarray]] = defaultdict(list)
    acc_cos: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        key = (r["family"], r["info_type"], r["rep"], r["site"], r["variant"])
        acc[key].append(np.array(r["l2ln"]))
        acc_cos[key].append(np.array(r["cos"]))
    mean = {k: np.mean(v, 0) for k, v in acc.items()}
    table = []
    for (fam, info, rep, site, variant), m in sorted(mean.items()):
        key = (fam, info, rep, site, variant)
        entry = {"family": fam, "info_type": info, "rep": rep, "site": site, "variant": variant,
                 "n": len(acc[key]), "mean_l2ln": m.round(6).tolist(),
                 "mean_cos": np.mean(acc_cos[key], 0).round(6).tolist()}
        if variant == "detail_swap":
            g = mean.get((fam, info, rep, site, "gist_swap"))
            u = mean.get((fam, info, rep, site, "unrelated_swap"))
            if g is not None:
                entry["ratio_detail_over_gist"] = _ratio(m, g)
            if u is not None:
                entry["ratio_detail_over_unrelated"] = _ratio(m, u)
        table.append(entry)
    return {"table": table}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("A2")
    ad = ctx.adapter
    loops = [int(t) for t in cfg.get("loops", ctx.spec.get("probe_loops", [1]))]
    layers = parse_layer_selector(cfg.get("layers", "all"), ad.n_layers)
    reps = list(cfg.get("reps", ["hidden", "v"]))
    out = ctx.exp_dir("A2")
    for fam, N, insts in ctx.cells("A2"):
        shard = out / "rows" / f"{fam}_N{N}.jsonl"
        if shard_done(shard):
            continue
        rows = []
        for inst in insts:
            rows += instance_rows(ctx, inst, loops, layers, reps)
        write_jsonl(shard, rows)
        mark_done(shard, ctx.meta("A2", {"shard": shard.name}))
        ctx.logger.info("A2 %s N=%d done (%d rows)", fam, N, len(rows))
    summary = summarize_a2(list(iter_jsonl_glob(out / "rows", "*.jsonl")))
    summary.update({"loops": loops, "layers": layers, "reps": reps})
    ctx.write_summary("A2", summary)
    return summary
