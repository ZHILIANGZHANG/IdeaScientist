"""ExpH3 (causal): with T = n_loops_trained, replace context-position K/V at loop(s) t
with loop-1 K/V and measure detail / gist accuracy.

Conditions (all patch the fact-block token positions, every layer unless stated):
  none            : unpatched T-loop forward
  same_loop1      : this input's own loop-1 K/V           (H3: should RESTORE detail)
  diff_loop1      : loop-1 K/V of a different context with identical labels/positions
                    (control: must HURT, since it overwrites the facts)
  noise           : Gaussian with the per-(position, head) norm of the replaced values
Targets: each single loop in `target_loops`, and 'all' = loops 2..T. Optional sweeps
(same_loop1 only, all loops 2..T): one layer at a time; one (layer, kv-head) at a time
for a configurable subset.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..models.base import noise_patch_like, patch_from_capture
from ..scoring import candidate_token_ids, choice_stats, score_with
from ..stimuli.generate import variant_values
from ..stimuli.render import INFO_TYPES, render
from ..utils import iter_jsonl_glob, mark_done, parse_layer_selector, shard_done, write_jsonl
from .common import RunContext


def instance_rows(ctx: RunContext, inst: dict, T: int, target_loops: list[int], cfg: dict,
                  gen: torch.Generator) -> list[dict]:
    ad = ctx.adapter
    chat_tok = ad.tok if ctx.chat else None
    all_layers = list(range(ad.n_layers))
    later = list(range(2, T + 1))
    targets = [(f"loop{t}", [t]) for t in target_loops] + [("all", later)]
    rows = []
    for info in INFO_TYPES:
        prep = ctx.prepare(render(inst, info, chat_tok=chat_tok))
        ctrl = ctx.prepare(render(inst, info, values=variant_values(inst, "control"), chat_tok=chat_tok))
        if len(ctrl.ids) != len(prep.ids) or ctrl.ctx_pos != prep.ctx_pos:
            ctx.logger.warning("H3 %s: control context has different token layout; skipped", inst["id"])
            continue
        cand = candidate_token_ids(ad.tok, prep.candidates)
        self_cap = ad.capture(ctx.tensor(prep.ids), T, prep.ctx_pos, what=("k", "v"),
                              store_dtype=None, store_device=ad.device)
        ctrl_cap = ad.capture(ctx.tensor(ctrl.ids), 1, ctrl.ctx_pos, what=("k", "v"),
                              store_dtype=None, store_device=ad.device)

        def run_patch(patch):
            def fn(ids, positions):
                return ad.patch_kv(ids, T, patch, positions)
            return score_with(fn, prep.ids, cand, ad.device)

        base_lp = score_with(lambda ids, pos: ad.readout_logits(ids, T, pos), prep.ids, cand, ad.device)
        base = choice_stats(base_lp, prep.answer)
        common = {"item_id": inst["id"], "family": inst["family"], "N": inst["N"],
                  "info_type": info, "T": T, "answer": prep.answer}
        rows.append({**common, "condition": "none", "target": "-", **base, "delta_margin": 0.0})

        def add(condition: str, target: str, patch) -> None:
            st = choice_stats(run_patch(patch), prep.answer)
            rows.append({**common, "condition": condition, "target": target, **st,
                         "delta_margin": st["margin"] - base["margin"]})

        for tname, tl in targets:
            add("same_loop1", tname, patch_from_capture(self_cap, 1, tl, all_layers))
            add("diff_loop1", tname, patch_from_capture(ctrl_cap, 1, tl, all_layers))
            add("noise", tname, noise_patch_like(self_cap, tl, all_layers, gen))
        if cfg.get("layer_sweep", False):
            for layer in all_layers:
                add("same_loop1", f"layer{layer}", patch_from_capture(self_cap, 1, later, [layer]))
        head_cfg = cfg.get("head_sweep")
        if head_cfg:
            hl = parse_layer_selector(head_cfg.get("layers", "all"), ad.n_layers)
            heads = list(range(min(int(head_cfg.get("heads", ad.n_kv_heads)), ad.n_kv_heads)))
            for layer in hl:
                for h in heads:
                    add("same_loop1", f"layer{layer}_head{h}",
                        patch_from_capture(self_cap, 1, later, [layer], heads=[h]))
    return rows


def summarize_h3(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["info_type"], r["condition"], r["target"])].append(r)
    table = [{"family": f, "info_type": i, "condition": c, "target": t, "n": len(rs),
              "acc": float(np.mean([r["correct"] for r in rs])),
              "delta_margin": float(np.mean([r["delta_margin"] for r in rs]))}
             for (f, i, c, t), rs in sorted(groups.items())]
    heads = sorted((r for r in table if "_head" in r["target"]),
                   key=lambda r: (r["family"], r["info_type"], -r["delta_margin"]))
    return {"table": table, "head_ranking": [{k: r[k] for k in ("family", "info_type", "target", "delta_margin", "acc")}
                                             for r in heads]}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("H3")
    ad = ctx.adapter
    if ad.loop_axis != "loop":
        ctx.logger.info("H3 skipped: %s is not a looped model", ad.name)
        return {"skipped": "non-looped model"}
    T = int(cfg.get("T", ad.n_loops_trained))
    target_loops = [int(t) for t in cfg.get("target_loops", ctx.spec.get("h3_target_loops", range(2, T + 1)))]
    out = ctx.exp_dir("H3")
    gen = torch.Generator().manual_seed(ctx.seed)
    for fam, N, insts in ctx.cells("H3"):
        shard = out / "rows" / f"{fam}_N{N}.jsonl"
        if shard_done(shard):
            continue
        rows = []
        for i, inst in enumerate(insts):
            rows += instance_rows(ctx, inst, T, target_loops, cfg, gen)
            if (i + 1) % 10 == 0:
                ctx.logger.info("H3 %s N=%d %d/%d", fam, N, i + 1, len(insts))
        write_jsonl(shard, rows)
        mark_done(shard, ctx.meta("H3", {"shard": shard.name}))
    summary = summarize_h3(list(iter_jsonl_glob(out / "rows", "*.jsonl")))
    summary.update({"T": T, "target_loops": target_loops})
    ctx.write_summary("H3", summary)
    return summary
