"""ExpB (behavioural, main): per-loop forced-choice accuracy and (calibrated) log-prob
margin for detail vs gist queries, t over the model's `loops_eval`."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..analysis.stats import wilson_ci
from ..scoring import candidate_token_ids, choice_stats, score_loops
from ..stimuli.render import INFO_TYPES, render
from ..utils import iter_jsonl_glob, mark_done, shard_done, write_jsonl
from .common import RunContext


def item_rows(ctx: RunContext, inst: dict, loops: list[int], method: str) -> list[dict]:
    chat_tok = ctx.adapter.tok if ctx.chat else None
    extrap = int(ctx.spec.get("extrapolation_from", 10**9))
    rows = []
    for info in INFO_TYPES:
        with_ctx = ctx.prepare(render(inst, info, chat_tok=chat_tok))
        no_ctx = ctx.prepare(render(inst, info, with_context=False, chat_tok=chat_tok))
        cand = candidate_token_ids(ctx.adapter.tok, with_ctx.candidates)
        lp = score_loops(ctx.adapter, with_ctx.ids, cand, loops, method)
        lp0 = score_loops(ctx.adapter, no_ctx.ids, cand, loops, method)
        for t in loops:
            st = choice_stats(lp[t], with_ctx.answer, lp0[t])
            st0 = choice_stats(lp0[t], with_ctx.answer)
            rows.append({
                "item_id": inst["id"], "context_id": inst["id"], "family": inst["family"],
                "N": inst["N"], "distance": inst["distance"], "query_pos": inst["query_idx"],
                "query_quartile": inst["query_quartile"], "info_type": info, "t": t,
                "extrapolated": t >= extrap, "answer": with_ctx.answer,
                "multi_token": any(len(c) > 1 for c in cand), **st,
                "nocontext_correct": st0["correct"],
                "logp": lp[t].round(5).tolist(), "logp_noctx": lp0[t].round(5).tolist(),
                "prompt_tokens": len(with_ctx.ids),
            })
    return rows


def summarize_b(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["info_type"], r["N"], r["t"])].append(r)
    table = []
    for (fam, info, N, t), rs in sorted(groups.items()):
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        table.append({
            "family": fam, "info_type": info, "N": N, "t": t, "n": len(rs), "acc": k / len(rs),
            "acc_ci95": [lo, hi], "acc_cal": float(np.mean([r["correct_cal"] for r in rs])),
            "margin": float(np.mean([r["margin"] for r in rs])),
            "margin_cal": float(np.mean([r["margin_cal"] for r in rs])),
            "nocontext_acc": float(np.mean([r["nocontext_correct"] for r in rs])),
            "extrapolated": rs[0]["extrapolated"], "chance": 0.25,
        })
    return {"table": table}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("B")
    loops = [int(t) for t in cfg.get("loops", ctx.loops_eval())]
    method = cfg.get("readout", "capture") if ctx.adapter.loop_axis == "loop" else "set"
    out = ctx.exp_dir("B")
    for fam, N, insts in ctx.cells("B"):
        shard = out / "rows" / f"{fam}_N{N}.jsonl"
        if shard_done(shard):
            ctx.logger.info("B %s N=%d already done", fam, N)
            continue
        rows = []
        for i, inst in enumerate(insts):
            rows += item_rows(ctx, inst, loops, method)
            if (i + 1) % 25 == 0:
                ctx.logger.info("B %s N=%d %d/%d", fam, N, i + 1, len(insts))
        write_jsonl(shard, rows)
        mark_done(shard, ctx.meta("B", {"shard": shard.name}))
    summary = summarize_b(list(iter_jsonl_glob(out / "rows", "*.jsonl")))
    summary["readout"] = method
    summary["loops"] = loops
    ctx.write_summary("B", summary)
    return summary
