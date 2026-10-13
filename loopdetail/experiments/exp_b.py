"""ExpB (behavioural, main): per-loop forced-choice accuracy and (calibrated) log-prob
margin for detail vs gist queries, t over the model's `loops_eval`, on the calibrated
difficulty levels.

Category-knowledge control (per item, per loop): the no-context prompt 'The zebra is a
kind of' / 'The number 4821 is in the' scored over the instance's own gist candidates.
`cat_known` = 1 if the model picks the right category at that loop. Summaries are reported
unrestricted AND restricted to rows whose category is known at that loop.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..analysis.stats import wilson_ci
from ..scoring import candidate_token_ids, choice_stats, score_loops
from ..stimuli.render import INFO_TYPES, render, render_knowledge
from ..utils import iter_jsonl_glob, mark_done, shard_done, write_jsonl
from .common import RunContext, cell_fields


def item_rows(ctx: RunContext, inst: dict, loops: list[int], method: str, knowledge: bool) -> list[dict]:
    ad = ctx.adapter
    chat_tok = ad.tok if ctx.chat else None
    extrap = int(ctx.spec.get("extrapolation_from", 10**9))
    known = {}
    if knowledge:
        kp = render_knowledge(inst)
        klp = score_loops(ad, ad.tok.encode(kp.text), candidate_token_ids(ad.tok, kp.candidates), loops, method)
        known = {t: choice_stats(klp[t], kp.answer) for t in loops}
    rows = []
    for info in INFO_TYPES:
        with_ctx = ctx.prepare(render(inst, info, chat_tok=chat_tok))
        no_ctx = ctx.prepare(render(inst, info, with_context=False, chat_tok=chat_tok))
        cand = candidate_token_ids(ad.tok, with_ctx.candidates)
        lp = score_loops(ad, with_ctx.ids, cand, loops, method)
        lp0 = score_loops(ad, no_ctx.ids, cand, loops, method)
        for t in loops:
            st = choice_stats(lp[t], with_ctx.answer, lp0[t])
            row = {**cell_fields(inst), "info_type": info, "t": t, "extrapolated": t >= extrap,
                   "answer": with_ctx.answer, "multi_token": any(len(c) > 1 for c in cand), **st,
                   "nocontext_correct": choice_stats(lp0[t], with_ctx.answer)["correct"],
                   "logp": lp[t].round(5).tolist(), "logp_noctx": lp0[t].round(5).tolist(),
                   "prompt_tokens": len(with_ctx.ids)}
            if knowledge:
                row.update({"cat_known": known[t]["correct"], "cat_known_margin": known[t]["margin"]})
            rows.append(row)
    return rows


def _table(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["info_type"], r["level"], r["t"])].append(r)
    table = []
    for (fam, info, level, t), rs in sorted(groups.items()):
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        entry = {"family": fam, "info_type": info, "level": level, "N": rs[0]["N"],
                 "fillers_per_fact": rs[0]["fillers_per_fact"], "t": t, "n": len(rs),
                 "acc": k / len(rs), "acc_ci95": [lo, hi],
                 "acc_cal": float(np.mean([r["correct_cal"] for r in rs])),
                 "margin": float(np.mean([r["margin"] for r in rs])),
                 "margin_cal": float(np.mean([r["margin_cal"] for r in rs])),
                 "nocontext_acc": float(np.mean([r["nocontext_correct"] for r in rs])),
                 "extrapolated": rs[0]["extrapolated"], "chance": 0.25}
        if "cat_known" in rs[0]:
            entry["cat_known_rate"] = float(np.mean([r["cat_known"] for r in rs]))
        table.append(entry)
    return table


def summarize_b(rows: list[dict]) -> dict:
    known = [r for r in rows if r.get("cat_known") == 1]
    return {"table": _table(rows), "table_category_known": _table(known),
            "note": "table_category_known keeps only rows whose queried item's category (box) or "
                    "thousand range (codes) the model gets right WITHOUT context at that loop"}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("B")
    loops = [int(t) for t in cfg.get("loops", ctx.loops_eval())]
    method = cfg.get("readout", "capture") if ctx.adapter.loop_axis == "loop" else "set"
    knowledge = bool(cfg.get("knowledge_control", True))
    out = ctx.exp_dir("B")
    for fam, level, insts in ctx.cells("B"):
        shard = out / "rows" / f"{fam}_{level['name']}.jsonl"
        if shard_done(shard):
            ctx.logger.info("B %s %s already done", fam, level["name"])
            continue
        rows = []
        for i, inst in enumerate(insts):
            rows += item_rows(ctx, inst, loops, method, knowledge)
            if (i + 1) % 25 == 0:
                ctx.logger.info("B %s %s %d/%d", fam, level["name"], i + 1, len(insts))
        write_jsonl(shard, rows)
        mark_done(shard, ctx.meta("B", {"shard": shard.name}))
    summary = summarize_b(list(iter_jsonl_glob(out / "rows", "*.jsonl")))
    summary.update({"readout": method, "loops": loops})
    ctx.write_summary("B", summary)
    return summary
