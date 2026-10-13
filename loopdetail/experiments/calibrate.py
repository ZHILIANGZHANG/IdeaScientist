"""Difficulty calibration: pick the levels used by the main runs.

On pilot contexts (taken from the END of each cell, so they do not overlap the first-k
contexts the experiments use when contexts_per_cell >= k + pilot) at the model's trained
loop count, measure detail and gist accuracy for every (family, level). Selected levels
per family = the easiest built level (always kept) + every level whose DETAIL accuracy is
in [band_lo, band_hi]. If no level falls in the band, the level whose detail accuracy is
closest to the band centre is added (ties -> the later, harder level; flagged in `notes`). Implicit families use their
explicit sibling's selection when `share_with_explicit` (default) so the two are paired.
Writes results/<model>/calibration.json, which every experiment reads.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..scoring import candidate_token_ids, choice_stats, score_loops
from ..stimuli.generate import is_explicit
from ..stimuli.render import INFO_TYPES, render
from ..utils import write_json
from .common import CALIBRATION_FILE, RunContext, per_family


def measure(ctx: RunContext, insts: list[dict], T: int) -> dict[str, Any]:
    ad = ctx.adapter
    chat_tok = ad.tok if ctx.chat else None
    hits: dict[str, list[int]] = {i: [] for i in INFO_TYPES}
    longest = 0
    for inst in insts:
        for info in INFO_TYPES:
            prep = ctx.prepare(render(inst, info, chat_tok=chat_tok))
            longest = max(longest, len(prep.ids))
            lp = score_loops(ad, prep.ids, candidate_token_ids(ad.tok, prep.candidates), [T],
                             "capture" if ad.loop_axis == "loop" else "set")[T]
            hits[info].append(choice_stats(lp, prep.answer)["correct"])
    return {"n": len(insts), "prompt_tokens": longest,
            **{f"{i}_acc": float(np.mean(h)) for i, h in hits.items()}}


def select(rows: list[dict], level_order: list[str], lo: float, hi: float) -> tuple[list[str], str]:
    built = [r for r in sorted(rows, key=lambda r: level_order.index(r["level"]))]
    if not built:
        return [], "no level fits the context window"
    chosen = [built[0]["level"]]
    band = [r["level"] for r in built if lo <= r["detail_acc"] <= hi]
    note = ""
    if not band:  # closest to the band centre; ties go to the HARDER (later) level
        mid = (lo + hi) / 2
        best = min(built, key=lambda r: (round(abs(r["detail_acc"] - mid), 6), -level_order.index(r["level"])))
        band = [best["level"]]
        note = f"no level with detail accuracy in [{lo}, {hi}]; added the closest ({best['level']}: {best['detail_acc']:.2f})"
    chosen += [lv for lv in band if lv not in chosen]
    return chosen, note


def run(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.profile.get("calibrate", {})
    ad = ctx.adapter
    T = ad.n_loops_trained if ad.loop_axis == "loop" else 1
    lo, hi = (float(x) for x in cfg.get("band", [0.4, 0.9]))
    families = list(cfg.get("families", ctx.profile["families"]))
    order = [lv["name"] for lv in ctx.levels]
    st = ctx.stimuli()
    rows = []
    for fam in families:
        for name in order:
            if (fam, name) not in st:
                continue
            n = per_family(cfg.get("contexts_per_level"), fam, 50)
            res = measure(ctx, st[(fam, name)][-n:], T)
            lv = ctx.level(name)
            rows.append({"family": fam, "level": name, "N": lv["N"], "fillers": lv.get("fillers", 0),
                         "quartiles": lv.get("quartiles", [0, 1, 2, 3]), "T": T, **res})
            ctx.logger.info("calibrate %s %s: detail %.2f gist %.2f (%d tokens)", fam, name,
                            res["detail_acc"], res["gist_acc"], res["prompt_tokens"])
    selected, notes = {}, {}
    for fam in families:
        sel, note = select([r for r in rows if r["family"] == fam], order, lo, hi)
        selected[fam], notes[fam] = sel, note
    if cfg.get("share_with_explicit", True):
        for fam in ctx.profile["families"]:
            sib = f"{fam}_explicit"
            if not is_explicit(fam) and sib in selected:
                selected[fam] = selected[sib]
                notes[fam] = f"uses the selection of {sib}"
    out = {"meta": ctx.meta("calibrate"), "T": T, "band": [lo, hi],
           "pilot_contexts": cfg.get("contexts_per_level", 50),
           "table": rows, "selected": selected, "notes": notes,
           "skipped_levels": [f"{f}/{lv}" for f in families for lv in order if (f, lv) not in st]}
    write_json(ctx.out_dir / CALIBRATION_FILE, out)
    return out
