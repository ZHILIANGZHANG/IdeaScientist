"""Exp0 sanity checks; writes sanity_report.json. Other experiments refuse to run unless
the overall status is PASS (override with --force).

S1 readout equivalence (looped): set-loop-count readout vs capture-based readout.
S2 published-trend reproduction: per-loop ARC accuracy should rise from t=1 to the
   trained loop count (needs HF `datasets`; skip with --skip-repro).
S3 patch no-op (fp32): patching every loop/layer with its OWN k/v reproduces logits.
S4 probe sanity: current-token identity decodable from V at loop 1 / layer 0.
S5 no-context baseline at chance.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..analysis.probes import probe_cell
from ..models.base import KVPatch
from ..scoring import candidate_token_ids, choice_stats, score_loops
from ..stimuli.generate import base_family
from ..stimuli.render import INFO_TYPES, render
from ..utils import read_json, write_json
from .common import RunContext
from .exp_a1 import extract_features


def _items(ctx: RunContext, n: int) -> list[dict]:
    """Round-robin over the (family, level) cells of the TWO shortest levels (sanity checks
    do not need long contexts)."""
    st = ctx.stimuli()
    out: list[dict] = []
    short = sorted({lv for _, lv in st}, key=lambda name: ctx.level(name)["N"])[:2]
    keys = sorted((k for k in st if k[1] in short), key=lambda k: (ctx.level(k[1])["N"], k[0]))
    i = 0
    while len(out) < n and i < max(len(v) for v in st.values()):
        for k in keys:
            if i < len(st[k]) and len(out) < n:
                out.append(st[k][i])
        i += 1
    return out


def s1_readout(ctx: RunContext, items: list[dict], loops: list[int]) -> dict[str, Any]:
    ad = ctx.adapter
    if ad.loop_axis != "loop":
        return {"status": "N/A", "reason": "non-looped model"}
    diffs, agree, choice_agree = [], [], []
    chat_tok = ad.tok if ctx.chat else None
    for inst in items:
        for info in INFO_TYPES:
            prep = ctx.prepare(render(inst, info, chat_tok=chat_tok))
            ids = ctx.tensor(prep.ids)
            cap = ad.readout_loops(ids, loops)
            cand = candidate_token_ids(ad.tok, prep.candidates)
            single = [c[0] for c in cand] if all(len(c) == 1 for c in cand) else None
            for t in loops:
                ref = ad.readout_logits(ids, t)
                diffs.append(float((ref - cap[t]).abs().max()))
                agree.append(int(ref[0].argmax() == cap[t][0].argmax()))
                if single is not None:
                    choice_agree.append(int(ref[0, single].argmax() == cap[t][0, single].argmax()))
    a, c = float(np.mean(agree)), float(np.mean(choice_agree))
    return {"status": "PASS" if a >= 0.99 and c >= 0.99 else "FAIL", "n": len(agree),
            "max_abs_logit_diff": max(diffs), "argmax_agreement": a, "candidate_choice_agreement": c,
            "criterion": "argmax and candidate-choice agreement >= 0.99"}


def s3_patch_noop(ctx: RunContext, items: list[dict]) -> dict[str, Any]:
    ad = ctx.adapter
    T = ad.n_loops_trained if ad.loop_axis == "loop" else 1
    orig_dtype = next(ad.model.parameters()).dtype
    ad.model.float()  # bf16 -> fp32 -> bf16 round trip is exact
    try:
        worst, diff_input = 0.0, []
        for inst in items:
            prep = ctx.prepare(render(inst, "detail"))
            ids = ctx.tensor(prep.ids)
            cap = ad.capture(ids, T, prep.ctx_pos, what=("k", "v"), store_dtype=None, store_device=ad.device)
            base = ad.readout_logits(ids, T)
            for t in range(1, T + 1):
                vals = {(t, l, k): cap.get(k)[t - 1, l] for l in range(ad.n_layers) for k in ("k", "v")}
                out = ad.patch_kv(ids, T, KVPatch(positions=prep.ctx_pos, values=vals))
                worst = max(worst, float((out - base).abs().max()))
            shifted = {(1, l, k): cap.get(k)[0, l].roll(1, dims=0) for l in range(ad.n_layers) for k in ("k", "v")}
            diff_input.append(float((ad.patch_kv(ids, T, KVPatch(prep.ctx_pos, shifted)) - base).abs().max()))
    finally:
        ad.model.to(orig_dtype)
    ok = worst <= 1e-4 and min(diff_input) > 1e-4
    return {"status": "PASS" if ok else "FAIL", "T": T, "max_abs_logit_diff_noop": worst,
            "min_abs_logit_diff_when_patching_other_values": min(diff_input),
            "criterion": "no-op diff <= 1e-4 (fp32) and a real patch changes logits"}


def s4_probe(ctx: RunContext, cfg: dict) -> dict[str, Any]:
    """Item identity at the item token from V at loop 1 / layer 0 (a near-linear function of
    the token embedding), on the smallest box level with N >= 16 so items recur."""
    st = ctx.stimuli()
    cells = sorted((k for k in st if base_family(k[0]) == "box"), key=lambda k: ctx.level(k[1])["N"])
    big = [k for k in cells if ctx.level(k[1])["N"] >= 16] or cells[-1:]
    n_ctx = int(cfg.get("s4_contexts", 40))
    insts = [i for k in big[:1] for i in st[k]][:n_ctx]
    data = extract_features(ctx, insts, [1], [0], ["v"], int(cfg.get("s4_facts_per_context", 8)), ctx.seed)
    meta = data["meta"]
    X = data["features"]["v"][:, 0, 0].float().numpy()
    groups = np.array([m["context_id"] for m in meta])
    uniq = np.unique(groups)
    test = set(uniq[: max(1, len(uniq) // 5)])
    test_mask = np.array([g in test for g in groups])
    labels = np.array([m["detail"] for m in meta])
    res = probe_cell(X, labels, labels, groups, test_mask, [0.1, 1.0, 10.0], 3, ctx.seed, n_boot=200)
    thr = float(cfg.get("s4_min_acc", 0.9))
    return {"status": "PASS" if res["acc"] >= thr else "FAIL", "acc": res["acc"],
            "n_classes": res["n_classes"], "n_test": res["n_test"], "n_train": res["n_train"],
            "criterion": f"item identity from V at loop 1 / layer 0 at the item token >= {thr}"}


def s5_nocontext(ctx: RunContext, items: list[dict], loops: list[int]) -> dict[str, Any]:
    ad = ctx.adapter
    chat_tok = ad.tok if ctx.chat else None
    hits: dict[tuple[str, str, int], list[int]] = {}
    for inst in items:
        for info in INFO_TYPES:
            prep = ctx.prepare(render(inst, info, with_context=False, chat_tok=chat_tok))
            lp = score_loops(ad, prep.ids, candidate_token_ids(ad.tok, prep.candidates), loops,
                             "capture" if ad.loop_axis == "loop" else "set")
            for t in loops:
                hits.setdefault((inst["family"], info, t), []).append(choice_stats(lp[t], prep.answer)["correct"])
    table, ok = [], True
    for (fam, info, t), h in sorted(hits.items()):
        acc, n = float(np.mean(h)), len(h)
        tol = max(0.1, 2.58 * math.sqrt(0.25 * 0.75 / n))
        good = abs(acc - 0.25) <= tol
        ok &= good
        table.append({"family": fam, "info_type": info, "t": t, "n": n, "acc": acc, "tolerance": tol, "ok": good})
    return {"status": "PASS" if ok else "FAIL", "table": table,
            "criterion": "|acc - 0.25| <= max(0.10, 2.58*SE) for every (family, info_type, t)"}


def s2_repro(ctx: RunContext, cfg: dict) -> dict[str, Any]:
    from .repro import run_repro

    return run_repro(ctx, list(cfg.get("repro_tasks", ["arc_easy", "arc_challenge"])),
                     int(cfg.get("repro_items", 500)))


def run(ctx: RunContext, skip_repro: bool = False) -> dict[str, Any]:
    cfg = ctx.exp_cfg("sanity")
    ad = ctx.adapter
    items = _items(ctx, int(cfg.get("n_items", 40)))
    loops = ctx.loops_eval()
    checks: dict[str, Any] = {}
    checks["S1_readout_equivalence"] = s1_readout(ctx, items[: int(cfg.get("s1_items", 20))], loops)
    ctx.logger.info("S1 %s", checks["S1_readout_equivalence"]["status"])
    if skip_repro or not ctx.spec.get("repro") or ad.loop_axis != "loop":
        checks["S2_published_trend"] = {"status": "SKIPPED",
                                        "reason": "--skip-repro" if skip_repro else "no repro target for this model"}
    else:
        checks["S2_published_trend"] = s2_repro(ctx, cfg)
    ctx.logger.info("S2 %s", checks["S2_published_trend"]["status"])
    checks["S3_patch_noop_fp32"] = s3_patch_noop(ctx, items[: int(cfg.get("s3_items", 3))])
    ctx.logger.info("S3 %s", checks["S3_patch_noop_fp32"]["status"])
    checks["S4_probe_current_token"] = s4_probe(ctx, cfg)
    ctx.logger.info("S4 %s", checks["S4_probe_current_token"]["status"])
    checks["S5_nocontext_chance"] = s5_nocontext(ctx, items, loops)
    ctx.logger.info("S5 %s", checks["S5_nocontext_chance"]["status"])
    statuses = [c["status"] for c in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else "PASS"
    report = {"meta": ctx.meta("sanity"), "overall": overall, "checks": checks}
    write_json(ctx.exp_dir("sanity") / "sanity_report.json", report)
    return report


def assert_sanity_passed(ctx: RunContext) -> None:
    path = ctx.out_dir / "sanity" / "sanity_report.json"
    if ctx.force:
        return
    if not path.exists():
        raise RuntimeError(f"{path} missing: run `--exp sanity` first (or pass --force)")
    if read_json(path)["overall"] != "PASS":
        raise RuntimeError(f"sanity checks did not PASS ({path}); fix them or pass --force")

