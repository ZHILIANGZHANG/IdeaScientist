"""S2: per-loop ARC accuracy (lm-eval-harness style forced choice) to check the published
qualitative trend that accuracy RISES from t=1 to the trained loop count.

Prompt "Question: {q}\nAnswer:" and continuation " {choice}"; acc uses the summed
log-prob, acc_norm divides by the choice's character length (lm-eval's acc_norm).
We do not compare to exact published numbers (prompting/eval details differ); PASS means
acc_norm(t = n_loops_trained) - acc_norm(t = 1) > 2 * SE of the difference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..scoring import candidate_token_ids, score_loops
from .common import RunContext

ARC_CONFIGS = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}


def load_arc(task: str, n: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset  # optional dependency, only needed for S2

    ds = load_dataset("allenai/ai2_arc", ARC_CONFIGS[task], split="test")
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    out = []
    for i in idx:
        row = ds[int(i)]
        labels = list(row["choices"]["label"])
        out.append({"question": row["question"], "choices": list(row["choices"]["text"]),
                    "answer": labels.index(row["answerKey"])})
    return out


def score_items(ctx: RunContext, items: list[dict], loops: list[int]) -> dict[int, dict[str, float]]:
    ad = ctx.adapter
    hits = {t: [] for t in loops}
    hits_norm = {t: [] for t in loops}
    for it in items:
        prompt = ad.tok.encode(f"Question: {it['question']}\nAnswer:")
        cands = [f" {c}" for c in it["choices"]]
        lp = score_loops(ad, prompt, candidate_token_ids(ad.tok, cands), loops, "capture")
        lens = np.array([len(c) for c in cands], float)
        for t in loops:
            hits[t].append(int(np.argmax(lp[t]) == it["answer"]))
            hits_norm[t].append(int(np.argmax(lp[t] / lens) == it["answer"]))
    return {t: {"acc": float(np.mean(hits[t])), "acc_norm": float(np.mean(hits_norm[t])), "n": len(items)}
            for t in loops}


def trend_status(per_t: dict[int, dict[str, float]], t_lo: int, t_hi: int) -> tuple[str, float]:
    a, b = per_t[t_lo]["acc_norm"], per_t[t_hi]["acc_norm"]
    n = per_t[t_lo]["n"]
    se = math.sqrt(max(a * (1 - a) + b * (1 - b), 1e-9) / n)
    return ("PASS" if b - a > 2 * se else "FAIL"), b - a


def run_repro(ctx: RunContext, tasks: list[str], n: int) -> dict[str, Any]:
    ad = ctx.adapter
    loops = sorted(set(range(1, ad.n_loops_trained + 1)) & set(ctx.loops_eval())) or [1, ad.n_loops_trained]
    try:
        data = {task: load_arc(task, n, ctx.seed) for task in tasks}
    except Exception as e:  # noqa: BLE001 - dataset access is environment-dependent
        return {"status": "SKIPPED", "reason": f"could not load ARC via HF datasets: {e!r}"}
    out: dict[str, Any] = {"loops": loops, "tasks": {}}
    statuses = []
    for task, items in data.items():
        per_t = score_items(ctx, items, loops)
        status, delta = trend_status(per_t, loops[0], loops[-1])
        statuses.append(status)
        out["tasks"][task] = {"per_loop": per_t, "delta_acc_norm": delta, "status": status}
        ctx.logger.info("S2 %s: %s", task, {t: round(v["acc_norm"], 3) for t, v in per_t.items()})
    out["status"] = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    out["criterion"] = "acc_norm rises from t=1 to t=n_loops_trained by > 2 SE on every task"
    return out
