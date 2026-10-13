"""ExpC (natural task): GSM8K greedy generation at fixed loop counts t, accuracy per t,
regression set (correct at some t' < t, wrong at t), automatic detail-error tagging, and
a CSV of random items for manual annotation (see `python -m loopdetail agreement`)."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .. import gsm8k
from ..utils import iter_jsonl_glob, mark_done, shard_done, write_jsonl
from .common import RunContext

ANNOTATION_COLUMNS = ["item_id", "t", "question", "generation", "pred", "gold", "correct",
                      "auto_detail_error", "auto_unexplained_operands",
                      "human_detail_error", "human_notes"]


def make_prompt(ctx: RunContext, question: str, shots: list[dict]) -> str:
    if ctx.chat:
        return ctx.adapter.tok.chat_prompt(gsm8k.CHAT_INSTRUCTION + question)
    return gsm8k.build_prompt(question, shots)


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("C")
    ad = ctx.adapter
    loops = [int(t) for t in cfg.get("loops", [1] if ad.loop_axis == "layer" else [1, 2, 3, 4])]
    data = gsm8k.load_gsm8k(cfg.get("path"), "test")
    rng = np.random.default_rng(ctx.seed)
    picked = rng.permutation(len(data))[: int(cfg.get("n_items", 300))]
    shots = []
    if not ctx.chat and int(cfg.get("n_shots", 4)) > 0:
        train = gsm8k.load_gsm8k(cfg.get("train_path", cfg.get("path")), "train")
        shots = [train[int(i)] for i in rng.permutation(len(train))[: int(cfg.get("n_shots", 4))]]
    out = ctx.exp_dir("C")
    stop = ad.tok.stop_ids
    for t in loops:
        shard = out / "rows" / f"t{t}.jsonl"
        if shard_done(shard):
            continue
        rows = []
        for k, i in enumerate(picked):
            item = data[int(i)]
            ids = ctx.tensor(ad.tok.encode(make_prompt(ctx, item["question"], shots), add_special=not ctx.chat))
            gen_ids = ad.generate_greedy(ids, t, int(cfg.get("max_new_tokens", 512)), stop,
                                         [] if ctx.chat else ["\nQuestion:"])
            text = ad.tok.decode(gen_ids).split("\nQuestion:")[0]
            pred, gold = gsm8k.extract_answer(text), gsm8k.gold_answer(item["answer"])
            tags = gsm8k.tag_operands(item["question"], text)
            rows.append({"item_id": f"gsm8k-test-{int(i)}", "t": t, "question": item["question"],
                         "generation": text, "n_generated": len(gen_ids), "pred": pred, "gold": gold,
                         "correct": int(pred is not None and abs(pred - gold) < 1e-6), **tags})
            if (k + 1) % 20 == 0:
                ctx.logger.info("C t=%d %d/%d acc so far %.3f", t, k + 1, len(picked),
                                np.mean([r["correct"] for r in rows]))
        write_jsonl(shard, rows)
        mark_done(shard, ctx.meta("C", {"t": t}))
    rows = list(iter_jsonl_glob(out / "rows", "t*.jsonl"))
    summary = summarize_c(rows)
    write_annotation_csv(rows, out / "annotation_sample.csv", int(cfg.get("annotate_n", 100)), ctx.seed)
    ctx.write_summary("C", summary)
    return summary


def summarize_c(rows: list[dict]) -> dict:
    by_t: dict[int, list[dict]] = defaultdict(list)
    by_item: dict[str, dict[int, int]] = defaultdict(dict)
    for r in rows:
        by_t[r["t"]].append(r)
        by_item[r["item_id"]][r["t"]] = r["correct"]
    per_t = {t: {"n": len(rs), "acc": float(np.mean([r["correct"] for r in rs])),
                 "detail_error_rate": float(np.mean([r["detail_error"] for r in rs])),
                 "detail_error_rate_among_wrong": float(np.mean([r["detail_error"] for r in rs if not r["correct"]]))
                 if any(not r["correct"] for r in rs) else None}
             for t, rs in sorted(by_t.items())}
    regressions = {t: sorted(item for item, res in by_item.items()
                             if res.get(t) == 0 and any(res.get(s) == 1 for s in res if s < t))
                   for t in sorted(by_t)}
    return {"per_t": per_t, "regression_set": regressions,
            "regression_counts": {t: len(v) for t, v in regressions.items()}}


def write_annotation_csv(rows: list[dict], path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 7)
    pick = [rows[int(i)] for i in rng.permutation(len(rows))[:n]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        w.writeheader()
        for r in pick:
            w.writerow({"item_id": r["item_id"], "t": r["t"], "question": r["question"],
                        "generation": r["generation"], "pred": r["pred"], "gold": r["gold"],
                        "correct": r["correct"], "auto_detail_error": r["detail_error"],
                        "auto_unexplained_operands": " ".join(map(str, r["unexplained_operands"])),
                        "human_detail_error": "", "human_notes": ""})


def agreement(csv_path: str) -> dict:
    """Cohen's kappa between auto_detail_error and the filled-in human_detail_error (0/1)."""
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["human_detail_error"].strip() in ("0", "1")]
    if not rows:
        return {"error": "no rows with human_detail_error in {0,1}"}
    a = np.array([int(r["auto_detail_error"]) for r in rows])
    h = np.array([int(r["human_detail_error"]) for r in rows])
    po = float((a == h).mean())
    pe = float(a.mean() * h.mean() + (1 - a.mean()) * (1 - h.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": len(rows), "percent_agreement": po, "cohen_kappa": kappa,
            "auto_rate": float(a.mean()), "human_rate": float(h.mean())}
