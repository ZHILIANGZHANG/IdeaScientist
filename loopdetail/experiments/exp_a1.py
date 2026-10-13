"""ExpA1: linear probes on K / V / hidden at FACT positions for every (loop, layer).

Features: the context alone (header + facts) is run once per instance; for the queried
fact plus up to `facts_per_context - 1` other facts we take the last token of the fact
value (box: the item token; codes: the last code token).
Labels:  box   detail = exact item, gist = category
         codes detail = last digit,  gist = first digit (thousand range)
Split by context instance (train/test), C by group-CV on train only, H&L control task.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..analysis.probes import probe_cell
from ..stimuli.render import render_context_only
from ..utils import iter_jsonl_glob, mark_done, parse_layer_selector, shard_done, write_jsonl
from .common import RunContext


def fact_labels(inst: dict, j: int) -> dict[str, str]:
    v = inst["values"][j]
    if inst["family"] == "box":
        return {"detail": v, "gist": inst["categories"][j], "type": v}
    return {"detail": v[-1], "gist": v[0], "type": v[-1]}


def extract_features(ctx: RunContext, insts: list[dict], loops: list[int], layers: list[int],
                     what: list[str], facts_per_context: int, seed: int) -> dict:
    T = max(loops)
    rng = np.random.default_rng(seed)
    feats: dict[str, list[torch.Tensor]] = defaultdict(list)
    meta = []
    for inst in insts:
        prep = ctx.prepare(render_context_only(inst))
        others = [j for j in range(inst["N"]) if j != inst["query_idx"]]
        rng.shuffle(others)
        chosen = [inst["query_idx"]] + others[: facts_per_context - 1]
        cap = ctx.adapter.capture(ctx.tensor(prep.ids), T, [prep.fact_pos[j] for j in chosen],
                                  what=what, loops=loops, layers=layers)
        for w in what:
            x = cap.get(w)  # [loops, layers, P, ...]
            feats[w].append(x.reshape(len(loops), len(layers), len(chosen), -1).permute(2, 0, 1, 3))
        for j in chosen:
            meta.append({"context_id": str(inst["id"]), "N": int(inst["N"]), "fact_idx": int(j),
                         "queried": bool(j == inst["query_idx"]),
                         **{k: str(v) for k, v in fact_labels(inst, j).items()}})
    return {"features": {w: torch.cat(v) for w, v in feats.items()}, "meta": meta,
            "loops": loops, "layers": layers}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("A1")
    ad = ctx.adapter
    loops = [int(t) for t in cfg.get("loops", ctx.spec.get("probe_loops", [1]))]
    layers = parse_layer_selector(cfg.get("layers", "all"), ad.n_layers)
    what = list(cfg.get("what", ["k", "v", "hidden"]))
    out = ctx.exp_dir("A1")
    device = "cuda" if torch.cuda.is_available() and cfg.get("probe_device", "auto") != "cpu" else "cpu"
    for fam in cfg.get("families", ctx.profile["families"]):
        feat_path = out / "features" / f"{fam}.pt"
        cells = [c for c in ctx.cells("A1") if c[0] == fam]
        if not feat_path.exists():
            parts = [extract_features(ctx, insts, loops, layers, what, int(cfg["facts_per_context"]),
                                      ctx.seed + N) for _, N, insts in cells]
            data = {"features": {w: torch.cat([p["features"][w] for p in parts]) for w in what},
                    "meta": [m for p in parts for m in p["meta"]], "loops": loops, "layers": layers}
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = feat_path.with_suffix(".tmp")
            torch.save(data, tmp)
            tmp.replace(feat_path)  # atomic: a crashed run never leaves a truncated file
            ctx.logger.info("A1 %s: features %s", fam, {w: tuple(v.shape) for w, v in data["features"].items()})
        rows_path = out / "rows" / f"{fam}.jsonl"
        if shard_done(rows_path):
            continue
        rows = probe_family(ctx, torch.load(feat_path), fam, cfg, device)
        write_jsonl(rows_path, rows)
        mark_done(rows_path, ctx.meta("A1", {"family": fam}))
    all_rows = list(iter_jsonl_glob(out / "rows", "*.jsonl"))
    summary = {"rows": len(all_rows), "loops": loops, "layers": layers, "what": what,
               "best_cells": _best_cells(all_rows)}
    ctx.write_summary("A1", summary)
    return summary


def probe_family(ctx: RunContext, data: dict, fam: str, cfg: dict, device: str) -> list[dict]:
    meta = data["meta"]
    groups = np.array([m["context_id"] for m in meta])
    uniq = np.unique(groups)
    rng = np.random.default_rng(ctx.seed)
    test_ctx = set(rng.choice(uniq, size=max(1, int(round(len(uniq) * float(cfg.get("test_frac", 0.2))))),
                              replace=False))
    test_mask = np.array([g in test_ctx for g in groups])
    types = np.array([m["type"] for m in meta])
    Cs = [float(c) for c in cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0])]
    rows = []
    for w, X in data["features"].items():
        X = X.float().numpy()  # [n, loops, layers, dim]
        for li, t in enumerate(data["loops"]):
            for lj, layer in enumerate(data["layers"]):
                for task in ("detail", "gist"):
                    labels = np.array([m[task] for m in meta])
                    res = probe_cell(X[:, li, lj], labels, types, groups, test_mask, Cs,
                                     int(cfg.get("cv_folds", 5)), ctx.seed,
                                     int(cfg.get("bootstrap", 1000)), device, bool(cfg.get("mdl", False)))
                    rows.append({"family": fam, "what": w, "loop": t, "layer": layer, "task": task, **res})
        ctx.logger.info("A1 %s %s probes done", fam, w)
    return rows


def _best_cells(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["family"], r["what"], r["task"], r["loop"])
        if key not in best or r["acc"] > best[key]["acc"]:
            best[key] = r
    return [{k: r[k] for k in ("family", "what", "task", "loop", "layer", "acc", "selectivity")}
            for r in best.values()]
