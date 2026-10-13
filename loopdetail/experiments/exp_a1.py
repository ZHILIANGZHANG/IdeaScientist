"""ExpA1: linear probes on K / V / hidden at FACT positions for every (loop, layer).

Features: the context alone (header + facts) is run once per instance; for the queried
fact plus up to `facts_per_context - 1` other facts we take the last token of the fact
value (box: the item token; codes: the last code token).
Labels:  box   detail = exact item, gist = category
         codes detail = last digit,  gist = first digit (thousand range)

Primary read-outs per cell (what, loop, layer, task):
  acc                 held-out accuracy (split by context; C by group-CV on train only)
  acc_minus_embed     acc minus the same probe on the INPUT EMBEDDING at that position
  acc_minus_l1l0      acc minus the same representation at loop 1 / layer 0
  mdl                 online code length (bits) and compression over the uniform code
Secondary: Hewitt & Liang control accuracy / selectivity. For word-type attributes probed at
that word's own position (as here), a type-memorising control decodes them too, so
selectivity is near 0 by construction and is NOT a primary measure.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..analysis.probes import probe_cell
from ..stimuli.generate import base_family
from ..stimuli.render import render_context_only
from ..utils import iter_jsonl_glob, mark_done, parse_layer_selector, shard_done, write_jsonl
from .common import RunContext


def fact_labels(inst: dict, j: int) -> dict[str, str]:
    v = inst["values"][j]
    if base_family(inst["family"]) == "box":
        return {"detail": v, "gist": inst["category_of"][v], "type": v}
    return {"detail": v[-1], "gist": v[0], "type": v[-1]}


def extract_features(ctx: RunContext, insts: list[dict], loops: list[int], layers: list[int],
                     what: list[str], facts_per_context: int, seed: int) -> dict:
    T = max(loops)
    rng = np.random.default_rng(seed)
    feats: dict[str, list[torch.Tensor]] = defaultdict(list)
    embed: list[torch.Tensor] = []
    meta = []
    for inst in insts:
        prep = ctx.prepare(render_context_only(inst))
        others = [j for j in range(inst["N"]) if j != inst["query_idx"]]
        rng.shuffle(others)
        chosen = [inst["query_idx"]] + others[: facts_per_context - 1]
        pos = [prep.fact_pos[j] for j in chosen]
        ids = ctx.tensor(prep.ids)
        cap = ctx.adapter.capture(ids, T, pos, what=what, loops=loops, layers=layers)
        for w in what:
            x = cap.get(w)  # [loops, layers, P, ...]
            feats[w].append(x.reshape(len(loops), len(layers), len(chosen), -1).permute(2, 0, 1, 3))
        embed.append(ctx.adapter.embed_at(ids, pos).to("cpu", torch.float16))
        for j in chosen:
            meta.append({"context_id": str(inst["id"]), "level": str(inst["level"]), "N": int(inst["N"]),
                         "fact_idx": int(j), "queried": bool(j == inst["query_idx"]),
                         **{k: str(v) for k, v in fact_labels(inst, j).items()}})
    return {"features": {w: torch.cat(v) for w, v in feats.items()}, "embed": torch.cat(embed),
            "meta": meta, "loops": loops, "layers": layers}


def run(ctx: RunContext) -> dict:
    cfg = ctx.exp_cfg("A1")
    ad = ctx.adapter
    loops = [int(t) for t in cfg.get("loops", ctx.spec.get("probe_loops", [1]))]
    layers = parse_layer_selector(cfg.get("layers", "all"), ad.n_layers)
    if 0 not in layers:
        layers = [0] + layers  # the loop-1 / layer-0 reference
    if 1 not in loops:
        loops = [1] + loops
    what = list(cfg.get("what", ["k", "v", "hidden"]))
    out = ctx.exp_dir("A1")
    device = "cuda" if torch.cuda.is_available() and cfg.get("probe_device", "auto") != "cpu" else "cpu"
    families = sorted({fam for fam, _, _ in ctx.cells("A1")})
    for fam in families:
        feat_path = out / "features" / f"{fam}.pt"
        if not feat_path.exists():
            parts = [extract_features(ctx, insts, loops, layers, what, int(cfg["facts_per_context"]),
                                      ctx.seed + lv["N"]) for f, lv, insts in ctx.cells("A1") if f == fam]
            data = {"features": {w: torch.cat([p["features"][w] for p in parts]) for w in what},
                    "embed": torch.cat([p["embed"] for p in parts]),
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
    ctx.write_summary("A1", {"rows": len(all_rows), "loops": loops, "layers": layers, "what": what,
                             "primary_measures": ["acc", "acc_minus_embed", "acc_minus_l1l0", "mdl"],
                             "secondary_measures": ["control_acc", "selectivity"],
                             "cells": _compact(all_rows)})
    return {"rows": len(all_rows)}


def probe_family(ctx: RunContext, data: dict, fam: str, cfg: dict, device: str) -> list[dict]:
    meta = data["meta"]
    groups = np.array([m["context_id"] for m in meta])
    uniq = np.unique(groups)
    rng = np.random.default_rng(ctx.seed)
    n_test = max(1, int(round(len(uniq) * float(cfg.get("test_frac", 0.2)))))
    test_mask = np.isin(groups, rng.choice(uniq, size=n_test, replace=False))
    types = np.array([m["type"] for m in meta])
    kw = dict(Cs=[float(c) for c in cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0])],
              folds=int(cfg.get("cv_folds", 5)), seed=ctx.seed, n_boot=int(cfg.get("bootstrap", 1000)),
              device=device, mdl=bool(cfg.get("mdl", True)), backend=cfg.get("probe_backend", "torch"))
    rows = []
    for task in ("detail", "gist"):
        labels = np.array([m[task] for m in meta])
        base = probe_cell(data["embed"].float().numpy(), labels, types, groups, test_mask, **kw)
        rows.append({"family": fam, "what": "embed", "loop": 0, "layer": -1, "task": task, **base})
        for w, X in data["features"].items():
            X = X.float().numpy()  # [n, loops, layers, dim]
            ref = None
            for li, t in enumerate(data["loops"]):
                for lj, layer in enumerate(data["layers"]):
                    res = probe_cell(X[:, li, lj], labels, types, groups, test_mask, **kw)
                    if t == 1 and layer == 0:
                        ref = res["acc"]
                    rows.append({"family": fam, "what": w, "loop": t, "layer": layer, "task": task, **res,
                                 "acc_minus_embed": res["acc"] - base["acc"]})
            for r in rows:
                if r["what"] == w and r["task"] == task:
                    r["acc_minus_l1l0"] = r["acc"] - ref
        ctx.logger.info("A1 %s %s probes done", fam, task)
    return rows


def _compact(rows: list[dict]) -> list[dict]:
    keys = ("family", "what", "task", "loop", "layer", "acc", "acc_ci95", "acc_minus_embed",
            "acc_minus_l1l0", "selectivity")
    out = []
    for r in rows:
        c = {k: r.get(k) for k in keys}
        if "mdl" in r:
            c["mdl_compression"] = r["mdl"]["compression"]
        out.append(c)
    return out
