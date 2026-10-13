"""`python -m loopdetail analyze`: statistics + figures for whatever results exist."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils import iter_jsonl_glob, read_json, run_metadata, write_json
from . import plots
from .stats import adjacent_mcnemar, loop_info_interaction, loop_trend


def analyze_b(model_dir: Path) -> dict[str, Any]:
    import pandas as pd

    rows = list(iter_jsonl_glob(model_dir / "B" / "rows", "*.jsonl"))
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"adjacent_mcnemar": adjacent_mcnemar(df)}
    if df["t"].nunique() < 2:
        out["note"] = "single loop count: no loop statistics (non-looped reference)"
        return out
    df["lt"] = np.log2(df["t"].astype(float))
    df["lt"] -= df["lt"].mean()
    for fam, d in df.groupby("family"):
        d_in = d[~d["extrapolated"]]
        out[f"{fam}_interaction_trained_range"] = loop_info_interaction(d_in)
        out[f"{fam}_trend_trained_range"] = loop_trend(d_in)
        if d["extrapolated"].any():
            out[f"{fam}_interaction_all_loops"] = loop_info_interaction(d)
        for N, dn in d_in.groupby("N"):
            out[f"{fam}_N{N}_interaction"] = loop_info_interaction(dn)
    return out


def analyze(model_dir: Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    fig_dir = model_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = model_dir.name
    report: dict[str, Any] = {"meta": run_metadata({"model_dir": str(model_dir)}), "figures": []}
    b_sum = model_dir / "B" / "summary.json"
    if b_sum.exists():
        s = read_json(b_sum)
        looped = s["meta"]["config"]["adapter"]["loop_axis"] == "loop"
        report["B_stats"] = analyze_b(model_dir)
        report["figures"] += plots.accuracy_curves(s["table"], fig_dir, "loops t" if looped else "t", name)
    a1 = list(iter_jsonl_glob(model_dir / "A1" / "rows", "*.jsonl"))
    if a1:
        report["figures"] += plots.probe_heatmaps(a1, fig_dir, name)
    a2_sum = model_dir / "A2" / "summary.json"
    if a2_sum.exists():
        s = read_json(a2_sum)
        looped = s["meta"]["config"]["adapter"]["loop_axis"] == "loop"
        report["figures"] += plots.sensitivity_curves(s["table"], s["loops"], s["layers"], looped, fig_dir, name)
    h3_sum = model_dir / "H3" / "summary.json"
    if h3_sum.exists() and "table" in read_json(h3_sum):
        report["figures"] += plots.patch_effects(read_json(h3_sum)["table"], fig_dir, name)
    report["figures"] = [str(p.relative_to(model_dir)) for p in report["figures"]]
    write_json(model_dir / "analysis.json", report)
    return report
