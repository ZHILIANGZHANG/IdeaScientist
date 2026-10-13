"""Matplotlib PNG figures. Categorical colours follow a fixed slot order
(slot 1 blue, slot 2 orange, slot 3 aqua, slot 4 yellow); heatmaps use one sequential hue."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INFO_COLOR = {"detail": SLOTS[0], "gist": SLOTS[1]}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _level_order(rows: list[dict]) -> list[str]:
    """Levels sorted by (N, fillers_per_fact, name)."""
    seen = {r["level"]: (r["N"], r.get("fillers_per_fact", 0), r["level"]) for r in rows}
    return sorted(seen, key=lambda k: seen[k])


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def accuracy_curves(table: list[dict], out: Path, axis_name: str, title: str) -> list[Path]:
    """ExpB: accuracy vs t, detail vs gist, one panel per level, one figure per family."""
    if len({r["t"] for r in table}) == 1:
        return accuracy_bars(table, out, title)
    paths = []
    for fam in sorted({r["family"] for r in table}):
        rows = [r for r in table if r["family"] == fam]
        levels = _level_order(rows)
        fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.0), sharey=True, squeeze=False)
        for ax, level in zip(axes[0], levels):
            for info in ("detail", "gist"):
                rs = sorted((r for r in rows if r["level"] == level and r["info_type"] == info), key=lambda r: r["t"])
                if not rs:
                    continue
                t = [r["t"] for r in rs]
                ax.plot(t, [r["acc"] for r in rs], "-o", color=INFO_COLOR[info], lw=2, ms=4, label=info)
                ax.fill_between(t, [r["acc_ci95"][0] for r in rs], [r["acc_ci95"][1] for r in rs],
                                color=INFO_COLOR[info], alpha=0.15, lw=0)
                ext = [r["t"] for r in rs if r.get("extrapolated")]
                if ext:
                    ax.axvspan(min(ext) - 0.5, max(ext) + 0.5, color=GRID, alpha=0.5, lw=0)
            ax.axhline(0.25, color=MUTED, ls="--", lw=1)
            ax.set_title(level, fontsize=9, color=INK)
            ax.set_xlabel(axis_name, fontsize=8, color=MUTED)
            if max(r["t"] for r in rows) > 8:
                ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.02)
            _style(ax)
        axes[0][0].set_ylabel("forced-choice accuracy", fontsize=8, color=MUTED)
        axes[0][-1].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{title} - {fam} (dashed: chance; grey: extrapolated loops)", fontsize=9, color=INK)
        fig.tight_layout()
        p = out / f"B_accuracy_{fam}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def accuracy_bars(table: list[dict], out: Path, title: str) -> list[Path]:
    """ExpB for a single loop count (non-looped references): detail vs gist bars per N."""
    paths = []
    for fam in sorted({r["family"] for r in table}):
        rows = [r for r in table if r["family"] == fam]
        Ns = _level_order(rows)
        fig, ax = plt.subplots(figsize=(1.4 * len(Ns) + 2.5, 3.0))
        for k, info in enumerate(("detail", "gist")):
            rs = {r["level"]: r for r in rows if r["info_type"] == info}
            x = np.arange(len(Ns)) + (k - 0.5) * 0.38
            acc = np.array([rs[N]["acc"] for N in Ns])
            err = np.array([[rs[N]["acc"] - rs[N]["acc_ci95"][0] for N in Ns],
                            [rs[N]["acc_ci95"][1] - rs[N]["acc"] for N in Ns]])
            ax.bar(x, acc, 0.34, color=INFO_COLOR[info], label=info)
            ax.errorbar(x, acc, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        ax.axhline(0.25, color=MUTED, ls="--", lw=1)
        ax.set_xticks(range(len(Ns)), Ns, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("forced-choice accuracy", fontsize=8, color=MUTED)
        _style(ax)
        ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"{title} - {fam}, single forward (dashed: chance; bars: 95% Wilson CI)", fontsize=9, color=INK)
        fig.tight_layout()
        p = out / f"B_accuracy_{fam}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def probe_heatmaps(rows: list[dict], out: Path, title: str) -> list[Path]:
    """ExpA1: (loop x layer) heatmaps of probe accuracy and selectivity."""
    paths = []
    keys = sorted({(r["family"], r["what"]) for r in rows})
    for fam, what in keys:
        sub = [r for r in rows if r["family"] == fam and r["what"] == what]
        loops = sorted({r["loop"] for r in sub})
        layers = sorted({r["layer"] for r in sub})
        fig, axes = plt.subplots(2, 2, figsize=(max(5, 0.3 * len(layers) + 3), 1.2 * len(loops) + 3.5), squeeze=False)
        for i, task in enumerate(("detail", "gist")):
            for j, metric in enumerate(("acc", "selectivity")):
                M = np.full((len(loops), len(layers)), np.nan)
                for r in sub:
                    if r["task"] == task:
                        M[loops.index(r["loop"]), layers.index(r["layer"])] = r[metric]
                ax = axes[i][j]
                im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0 if metric == "acc" else None,
                               vmax=1 if metric == "acc" else None, origin="lower")
                ax.set_xticks(range(len(layers)), layers, fontsize=6)
                ax.set_yticks(range(len(loops)), loops, fontsize=7)
                ax.set_xlabel("layer", fontsize=8, color=MUTED)
                ax.set_ylabel("loop", fontsize=8, color=MUTED)
                ax.set_title(f"{task}: {metric}", fontsize=9, color=INK)
                fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(f"{title} - probes on {what} at fact positions ({fam})", fontsize=9, color=INK)
        fig.tight_layout()
        p = out / f"A1_probe_{fam}_{what}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def sensitivity_curves(table: list[dict], loops: list[int], layers: list[int], looped: bool,
                       out: Path, title: str) -> list[Path]:
    """ExpA2: mean normalised L2 distance (base vs edit) per variant; x = loop (at the last
    captured layer) for looped models, x = layer for non-looped."""
    paths = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in table:
        groups[(r["family"], r["rep"])].append(r)
    for (fam, rep), rs in sorted(groups.items()):
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, squeeze=False)
        for i, info in enumerate(("detail", "gist")):
            for j, site in enumerate(("fact", "query")):
                ax = axes[i][j]
                for k, variant in enumerate(("detail_swap", "gist_swap", "unrelated_swap")):
                    m = [r for r in rs if r["info_type"] == info and r["site"] == site and r["variant"] == variant]
                    if not m:
                        continue
                    arr = np.array(m[0]["mean_l2ln"])
                    x, y = (loops, arr[:, -1]) if looped else (layers, arr[0])
                    ax.plot(x, y, "-o", color=SLOTS[k], lw=2, ms=3, label=variant)
                ax.set_title(f"{info} query, {site} position", fontsize=9, color=INK)
                ax.set_xlabel("loop" if looped else "layer", fontsize=8, color=MUTED)
                _style(ax)
        axes[0][0].set_ylabel("normalised L2 (base vs edit)", fontsize=8, color=MUTED)
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{title} - counterfactual sensitivity ({fam}, {rep})", fontsize=9, color=INK)
        fig.tight_layout()
        p = out / f"A2_sensitivity_{fam}_{rep}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def patch_effects(table: list[dict], out: Path, title: str) -> list[Path]:
    """ExpH3: accuracy per condition and patch target (loop sets), detail vs gist."""
    paths = []
    for fam in sorted({r["family"] for r in table}):
        rows = [r for r in table if r["family"] == fam and (r["target"].startswith("loop") or r["target"] in ("all", "-"))]
        targets = sorted({r["target"] for r in rows if r["target"] != "-"}, key=lambda s: (s == "all", s))
        conds = ["same_loop1", "diff_loop1", "noise"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, squeeze=False)
        for ax, info in zip(axes[0], ("detail", "gist")):
            base = [r for r in rows if r["info_type"] == info and r["condition"] == "none"]
            width = 0.8 / len(conds)
            for k, c in enumerate(conds):
                ys = []
                for t in targets:
                    m = [r for r in rows if r["info_type"] == info and r["condition"] == c and r["target"] == t]
                    ys.append(m[0]["acc"] if m else np.nan)
                ax.bar(np.arange(len(targets)) + (k - 1) * width, ys, width * 0.9, color=SLOTS[k], label=c)
            if base:
                ax.axhline(base[0]["acc"], color=INK, lw=1.5, label="none (unpatched)")
            ax.axhline(0.25, color=MUTED, ls="--", lw=1)
            ax.set_xticks(range(len(targets)), targets, fontsize=8)
            ax.set_title(f"{info}", fontsize=9, color=INK)
            ax.set_ylim(0, 1.02)
            _style(ax)
        axes[0][0].set_ylabel("accuracy", fontsize=8, color=MUTED)
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{title} - K/V patching at context positions ({fam})", fontsize=9, color=INK)
        fig.tight_layout()
        p = out / f"H3_patch_{fam}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
