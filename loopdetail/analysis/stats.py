"""Statistics: Wilson CIs, exact McNemar, bootstrap, mixed-effects / GEE logistic models."""
from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import numpy as np
from scipy import stats as sps


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes a, b (same items)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * sps.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p": p}


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 1000,
                 seed: int = 0, stat=np.mean, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat(values)`, resampling `groups` (clusters) if given."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if groups is None:
        idx_sets = [np.array([i]) for i in range(len(values))]
    else:
        uniq = np.unique(groups)
        idx_sets = [np.flatnonzero(groups == g) for g in uniq]
    m = len(idx_sets)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        boots[b] = stat(values[np.concatenate([idx_sets[i] for i in pick])])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def loop_info_interaction(df, loop_col: str = "lt", group_col: str = "context_id") -> dict[str, Any]:
    """correct ~ lt * info_type + (1 | context). Primary: statsmodels
    BinomialBayesMixedGLM (variational Bayes; 95% interval = mean +- 1.96 posterior SD).
    Fallback (or if VB fails): GEE logistic with exchangeable correlation clustered by
    context (robust SEs). info_type is coded detail=1, gist=0, so a NEGATIVE interaction
    means accuracy falls (or rises less) with loops for detail than for gist."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.copy()
    d["detail"] = (d["info_type"] == "detail").astype(int)
    formula = f"correct ~ {loop_col} * detail"
    term = f"{loop_col}:detail"
    out: dict[str, Any] = {"formula": formula + f" + (1|{group_col})", "n": int(len(d)),
                           "n_groups": int(d[group_col].nunique())}
    if d["correct"].nunique() < 2:
        out["error"] = "outcome has no variance"
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BinomialBayesMixedGLM.from_formula(formula, {"ctx": f"0 + C({group_col})"}, d)
            res = model.fit_vb()
        names = list(model.exog_names)
        i = names.index(term)
        mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        out["mixed_vb"] = {"coef": mean, "sd": sd, "ci95": [mean - 1.96 * sd, mean + 1.96 * sd],
                           "fe": dict(zip(names, map(float, res.fe_mean)))}
    except Exception as e:  # noqa: BLE001 - report and fall back
        out["mixed_vb_error"] = repr(e)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee = smf.gee(formula, groups=group_col, data=d, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = gee.conf_int().loc[term].tolist()
        out["gee"] = {"coef": float(gee.params[term]), "se": float(gee.bse[term]),
                      "ci95": [float(ci[0]), float(ci[1])], "p": float(gee.pvalues[term])}
    except Exception as e:  # noqa: BLE001
        out["gee_error"] = repr(e)
    return out


def loop_trend(df, loop_col: str = "lt", group_col: str = "context_id") -> dict[str, Any]:
    """Per-info-type GEE logistic slope of correct on lt (clustered by context)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    out = {}
    for info, d in df.groupby("info_type"):
        if d["correct"].nunique() < 2:
            out[info] = {"error": "no variance"}
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.gee(f"correct ~ {loop_col}", groups=group_col, data=d,
                          family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = res.conf_int().loc[loop_col].tolist()
        out[info] = {"slope": float(res.params[loop_col]), "ci95": [float(ci[0]), float(ci[1])],
                     "p": float(res.pvalues[loop_col])}
    return out


def adjacent_mcnemar(df, item_col: str = "item_id") -> list[dict[str, Any]]:
    """Exact McNemar between adjacent loop counts, per (family, info_type, level)."""
    out = []
    for (fam, info, level), d in df.groupby(["family", "info_type", "level"]):
        ts = sorted(d["t"].unique())
        piv = d.pivot_table(index=item_col, columns="t", values="correct")
        for t0, t1 in zip(ts[:-1], ts[1:]):
            pair = piv[[t0, t1]].dropna()
            r = mcnemar_exact(pair[t0].values, pair[t1].values)
            out.append({"family": fam, "info_type": info, "level": level, "t_from": int(t0), "t_to": int(t1),
                        "acc_from": float(pair[t0].mean()), "acc_to": float(pair[t1].mean()), **r})
    return out
