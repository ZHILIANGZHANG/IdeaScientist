"""Linear probes: L2-regularised multinomial logistic regression.

Backends
  torch   (default) full-batch L-BFGS; ALL (CV fold x C) problems of a cell are fitted
          jointly in one batched optimisation (independent objectives, so the joint optimum
          is each problem's optimum); runs on CUDA when available.
  sklearn fallback: sklearn.linear_model.LogisticRegression(lbfgs), one fit per (fold, C).
Both minimise the same objective, mean(CE) + ||W||^2 / (2 C n_train) (sklearn's
C*sum(CE) + ||W||^2/2 divided by C*n_train; intercept unpenalised), on features
standardised with TRAIN statistics only. C is chosen by group K-fold CV on the train
split only (groups = context instances), from the SAME grid for every cell.

Also: Hewitt & Liang control task (secondary), context-level bootstrap CIs, and
online-code MDL (Voita & Titov 2020; first block coded uniformly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class ProbeResult:
    acc: float
    C: float
    cv_acc: dict[float, float]
    test_correct: np.ndarray  # per test sample 0/1


def _standardize(Xtr: torch.Tensor, Xte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_logreg_batched(Xs: torch.Tensor, y: torch.Tensor, masks: torch.Tensor, n_classes: int,
                       Cs: Sequence[float], max_iter: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly fit P = len(masks) x len(Cs) problems.
    Xs [P_x, n, D] standardised features (P_x = len(masks): one standardisation per mask),
    masks [P_x, n] bool training rows. Returns W [P_x, nC, D, K], b [P_x, nC, K]."""
    F, n, D = Xs.shape
    nC = len(Cs)
    W = torch.zeros(F, nC, D, n_classes, device=Xs.device, requires_grad=True)
    b = torch.zeros(F, nC, n_classes, device=Xs.device, requires_grad=True)
    m = masks.float()
    n_tr = m.sum(1)  # [F]
    lam = torch.tensor([1.0 / c for c in Cs], device=Xs.device)[None, :] / n_tr[:, None]  # [F, nC]
    onehot = torch.nn.functional.one_hot(y, n_classes).float()
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        logits = torch.einsum("fnd,fcdk->fcnk", Xs, W) + b[:, :, None, :]
        nll = -(torch.log_softmax(logits, -1) * onehot).sum(-1)  # [F, nC, n]
        ce = (nll * m[:, None, :]).sum(-1) / n_tr[:, None]
        loss = (ce + 0.5 * lam * W.pow(2).sum((2, 3))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    parts = np.array_split(uniq, min(k, len(uniq)))
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def _probe_torch(Xtr, ytr, gtr, Xte, yte, n_classes, Cs, folds, seed, device) -> ProbeResult:
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr, dtype=torch.long, device=device)
    fold_idx = group_folds(gtr, folds, seed)
    masks = torch.ones(len(fold_idx), len(ytr), dtype=torch.bool, device=device)
    for f, val in enumerate(fold_idx):
        masks[f, val] = False
    Xs = torch.stack([_standardize(X[mk], X)[1] for mk in masks])  # per-fold train statistics
    W, b = fit_logreg_batched(Xs, y, masks, n_classes, Cs)
    pred = (torch.einsum("fnd,fcdk->fcnk", Xs, W) + b[:, :, None, :]).argmax(-1)  # [F, nC, n]
    hits = ((pred == y) & ~masks[:, None, :]).sum((0, 2)).float().cpu().numpy()
    cv_acc = hits / len(ytr)
    best = int(np.argmax(cv_acc))
    a, v = _standardize(X, torch.as_tensor(Xte, dtype=torch.float32, device=device))
    W, b = fit_logreg_batched(a[None], y, torch.ones(1, len(ytr), dtype=torch.bool, device=device),
                              n_classes, [Cs[best]])
    p = (v @ W[0, 0] + b[0, 0]).argmax(-1).cpu().numpy()
    return ProbeResult(acc=float((p == yte).mean()), C=float(Cs[best]),
                       cv_acc={float(c): float(x) for c, x in zip(Cs, cv_acc)},
                       test_correct=(p == yte).astype(int))


def _sk_fit(X, y, C):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=2000, tol=1e-8).fit(X, y)


def _probe_sklearn(Xtr, ytr, gtr, Xte, yte, n_classes, Cs, folds, seed) -> ProbeResult:
    def std(a, b):
        mu, sd = a.mean(0), np.maximum(a.std(0, ddof=1), 1e-6)
        return (a - mu) / sd, (b - mu) / sd

    hits = np.zeros(len(Cs))
    for val in group_folds(gtr, folds, seed):
        tr = np.ones(len(ytr), bool)
        tr[val] = False
        a, v = std(Xtr[tr], Xtr[val])
        for i, C in enumerate(Cs):
            hits[i] += (_sk_fit(a, ytr[tr], C).predict(v) == ytr[val]).sum()
    cv_acc = hits / len(ytr)
    best = int(np.argmax(cv_acc))
    a, v = std(Xtr, Xte)
    p = _sk_fit(a, ytr, Cs[best]).predict(v)
    return ProbeResult(acc=float((p == yte).mean()), C=float(Cs[best]),
                       cv_acc={float(c): float(x) for c, x in zip(Cs, cv_acc)},
                       test_correct=(p == yte).astype(int))


def probe(Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
          n_classes: int, Cs: Sequence[float], folds: int, seed: int, device: str = "cpu",
          backend: str = "torch") -> ProbeResult:
    if backend == "torch":
        return _probe_torch(Xtr, ytr, gtr, Xte, yte, n_classes, Cs, folds, seed, device)
    if backend == "sklearn":
        return _probe_sklearn(Xtr, ytr, gtr, Xte, yte, n_classes, Cs, folds, seed)
    raise ValueError(f"probe backend must be 'torch' or 'sklearn', got {backend!r}")


def control_labels(types: np.ndarray, n_classes: int, seed: int) -> np.ndarray:
    """Hewitt & Liang control task: a fixed random label per word type."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(types)
    table = dict(zip(uniq, rng.integers(0, n_classes, size=len(uniq))))
    return np.array([table[t] for t in types])


def online_code_mdl(X: np.ndarray, y: np.ndarray, n_classes: int, C: float, seed: int,
                    fractions: Sequence[float] = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                                                  0.0625, 0.125, 0.25, 0.5, 1.0),
                    device: str = "cpu") -> dict[str, float]:
    """Online (prequential) codelength in bits; first block coded uniformly."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    cuts = sorted({max(n_classes, int(round(f * len(y)))) for f in fractions} | {len(y)})
    cuts = [c for c in cuts if c <= len(y)]
    bits = cuts[0] * math.log2(n_classes)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        a, v = _standardize(Xt[:lo], Xt[lo:hi])
        W, b = fit_logreg_batched(a[None], yt[:lo], torch.ones(1, lo, dtype=torch.bool, device=device),
                                  n_classes, [C])
        logits = v @ W[0, 0] + b[0, 0]
        bits += float(torch.nn.functional.cross_entropy(logits, yt[lo:hi], reduction="sum")) / math.log(2)
    uniform = len(y) * math.log2(n_classes)
    return {"mdl_bits": bits, "uniform_bits": uniform, "compression": uniform / bits}


def probe_cell(X: np.ndarray, labels: np.ndarray, types: np.ndarray, groups: np.ndarray,
               test_mask: np.ndarray, Cs: Sequence[float], folds: int, seed: int,
               n_boot: int = 1000, device: str = "cpu", mdl: bool = False,
               backend: str = "torch") -> dict[str, Any]:
    """Task probe (+ MDL) and the secondary control probe for one cell."""
    from .stats import bootstrap_ci

    classes, y = np.unique(labels, return_inverse=True)
    K = len(classes)
    tr = ~test_mask
    te = test_mask & np.isin(y, y[tr])  # a test label never seen in training cannot be decoded
    res = probe(X[tr], y[tr], groups[tr], X[te], y[te], K, Cs, folds, seed, device, backend)
    yc = control_labels(types, K, seed + 1)
    ctrl = probe(X[tr], yc[tr], groups[tr], X[te], yc[te], K, Cs, folds, seed, device, backend)
    out = {
        "acc": res.acc, "C": res.C, "n_classes": K,
        "majority_baseline": float(np.bincount(y[te], minlength=K).max() / max(1, te.sum())),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_test_dropped_unseen_label": int(test_mask.sum() - te.sum()),
        "acc_ci95": list(bootstrap_ci(res.test_correct, groups[te], n_boot, seed)),
        "control_acc": ctrl.acc, "selectivity": res.acc - ctrl.acc,  # secondary (see module doc)
    }
    if mdl:
        out["mdl"] = online_code_mdl(X[tr], y[tr], K, res.C, seed, device=device)
    return out
