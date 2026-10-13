"""Linear probes: multinomial logistic regression in torch (CPU or GPU).

* features standardised with TRAIN statistics only;
* L2 strength chosen by group K-fold CV on the train split only (groups = context
  instances) from the SAME C grid for every cell; sklearn-equivalent objective
  mean(CE) + ||W||^2 / (2 C n_train);
* all C values of a fold are fitted jointly (independent problems, one L-BFGS run);
* Hewitt & Liang control task: every word TYPE gets a random label from the same label
  set; selectivity = acc(task) - acc(control);
* bootstrap CI over test contexts; optional online-code MDL (Voita & Titov 2020).
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


def fit_logreg(X: torch.Tensor, y: torch.Tensor, n_classes: int, Cs: Sequence[float],
               max_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one multinomial LR per C jointly. Returns W [nC, D, K], b [nC, K]."""
    n, D = X.shape
    nC = len(Cs)
    W = torch.zeros(nC, D, n_classes, device=X.device, requires_grad=True)
    b = torch.zeros(nC, n_classes, device=X.device, requires_grad=True)
    lam = torch.tensor([1.0 / (c * n) for c in Cs], device=X.device)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        logits = torch.einsum("nd,cdk->cnk", X, W) + b[:, None, :]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, n_classes), y.repeat(nC),
                                               reduction="none").view(nC, n).mean(1)
        loss = (ce + 0.5 * lam * W.pow(2).sum((1, 2))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _predict(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (torch.einsum("nd,cdk->cnk", X, W) + b[:, None, :]).argmax(-1)  # [nC, n]


def group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    parts = np.array_split(uniq, min(k, len(uniq)))
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def probe(Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
          n_classes: int, Cs: Sequence[float], folds: int, seed: int, device: str = "cpu",
          max_iter: int = 100) -> ProbeResult:
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    yte_t = torch.as_tensor(yte, dtype=torch.long, device=device)
    cv_hits = np.zeros(len(Cs))
    for val_idx in group_folds(gtr, folds, seed):
        tr_mask = np.ones(len(ytr), bool)
        tr_mask[val_idx] = False
        a, v = _standardize(Xtr_t[tr_mask], Xtr_t[val_idx])
        W, b = fit_logreg(a, ytr_t[tr_mask], n_classes, Cs, max_iter)
        cv_hits += (_predict(v, W, b) == ytr_t[val_idx]).float().sum(1).cpu().numpy()
    cv_acc = cv_hits / len(ytr)
    best = int(np.argmax(cv_acc))
    a, v = _standardize(Xtr_t, Xte_t)
    W, b = fit_logreg(a, ytr_t, n_classes, [Cs[best]], max_iter)
    correct = (_predict(v, W, b)[0] == yte_t).cpu().numpy().astype(int)
    return ProbeResult(acc=float(correct.mean()), C=float(Cs[best]),
                       cv_acc={float(c): float(x) for c, x in zip(Cs, cv_acc)}, test_correct=correct)


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
        W, b = fit_logreg(a, yt[:lo], n_classes, [C])
        logits = torch.einsum("nd,dk->nk", v, W[0]) + b[0]
        bits += float(torch.nn.functional.cross_entropy(logits, yt[lo:hi], reduction="sum")) / math.log(2)
    uniform = len(y) * math.log2(n_classes)
    return {"mdl_bits": bits, "uniform_bits": uniform, "compression": uniform / bits}


def probe_cell(X: np.ndarray, labels: np.ndarray, types: np.ndarray, groups: np.ndarray,
               test_mask: np.ndarray, Cs: Sequence[float], folds: int, seed: int,
               n_boot: int = 1000, device: str = "cpu", mdl: bool = False) -> dict[str, Any]:
    """Task probe + control probe for one (representation, loop, layer, task) cell."""
    from .stats import bootstrap_ci

    classes, y = np.unique(labels, return_inverse=True)
    K = len(classes)
    tr = ~test_mask
    te = test_mask & np.isin(y, y[tr])  # a test label never seen in training cannot be decoded
    res = probe(X[tr], y[tr], groups[tr], X[te], y[te], K, Cs, folds, seed, device)
    yc = control_labels(types, K, seed + 1)
    ctrl = probe(X[tr], yc[tr], groups[tr], X[te], yc[te], K, Cs, folds, seed, device)
    sel = res.test_correct.astype(float) - ctrl.test_correct.astype(float)
    out = {
        "acc": res.acc, "C": res.C, "control_acc": ctrl.acc, "control_C": ctrl.C,
        "selectivity": res.acc - ctrl.acc, "n_classes": K,
        "majority_baseline": float(np.bincount(y[te], minlength=K).max() / te.sum()),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_test_dropped_unseen_label": int(test_mask.sum() - te.sum()),
        "acc_ci95": list(bootstrap_ci(res.test_correct, groups[te], n_boot, seed)),
        "selectivity_ci95": list(bootstrap_ci(sel, groups[te], n_boot, seed)),
    }
    if mdl:
        out["mdl"] = online_code_mdl(X[tr], y[tr], K, res.C, seed, device=device)
    return out
