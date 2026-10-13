"""`python -m loopdetail inspect --model X`: load a model, summarise its module tree and
verify every structural assumption the adapter relies on, printing PASS/FAIL lines."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch
from torch import nn

from .base import KVPatch, LoopAdapter


def module_summary(model: nn.Module, depth: int = 3) -> list[str]:
    lines = []
    for name, mod in model.named_modules():
        d = name.count(".") + 1 if name else 0
        if d > depth:
            continue
        kids = list(mod.children())
        kinds = Counter(type(k).__name__ for k in kids)
        extra = f" children={dict(kinds)}" if isinstance(mod, nn.ModuleList) else ""
        lines.append(f"{'  ' * d}{name or '<root>'}: {type(mod).__name__}{extra}")
    return lines


def _check(results: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


@torch.no_grad()
def inspect_adapter(ad: LoopAdapter, T: int | None = None, n_tokens: int = 16,
                    check_generation: bool = True) -> list[dict[str, Any]]:
    print("\n".join(module_summary(ad.model)))
    print("adapter:", ad.describe())
    T = (ad.n_loops_trained if ad.loop_axis == "loop" else 1) if T is None else T
    ids = torch.tensor([ad.tok.encode("Box AB contains the apple. Box CD contains the hammer. Box AB")[:n_tokens]],
                       device=ad.device)
    results: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    fns = []
    for si, site in enumerate(ad.kv_sites):
        fns.append((site.module, lambda m, i, o, k=f"recurrent L{site.layer} {'/'.join(site.slices)}": counts.__setitem__(k, counts[k] + 1)))
    for group, mods in (("prelude", ad.prelude), ("coda", ad.coda)):
        for li, mod in enumerate(mods):
            fns.append((mod, lambda m, i, o, k=f"{group} L{li}": counts.__setitem__(k, counts[k] + 1)))
    if ad.loop_state_module is not None:
        fns.append((ad.loop_state_module, lambda m, i, o: counts.__setitem__("loop_state", counts["loop_state"] + 1)))
    try:
        with ad._hooks(fns):
            ad._run(ids, T)
        run_ok = True
    except Exception as e:  # noqa: BLE001 - report as a failed check
        run_ok = False
        _check(results, "forward", False, repr(e))
    if run_ok:
        rec = {k: v for k, v in counts.items() if k.startswith("recurrent")}
        _check(results, f"each recurrent k/v projection called exactly T={T} times",
               bool(rec) and all(v == T for v in rec.values()), str(rec))
        once = {k: v for k, v in counts.items() if k.startswith(("prelude", "coda"))}
        if ad.prelude or ad.coda:
            _check(results, "prelude/coda layers called exactly once",
                   all(v == 1 for v in once.values()) and len(once) == len(ad.prelude) + len(ad.coda), str(once))
        if ad.loop_state_module is not None and ad.loop_axis == "loop":
            _check(results, f"loop_state module called T={T} times", counts["loop_state"] == T,
                   f"{counts['loop_state']} calls")
        if ad.loop_axis == "loop":
            loops = sorted({1, max(1, T // 2), T})
            cap = ad.readout_loops(ids, loops)
            diffs = {t: float((ad.readout_logits(ids, t) - cap[t]).abs().max()) for t in loops}
            agree = all(int(ad.readout_logits(ids, t)[0].argmax()) == int(cap[t][0].argmax()) for t in loops)
            _check(results, "set-loop-count readout == capture-based readout (argmax)", agree,
                   f"max |logit diff| per t: {diffs}")
            if T > 1:
                d1 = float((ad.readout_logits(ids, 1) - ad.readout_logits(ids, T)).abs().max())
                _check(results, "loop count changes the output", d1 > 0, f"|logits(t=1) - logits(t={T})|max = {d1:.4g}")
        pos = list(range(1, ids.shape[1]))
        cap = ad.capture(ids, T, pos, what=("k", "v"), store_dtype=None, store_device=ad.device)
        vals = {(t, l, k): cap.get(k)[t - 1, l] for t in range(1, T + 1) for l in range(ad.n_layers) for k in ("k", "v")}
        base = ad.readout_logits(ids, T)
        diff = float((ad.patch_kv(ids, T, KVPatch(pos, vals)) - base).abs().max())
        _check(results, "patching every k/v with its own value is a no-op", diff == 0.0, f"max diff {diff:.3g}")
        if check_generation:
            stop = ad.tok.stop_ids
            native = ad.generate_greedy(ids, T, 8, stop)
            ref = LoopAdapter.generate_greedy(ad, ids, T, 8, stop)
            same = sum(a == b for a, b in zip(native, ref))
            print(f"[INFO] native generate vs reference no-cache greedy (8 tokens): {same}/{min(len(native), len(ref))} "
                  "tokens identical (Raven models draw a new random state per generated token, so small "
                  "differences are expected there)")
    return results
