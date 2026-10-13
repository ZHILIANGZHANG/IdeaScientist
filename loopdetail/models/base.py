"""Generic hook-based adapter for looped (and non-looped) decoder LMs.

Everything is batch size 1 and prefill-only (no KV cache), so hooks see exactly one call
per (loop, layer). Shared weights mean the SAME k/v projection module is called once per
loop; a per-module call counter assigns the loop index (1-based). After every hooked
forward the counters are checked against the requested loop count and a loud
`AdapterAssumptionError` is raised if the model's internals do not match the config.

Representations captured:
  k / v   : output of the layer's k / v projection (PRE-RoPE; for fused QKV the k / v
            chunk), reshaped to [kv_heads, head_dim]. Patching pre-RoPE k at identical
            positions is equivalent to patching post-RoPE K (RoPE / qk-norm / additive
            qk-bias are deterministic per (token, position)).
  hidden  : output of the decoder layer (residual stream after the layer).
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import nn

from ..tokenization import TextTokenizer

KINDS = ("k", "v")


class AdapterAssumptionError(RuntimeError):
    """A structural assumption about the wrapped model failed at runtime."""


@dataclass
class ModulePaths:
    """Dotted module paths (relative to the loaded model unless stated otherwise).

    recurrent_layers : ModuleList whose layers are looped (non-looped: all decoder layers)
    k_proj / v_proj  : path INSIDE a layer to the separate k / v Linear, or
    qkv_proj         : path INSIDE a layer to a fused QKV Linear (+ qkv_chunks_attr)
    loop_state       : module whose output (called once per loop) is the recurrent state
    prelude_layers / coda_layers : non-recurrent layer lists (checked to run once)
    final_norm / lm_head : readout modules
    """

    recurrent_layers: str
    k_proj: str | None = None
    v_proj: str | None = None
    qkv_proj: str | None = None
    qkv_chunks_attr: str | None = None
    loop_state: str | None = None
    prelude_layers: str | None = None
    coda_layers: str | None = None
    final_norm: str | None = None
    lm_head: str = "lm_head"


@dataclass
class Capture:
    """Captured tensors. k/v: [n_loops, n_layers, P, kv_heads, head_dim];
    hidden: [n_loops, n_layers, P, d_model]. `loops` are 1-based."""

    loops: list[int]
    layers: list[int]
    positions: list[int]
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    hidden: torch.Tensor | None = None

    def get(self, what: str) -> torch.Tensor:
        t = getattr(self, what)
        if t is None:
            raise KeyError(f"{what} was not captured")
        return t


@dataclass
class KVPatch:
    """Replacement values for k/v projection outputs at `positions`.
    values[(loop, layer, kind)] -> tensor [P, kv_heads, head_dim]; `heads` restricts the
    replacement to a subset of kv heads (None = all)."""

    positions: list[int]
    values: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)
    heads: list[int] | None = None


def resolve_path(root: nn.Module, path: str, field_name: str, model_name: str) -> Any:
    obj: Any = root
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError, KeyError) as e:
                raise AdapterAssumptionError(
                    f"[{model_name}] cannot index '{part}' in path '{path}' "
                    f"(config field module_paths.{field_name}): {e}"
                ) from e
            continue
        if not hasattr(obj, part):
            children = list(dict(obj.named_children()).keys())[:25] if isinstance(obj, nn.Module) else []
            raise AdapterAssumptionError(
                f"[{model_name}] module path '{path}' (config field module_paths.{field_name}) "
                f"failed at '{part}': {type(obj).__name__} has children {children}. "
                f"Fix module_paths.{field_name} in loopdetail/configs/models.yaml."
            )
        obj = getattr(obj, part)
    return obj


def _cfg(config: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        v = getattr(config, n, None)
        if v is not None:
            return v
    return default


def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


@dataclass
class _KVSite:
    module: nn.Module
    layer: int
    slices: dict[str, tuple[int, int]]  # kind -> [start, end) on the output's last dim


class LoopAdapter:
    """Base adapter. Subclasses implement `_run`, `_logits_after`, `_readout_state`."""

    loop_axis = "loop"
    default_paths: ModulePaths | None = None

    def __init__(self, model: nn.Module, tokenizer: TextTokenizer, *, name: str,
                 paths: ModulePaths, n_loops_trained: int, device: torch.device | str,
                 options: dict[str, Any] | None = None):
        self.model = model.eval()
        self.tok = tokenizer
        self.name = name
        self.paths = paths
        self.n_loops_trained = int(n_loops_trained)
        self.device = torch.device(device)
        self.options = dict(options or {})
        cfg = getattr(model, "config", None)
        self.layers: list[nn.Module] = list(self._resolve(paths.recurrent_layers, "recurrent_layers"))
        self.n_layers = len(self.layers)
        self.d_model = int(_cfg(cfg, "hidden_size", "n_embd"))
        n_heads = int(_cfg(cfg, "num_attention_heads", "n_heads"))
        self.n_kv_heads = int(_cfg(cfg, "num_key_value_heads", default=n_heads))
        self.head_dim = int(_cfg(cfg, "head_dim", default=self.d_model // n_heads))
        self.prelude = list(self._resolve(paths.prelude_layers, "prelude_layers")) if paths.prelude_layers else []
        self.coda = list(self._resolve(paths.coda_layers, "coda_layers")) if paths.coda_layers else []
        self.loop_state_module = self._resolve(paths.loop_state, "loop_state") if paths.loop_state else None
        self.kv_sites = self._build_kv_sites()

    # ------------------------------------------------------------------ setup helpers
    def _resolve(self, path: str, field_name: str) -> Any:
        return resolve_path(self.model, path, field_name, self.name)

    def _build_kv_sites(self) -> list[_KVSite]:
        kv_dim = self.n_kv_heads * self.head_dim
        sites: list[_KVSite] = []
        for li, layer in enumerate(self.layers):
            if self.paths.qkv_proj:
                mod = resolve_path(layer, self.paths.qkv_proj, "qkv_proj", self.name)
                if self.paths.qkv_chunks_attr:
                    chunks = list(resolve_path(layer, self.paths.qkv_chunks_attr, "qkv_chunks_attr", self.name))
                else:
                    chunks = [mod.out_features - 2 * kv_dim, kv_dim, kv_dim]
                if chunks[1] != kv_dim or chunks[2] != kv_dim or sum(chunks) != mod.out_features:
                    raise AdapterAssumptionError(
                        f"[{self.name}] fused QKV chunks {chunks} inconsistent with "
                        f"kv_heads*head_dim={kv_dim} / out_features={mod.out_features}; check "
                        "config num_key_value_heads/head_dim or module_paths.qkv_chunks_attr"
                    )
                q = chunks[0]
                sites.append(_KVSite(mod, li, {"k": (q, q + kv_dim), "v": (q + kv_dim, q + 2 * kv_dim)}))
            else:
                if not (self.paths.k_proj and self.paths.v_proj):
                    raise AdapterAssumptionError(f"[{self.name}] set module_paths.k_proj and v_proj (or qkv_proj)")
                for kind, p in (("k", self.paths.k_proj), ("v", self.paths.v_proj)):
                    mod = resolve_path(layer, p, f"{kind}_proj", self.name)
                    if getattr(mod, "out_features", kv_dim) != kv_dim:
                        raise AdapterAssumptionError(
                            f"[{self.name}] {p}.out_features={mod.out_features} != kv_heads*head_dim="
                            f"{kv_dim}; check config num_key_value_heads/head_dim"
                        )
                    sites.append(_KVSite(mod, li, {kind: (0, kv_dim)}))
        return sites

    # ------------------------------------------------------------------ subclass API
    def _run(self, input_ids: torch.Tensor, T: int) -> None:
        """Forward pass with exactly T loops (outputs unused; hooks observe)."""
        raise NotImplementedError

    def _logits_after(self, input_ids: torch.Tensor, T: int, positions: Sequence[int]) -> torch.Tensor:
        """Set-loop-count readout: logits [P, V] at `positions` after a T-loop forward."""
        raise NotImplementedError

    def _readout_state(self, input_ids: torch.Tensor, state: torch.Tensor,
                       positions: Sequence[int]) -> torch.Tensor:
        """Capture-based readout: logits [P, V] from the full-sequence loop state [L, d]."""
        raise NotImplementedError

    def loop_hint(self) -> str:
        """Which config field controls the loop count (for error messages)."""
        return "n/a"

    # ------------------------------------------------------------------ helpers
    def ids(self, text: str, add_special: bool = True) -> torch.Tensor:
        return torch.tensor([self.tok.encode(text, add_special=add_special)], device=self.device)

    def _check_T(self, T: int) -> None:
        if self.loop_axis == "layer" and T != 1:
            raise ValueError(f"{self.name} is non-looped; only T=1 is valid")
        if T < 1:
            raise ValueError("T must be >= 1")

    @contextlib.contextmanager
    def _hooks(self, fns: list[tuple[nn.Module, Callable]]) -> Iterator[None]:
        handles = [m.register_forward_hook(f) for m, f in fns]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _assert_counts(self, counts: dict[Any, int], expected: int, what: str) -> None:
        bad = {k: v for k, v in counts.items() if v != expected}
        if bad or not counts:
            raise AdapterAssumptionError(
                f"[{self.name}] expected each {what} to be called {expected} time(s) per forward, "
                f"observed {dict(counts) if counts else 'no calls'}. The loop count is controlled by "
                f"{self.loop_hint()}; module paths are in loopdetail/configs/models.yaml."
            )

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def readout_logits(self, input_ids: torch.Tensor, t: int, positions: Sequence[int] | None = None) -> torch.Tensor:
        """Logits [P, V] (float32) as if the model ran exactly t loops (set-loop-count)."""
        self._check_T(t)
        positions = [input_ids.shape[1] - 1] if positions is None else list(positions)
        return self._logits_after(input_ids, t, positions).float()

    @torch.no_grad()
    def readout_loops(self, input_ids: torch.Tensor, loops: Sequence[int],
                      positions: Sequence[int] | None = None) -> dict[int, torch.Tensor]:
        """Capture-based readout: ONE forward with max(loops) loops, the recurrent state is
        captured after each loop and passed through the model's readout. {t: logits [P, V]}."""
        positions = [input_ids.shape[1] - 1] if positions is None else list(positions)
        loops = sorted(set(int(t) for t in loops))
        if self.loop_axis == "layer":
            if loops != [1]:
                raise ValueError(f"{self.name} is non-looped; loops must be [1]")
            return {1: self._logits_after(input_ids, 1, positions).float()}
        if self.loop_state_module is None:
            raise AdapterAssumptionError(f"[{self.name}] module_paths.loop_state is not set")
        T = loops[-1]
        keep = set(loops)
        states: dict[int, torch.Tensor] = {}
        count = [0]

        def hook(_m, _i, out):
            count[0] += 1
            if count[0] in keep:
                states[count[0]] = _first(out)[0].detach().clone()

        with self._hooks([(self.loop_state_module, hook)]):
            self._run(input_ids, T)
        self._assert_counts({"loop_state": count[0]}, T, "loop_state module")
        return {t: self._readout_state(input_ids, states[t], positions).float() for t in loops}

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, T: int, positions: Sequence[int],
                what: Sequence[str] = ("k", "v", "hidden"), loops: Sequence[int] | None = None,
                layers: Sequence[int] | None = None, store_dtype: torch.dtype | None = torch.float16,
                store_device: str | torch.device = "cpu") -> Capture:
        """Run T loops and capture k/v projection outputs and/or layer outputs at
        `positions` for each (loop, layer). Non-looped models: T=1 (one 'loop')."""
        self._check_T(T)
        loops = list(range(1, T + 1)) if loops is None else sorted(set(loops))
        layers = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        lp_idx = {t: i for i, t in enumerate(loops)}
        ly_idx = {l: i for i, l in enumerate(layers)}
        pos = torch.tensor(list(positions), device=self.device)
        P = len(positions)
        bufs: dict[str, torch.Tensor] = {}
        for w in what:
            last = (self.n_kv_heads, self.head_dim) if w in KINDS else (self.d_model,)
            bufs[w] = torch.zeros((len(loops), len(layers), P, *last),
                                  dtype=store_dtype or torch.float32, device=store_device)
        counts: dict[tuple, int] = defaultdict(int)
        fns: list[tuple[nn.Module, Callable]] = []

        def store(w: str, t: int, layer: int, val: torch.Tensor) -> None:
            if w in bufs and t in lp_idx and layer in ly_idx:
                dt = store_dtype or val.dtype
                if bufs[w].dtype != dt:
                    bufs[w] = bufs[w].to(dt)
                bufs[w][lp_idx[t], ly_idx[layer]] = val.to(device=store_device, dtype=dt)

        for si, site in enumerate(self.kv_sites):
            def kv_hook(_m, _i, out, site=site, si=si):
                counts[("kv", si)] += 1
                t = counts[("kv", si)]
                for kind, (s, e) in site.slices.items():
                    if kind in bufs:
                        val = out[0, pos, s:e].reshape(P, self.n_kv_heads, self.head_dim)
                        store(kind, t, site.layer, val)
            fns.append((site.module, kv_hook))
        for li, layer in enumerate(self.layers):
            def h_hook(_m, _i, out, li=li):
                counts[("hidden", li)] += 1
                if "hidden" in bufs:
                    store("hidden", counts[("hidden", li)], li, _first(out)[0, pos])
            fns.append((layer, h_hook))
        with self._hooks(fns):
            self._run(input_ids, T)
        self._assert_counts(counts, T, "recurrent k/v projection and layer")
        return Capture(loops=loops, layers=layers, positions=list(positions),
                       k=bufs.get("k"), v=bufs.get("v"), hidden=bufs.get("hidden"))

    @torch.no_grad()
    def patch_kv(self, input_ids: torch.Tensor, T: int, patch: KVPatch,
                 out_positions: Sequence[int] | None = None) -> torch.Tensor:
        """T-loop forward in which k/v projection outputs at patch.positions are replaced
        for every (loop, layer, kind) in patch.values. Returns logits [P_out, V] (float32)."""
        self._check_T(T)
        out_positions = [input_ids.shape[1] - 1] if out_positions is None else list(out_positions)
        pos = torch.tensor(patch.positions, device=self.device)
        P = len(patch.positions)
        heads = None if patch.heads is None else torch.tensor(patch.heads, device=self.device)
        counts: dict[int, int] = defaultdict(int)
        applied = [0]
        fns = []
        for si, site in enumerate(self.kv_sites):
            def hook(_m, _i, out, site=site, si=si):
                counts[si] += 1
                t = counts[si]
                todo = [(k, se) for k, se in site.slices.items() if (t, site.layer, k) in patch.values]
                if not todo:
                    return None
                out = out.clone()
                for kind, (s, e) in todo:
                    new = patch.values[(t, site.layer, kind)].to(device=out.device, dtype=out.dtype)
                    cur = out[0, pos, s:e].reshape(P, self.n_kv_heads, self.head_dim)
                    if heads is not None:
                        cur = cur.clone()
                        cur[:, heads] = new[:, heads]
                        new = cur
                    out[0, pos, s:e] = new.reshape(P, -1)
                    applied[0] += 1
                return out
            fns.append((site.module, hook))
        with self._hooks(fns):
            logits = self._logits_after(input_ids, T, out_positions)
        self._assert_counts(dict(counts), T, "recurrent k/v projection")
        if applied[0] != len(patch.values):
            raise AdapterAssumptionError(
                f"[{self.name}] {len(patch.values)} patch entries requested but {applied[0]} applied; "
                "a (loop, layer) in the patch is outside the forward's range"
            )
        return logits.float()

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, t: int, max_new_tokens: int,
                        stop_ids: Sequence[int], stop_strings: Sequence[str] = ()) -> list[int]:
        """Reference greedy decoding WITHOUT KV cache (re-runs the prompt each step).
        Slow but model-agnostic; subclasses override with the model's native cached path."""
        ids = input_ids
        out: list[int] = []
        for _ in range(max_new_tokens):
            nxt = int(self.readout_logits(ids, t)[-1].argmax())
            out.append(nxt)
            if nxt in stop_ids or any(s in self.tok.decode(out[-16:]) for s in stop_strings):
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return out

    def _hf_generate_kwargs(self, stop_ids: Sequence[int], stop_strings: Sequence[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {"do_sample": False, "eos_token_id": list(stop_ids), "pad_token_id": stop_ids[0]}
        if stop_strings and hasattr(self.tok, "tok"):
            kw.update(stop_strings=list(stop_strings), tokenizer=self.tok.tok)
        return kw

    def max_context(self) -> int | None:
        """Context limit from the config (smallest of max_position_embeddings / block_size /
        n_positions; options.max_context overrides)."""
        if self.options.get("max_context"):
            return int(self.options["max_context"])
        cfg = getattr(self.model, "config", None)
        vals = [getattr(cfg, n, None) for n in ("max_position_embeddings", "block_size", "n_positions")]
        vals = [int(v) for v in vals if isinstance(v, int) and v > 0]
        return min(vals) if vals else None

    @torch.no_grad()
    def embed_at(self, input_ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        """Input-embedding vectors [P, d] at `positions` (the 'layer -1' probe baseline)."""
        if not hasattr(self.model, "get_input_embeddings"):
            raise AdapterAssumptionError(f"[{self.name}] model has no get_input_embeddings()")
        return self.model.get_input_embeddings()(input_ids)[0, list(positions)]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "adapter": type(self).__name__, "loop_axis": self.loop_axis,
            "n_loops_trained": self.n_loops_trained, "n_layers_per_loop": self.n_layers,
            "d_model": self.d_model, "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
            "n_prelude": len(self.prelude), "n_coda": len(self.coda),
            "kv_mode": "fused_qkv" if self.paths.qkv_proj else "separate_kv",
            "dtype": str(next(self.model.parameters()).dtype), "device": str(self.device),
            "max_context": self.max_context(),
        }


def patch_from_capture(src: Capture, src_loop: int, target_loops: Sequence[int],
                       target_layers: Sequence[int], kinds: Sequence[str] = KINDS,
                       heads: Sequence[int] | None = None) -> KVPatch:
    """Replace (target_loop, layer) k/v with src's values at `src_loop` for the same layer."""
    li = {l: i for i, l in enumerate(src.layers)}
    s = src.loops.index(src_loop)
    values = {}
    for t in target_loops:
        for layer in target_layers:
            for kind in kinds:
                values[(t, layer, kind)] = src.get(kind)[s, li[layer]]
    return KVPatch(positions=list(src.positions), values=values,
                   heads=None if heads is None else list(heads))


def noise_patch_like(ref: Capture, target_loops: Sequence[int], target_layers: Sequence[int],
                     generator: torch.Generator, kinds: Sequence[str] = KINDS) -> KVPatch:
    """Gaussian replacement with the per-(position, head) L2 norm of the values it replaces
    (`ref` must hold the target loops' own values)."""
    li = {l: i for i, l in enumerate(ref.layers)}
    values = {}
    for t in target_loops:
        ti = ref.loops.index(t)
        for layer in target_layers:
            for kind in kinds:
                orig = ref.get(kind)[ti, li[layer]].float()
                noise = torch.randn(orig.shape, generator=generator)
                noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                values[(t, layer, kind)] = (noise * orig.norm(dim=-1, keepdim=True)).to(ref.get(kind).dtype)
    return KVPatch(positions=list(ref.positions), values=values)
