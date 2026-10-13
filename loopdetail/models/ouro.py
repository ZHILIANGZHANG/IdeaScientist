"""Ouro (ByteDance/Ouro-*; trust_remote_code `modeling_ouro.py`, transformers 4.55-era).

Relied-upon internals (see ADAPTERS.md for sources):
  * OuroForCausalLM.model is an OuroModel with attribute `total_ut_steps`; its forward loops
    the shared `layers` stack that many times, applies `norm` after EVERY loop, and returns
    the tuple (BaseModelOutputWithPast, hidden_states_list, gate_list).
  * logits for loop t are `lm_head(hidden_states_list[t-1])` (no extra norm).
  * OuroForCausalLM.forward may early-exit / mix steps via `early_exit_threshold`,
    `early_exit_step`, `use_weighted_exit`; we never use that path for prefill, and for
    generation we set the threshold/step to None so exactly t loops are used.
  * The KV cache (generation only) is sized from config.total_ut_steps.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Sequence

import torch

from .base import AdapterAssumptionError, LoopAdapter, ModulePaths


class OuroAdapter(LoopAdapter):
    loop_axis = "loop"
    default_paths = ModulePaths(recurrent_layers="model.layers", k_proj="self_attn.k_proj",
                                v_proj="self_attn.v_proj", loop_state="model.norm",
                                final_norm="model.norm")

    def loop_hint(self) -> str:
        return "OuroModel.total_ut_steps (set by OuroAdapter._loops)"

    @contextlib.contextmanager
    def _loops(self, T: int, generation: bool = False) -> Iterator[None]:
        inner = self.model.model
        if not hasattr(inner, "total_ut_steps"):
            raise AdapterAssumptionError(
                f"[{self.name}] {type(inner).__name__} has no `total_ut_steps`; this Ouro modeling "
                "code differs from the 4.55-era file the adapter was written against (ADAPTERS.md)"
            )
        saved = {"inner": inner.total_ut_steps, "cfg": getattr(self.model.config, "total_ut_steps", None),
                 "thr": getattr(self.model, "early_exit_threshold", None),
                 "step": getattr(self.model, "early_exit_step", None)}
        inner.total_ut_steps = T
        if generation:
            self.model.config.total_ut_steps = T  # cache size = layers * total_ut_steps
            self.model.early_exit_threshold = None
            self.model.early_exit_step = None
        try:
            yield
        finally:
            inner.total_ut_steps = saved["inner"]
            if generation:
                self.model.config.total_ut_steps = saved["cfg"]
                self.model.early_exit_threshold = saved["thr"]
                self.model.early_exit_step = saved["step"]

    def _hidden_list(self, input_ids: torch.Tensor, T: int) -> list[torch.Tensor]:
        with self._loops(T):
            out = self.model.model(input_ids=input_ids, use_cache=False)
        if not (isinstance(out, (tuple, list)) and len(out) >= 2):
            raise AdapterAssumptionError(
                f"[{self.name}] OuroModel.forward returned {type(out).__name__}; expected "
                "(outputs, hidden_states_list, gate_list)"
            )
        hs = out[1]
        if len(hs) != T:
            raise AdapterAssumptionError(
                f"[{self.name}] hidden_states_list has {len(hs)} entries for total_ut_steps={T}"
            )
        return hs

    def _run(self, input_ids: torch.Tensor, T: int) -> None:
        self._hidden_list(input_ids, T)

    def _logits_after(self, input_ids: torch.Tensor, T: int, positions: Sequence[int]) -> torch.Tensor:
        return self.model.lm_head(self._hidden_list(input_ids, T)[-1][0, list(positions)])

    def _readout_state(self, input_ids, state, positions):
        return self.model.lm_head(state[list(positions)])  # state is already post-`norm`

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, t: int, max_new_tokens: int,
                        stop_ids: Sequence[int], stop_strings: Sequence[str] = ()) -> list[int]:
        if not hasattr(self.model, "generate"):
            return super().generate_greedy(input_ids, t, max_new_tokens, stop_ids, stop_strings)
        with self._loops(t, generation=True):
            out = self.model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                      max_new_tokens=max_new_tokens,
                                      **self._hf_generate_kwargs(stop_ids, stop_strings))
        return out[0, input_ids.shape[1]:].tolist()


def patch_ouro_for_transformers5(model_id: str) -> None:
    """Compatibility shims (from wavefront-decoding's OuroWrapper) needed only when running
    Ouro's 4.55-era remote code under transformers>=5; no-ops otherwise."""
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _default_rope(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim))
        return inv, 1.0

    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        rot = get_class_from_dynamic_module("modeling_ouro.OuroRotaryEmbedding", model_id)
    except Exception:  # noqa: BLE001 - best effort shim
        return
    if not hasattr(rot, "compute_default_rope_parameters"):
        rot.compute_default_rope_parameters = lambda self, config, device=None, seq_len=None, **kw: _default_rope(config, device)
