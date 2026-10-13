"""Raven-family recurrent-depth models: Huginn-0125 and the retrofitted
Recurrent-{Llama,OLMo,TinyLlama} checkpoints (both ship `raven_modeling_minimal.py`).

Relied-upon internals (see ADAPTERS.md):
  * model.transformer = {wte, prelude, adapter, core_block, coda, ln_f}; forward runs
    prelude once, then `num_steps` times: x = core_block(adapter(cat[x, prelude_out])),
    then ln_f -> coda -> ln_f -> lm_head. The input is RE-INJECTED every iteration.
  * forward(input_ids, input_states=..., num_steps=<int>) uses `input_states` as the
    initial recurrent state; otherwise a RANDOM state is drawn (initialize_state). We
    always pass a seeded state (or zeros) so runs are deterministic and so that a run
    with t steps is an exact prefix of a run with T > t steps.
  * forward(num_steps=0, input_states=S) returns logits of coda(ln_f(S)), i.e. the
    readout of an arbitrary recurrent state (used for capture-based per-loop readout;
    `predict_from_latents` is NOT used because the retrofitted Llama file's version
    references a non-existent `freqs_cis` buffer).
  * attention: fused `attn.Wqkv` split by `attn.chunks` = [q, k, v] (GQA in retrofitted
    Llama; additive qk_bias in Huginn is applied after the split, before RoPE).
  * attention_mask is ignored by these models -> batch size 1, no padding (as here).
  * config.test_time_noise must be 0 (otherwise the recurrence is stochastic).
"""
from __future__ import annotations

from typing import Sequence

import torch

from .base import AdapterAssumptionError, LoopAdapter, ModulePaths

_DETAILS = {"return_logits": True, "return_latents": False, "return_head": False, "return_stats": False}


class RavenAdapter(LoopAdapter):
    loop_axis = "loop"
    default_paths = ModulePaths(recurrent_layers="transformer.core_block", qkv_proj="attn.Wqkv",
                                qkv_chunks_attr="attn.chunks", loop_state="transformer.core_block.-1",
                                prelude_layers="transformer.prelude", coda_layers="transformer.coda",
                                final_norm="transformer.ln_f")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        noise = getattr(self.model.config, "test_time_noise", 0) or 0
        if noise != 0:
            raise AdapterAssumptionError(
                f"[{self.name}] config.test_time_noise={noise}; set it to 0 for deterministic loops"
            )
        self.state_init = self.options.get("state_init", "seeded_random")
        self.state_seed = int(self.options.get("state_seed", 0))
        if self.state_init not in ("seeded_random", "zero"):
            raise ValueError("options.state_init must be 'seeded_random' or 'zero'")

    def loop_hint(self) -> str:
        return "the `num_steps` forward argument (RavenAdapter passes it explicitly)"

    def initial_state(self, length: int) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        shape = (1, length, self.d_model)
        if self.state_init == "zero":
            return torch.zeros(shape, dtype=dtype, device=self.device)
        if not hasattr(self.model, "initialize_state"):
            raise AdapterAssumptionError(f"[{self.name}] model has no initialize_state(); use state_init: zero")
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.state_seed)
            return self.model.initialize_state(torch.zeros(shape, dtype=dtype, device=self.device))

    def _forward(self, input_ids: torch.Tensor, num_steps: int, state: torch.Tensor):
        return self.model(input_ids=input_ids, input_states=state, num_steps=int(num_steps),
                          output_details=dict(_DETAILS), use_cache=False)

    def _run(self, input_ids: torch.Tensor, T: int) -> None:
        self._forward(input_ids, T, self.initial_state(input_ids.shape[1]))

    def _logits_after(self, input_ids: torch.Tensor, T: int, positions: Sequence[int]) -> torch.Tensor:
        out = self._forward(input_ids, T, self.initial_state(input_ids.shape[1]))
        return out.logits[0, list(positions)]

    def _readout_state(self, input_ids, state, positions):
        out = self._forward(input_ids, 0, state.unsqueeze(0))
        return out.logits[0, list(positions)]

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, t: int, max_new_tokens: int,
                        stop_ids: Sequence[int], stop_strings: Sequence[str] = ()) -> list[int]:
        """Native Raven generation (HF generate, KV cache, num_steps=t). The model draws a
        fresh random initial state per new token; we seed torch so runs are reproducible."""
        if not hasattr(self.model, "generate") or not hasattr(self.tok, "tok"):
            return super().generate_greedy(input_ids, t, max_new_tokens, stop_ids, stop_strings)
        from transformers import GenerationConfig

        cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                               eos_token_id=list(stop_ids), pad_token_id=stop_ids[0],
                               return_dict_in_generate=False, stop_strings=list(stop_strings) or None)
        torch.manual_seed(self.state_seed)
        out = self.model.generate(input_ids, cfg, tokenizer=self.tok.tok, num_steps=int(t))
        return out[0, input_ids.shape[1]:].tolist()


def patch_raven_for_transformers5(model_id: str) -> None:
    """transformers>=5 expects `_tied_weights_keys` to be a dict (shim from
    wavefront-decoding's HuginnWrapper); no-op on 4.x."""
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", model_id)
    except Exception:  # noqa: BLE001 - best effort shim
        return
    if isinstance(getattr(cls, "_tied_weights_keys", None), list):
        cls._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
