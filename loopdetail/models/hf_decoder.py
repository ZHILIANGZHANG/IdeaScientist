"""Non-looped HF decoders (Qwen3, Llama-3.2, OLMo-2): 'layer' plays the role of 'loop'
for representation-level analyses; capture returns a single loop with all layers."""
from __future__ import annotations

from typing import Sequence

import torch

from .base import LoopAdapter, ModulePaths


class HFDecoderAdapter(LoopAdapter):
    loop_axis = "layer"
    default_paths = ModulePaths(recurrent_layers="model.layers", k_proj="self_attn.k_proj",
                                v_proj="self_attn.v_proj", final_norm="model.norm")

    def loop_hint(self) -> str:
        return "nothing (non-looped model; every decoder layer runs once)"

    def _run(self, input_ids: torch.Tensor, T: int) -> None:
        self.model.model(input_ids=input_ids, use_cache=False)

    def _logits_after(self, input_ids: torch.Tensor, T: int, positions: Sequence[int]) -> torch.Tensor:
        hidden = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return self.model.lm_head(hidden[0, list(positions)])

    def _readout_state(self, input_ids, state, positions):
        norm = self._resolve(self.paths.final_norm, "final_norm")
        return self.model.lm_head(norm(state[list(positions)]))

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, t: int, max_new_tokens: int,
                        stop_ids: Sequence[int], stop_strings: Sequence[str] = ()) -> list[int]:
        out = self.model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                  max_new_tokens=max_new_tokens,
                                  **self._hf_generate_kwargs(stop_ids, stop_strings))
        return out[0, input_ids.shape[1]:].tolist()
