# Adapter internals: what each adapter relies on, and why

All adapters share one mechanism (`models/base.py`): forward hooks on the k/v
projection modules of the looped layers, plus one hook on the module whose output is the
recurrent state. Shared weights mean the *same* module object runs once per loop, so
a per-module call counter assigns the loop index. After every hooked forward the counters
are checked (`AdapterAssumptionError` names the config field to fix). Everything runs at
batch size 1 with no KV cache, so each (loop, layer) produces exactly one call.

`python -m loopdetail inspect --model <name>` checks every assumption below on the real
weights and prints PASS/FAIL lines. **Run it first on the A100.**

Sources read while writing the adapters (cloned under the session scratchpad
`repos2/`):

| source | used for |
|---|---|
| wavefront-decoding `src/recursive_models/{ouro,huginn,parcae}_wrapper.py`, `src/decoding/{ouro,huginn,parcae}_blocks.py` | how to set Ouro's loop count (`model.model.total_ut_steps`); `lm_head(hidden_states_list[-1])` equals the stock logits (their claim, verified there on real weights); transformers-5 shims; Huginn P/R/C structure, fused `Wqkv` + `qk_bias`, seeded like-init state |
| FlashLoop `flashloop_torch/{attention_utils,token_updates}.py` | Ouro attention kwargs (`current_ut`, `past_key_value`), cache slot `current_ut * num_layers + layer_idx`, `model.model.norm` applied every loop, `transformers==4.56.2` pin |
| ThinkFlowLab/vllm-rlt `vllm_rlt/models/ouro.py`, `tests/reference/{models,huginn}.py` | independent re-implementation of the Ouro sandwich-norm layer and loop (norm at every loop boundary); functional Huginn oracle (prelude → adapter(cat) → core × r → ln_f → coda → ln_f); vLLM-rlt rejects Huginn configs with `test_time_noise != 0` |
| third-party copies of HF `modeling_ouro.py` (GitHub: daedalus/llm-fusion, jk4011/Loop-TTT) | `OuroModel.forward` / `OuroForCausalLM.forward` source: the 3-tuple return, `total_ut_steps` attribute, early-exit logic. **These copies are slightly modified from the hub file. I could not read the hub original because Hugging Face is blocked in the sandbox.** |
| seal-rg/recurrent-pretraining `recpre/raven_modeling_minimal.py` | Huginn forward signature (`input_states`, `num_steps`), `iterate_forward`, `initialize_state`, `predict_from_latents`, `CausalSelfAttention` (`Wqkv`, `chunks`), the fact that `attention_mask` is ignored |
| mcleish7/retrofitting-recurrence `convert_pretrained_model/raven_modeling_minimal_{llama,olmo}.py` | the Recurrent-Llama/OLMo files: same Raven skeleton with Llama RoPE and GQA, pre-norm blocks, a broken `predict_from_latents` (it reads a missing `freqs_cis`), and `transformers==4.51.0` for the conversion code |
| rkstgr/LoopLM | skimmed for loop and exit-gate conventions only; nothing in the adapters depends on it |

## OuroAdapter (`models/ouro.py`): Ouro-1.4B/2.6B(-Thinking)

| assumption | why needed | checked by |
|---|---|---|
| `model.model.layers[i].self_attn.{k_proj,v_proj}` are `nn.Linear` with `out_features = num_key_value_heads * head_dim` | capture/patch site | load-time shape check; inspect call counts |
| `model.model.total_ut_steps` (instance attribute) sets the number of loops of `OuroModel.forward` | set-loop-count readout | inspect: k_proj called T times; "loop count changes the output" |
| `OuroModel.forward(input_ids, use_cache=False)` returns `(outputs, hidden_states_list, gate_list)` with `len(hidden_states_list) == T` | readout for loop t = `lm_head(hidden_states_list[t-1])` | runtime check in `_hidden_list` |
| `model.model.norm` is applied after **every** loop, and its output is both the next loop's input and the LM-head input | capture-based per-loop readout = `lm_head(norm output at loop t)`, with no extra norm | inspect: loop_state called T times; S1 equivalence |
| `OuroForCausalLM.forward` may pick an early-exit step (`early_exit_threshold` from config = 1.0, `early_exit_step`, `use_weighted_exit`) | **never used for prefill** (we call the inner model). For generation (ExpC) we set `early_exit_threshold = early_exit_step = None` so exactly t loops are used, and `config.total_ut_steps = t` so the `UniversalTransformerCache` is sized correctly | inspect prints native-generate vs no-cache-greedy agreement |
| transformers 5.x only: `ROPE_INIT_FUNCTIONS["default"]`, `OuroRotaryEmbedding.compute_default_rope_parameters`, `config.pad_token_id` | loading the 4.55-era remote code | shims are no-ops on 4.x |

Why the early exit matters: with `exit_threshold = 1.0` the stock forward exits at the first
step whose cumulative exit probability is >= 1.0. In bf16 that can happen before the last
step. Calling `OuroForCausalLM.forward` directly could therefore silently use fewer than t
loops.

## RavenAdapter (`models/raven.py`): Huginn-0125 and Recurrent-{Llama,OLMo,TinyLlama}

| assumption | why needed | checked by |
|---|---|---|
| `model.transformer.{prelude, adapter, core_block, coda, ln_f}`, `lm_head` | module paths | resolve-time error naming `module_paths.*` |
| `core_block[i].attn.Wqkv` is a fused Linear split by `attn.chunks = [q, k, v]` | k/v slices of the fused output; additive `qk_bias` (Huginn) and RoPE come after the split, so patching the slice is equivalent to patching K/V at the same position | load-time chunk/shape check; inspect counts |
| `forward(input_ids, input_states=S0, num_steps=<int>, output_details=..., use_cache=False)` runs prelude once, then `num_steps` × [adapter(cat[x, prelude_out]) → core_block], then ln_f → coda → ln_f → lm_head | set-loop-count readout. The input is re-injected every loop | inspect: core k/v called T times, prelude/coda once |
| With `input_states` given, no random state is drawn; `initialize_state(embeds)` draws the model-native random state (trunc-normal × emb_scale) | determinism. We draw S0 once per prompt length under a fixed seed (`options.state_seed`), or use zeros (`options.state_init: zero`). Without this, Huginn's per-call random S0 would make runs with t and T loops non-nested and minimal pairs incomparable | S1 (capture vs set readout agree only if S0 is identical) |
| `forward(num_steps=0, input_states=S)` returns the logits of `coda(ln_f(S))` | capture-based per-loop readout. `predict_from_latents` is avoided because the retrofitted Llama file's version references a non-existent `freqs_cis` | S1 |
| `config.test_time_noise == 0` | otherwise the recurrence injects noise | load-time error |
| `attention_mask` is ignored | batch size 1, no padding (the whole kit is bs=1) | n/a |
| generation: `model.generate(ids, GenerationConfig(...), tokenizer=..., num_steps=t)` (HF generate + Huginn cache) | ExpC | inspect prints agreement with the no-cache greedy reference. Exact agreement is **not** expected: native generation draws a fresh random state for every new token |

Recurrent-Llama / OLMo (retrofitted) were converted from Llama-3.2-1B / OLMo-2-1B and
use the same Raven file with Llama RoPE (`rotary_emb`), GQA and plain pre-norm blocks.
The adapter does not depend on those differences. Their layer counts (prelude/core/coda)
come from the config at runtime.

## HFDecoderAdapter (`models/hf_decoder.py`): Qwen3, Llama-3.2, OLMo-2

`model.model.layers[i].self_attn.{k_proj,v_proj}`, `model.model.norm`, `lm_head`. There is
one "loop" and the layer axis takes the role of the loop axis for representation
analyses. Qwen3 applies `k_norm` after `k_proj`. Patching the `k_proj` output at the same
position is still equivalent to patching K. **Verified on local Qwen3-0.6B with
transformers 5.19** (inspect: all PASS).

## Unverifiable here (no Ouro / Huginn / Recurrent-* weights in the sandbox)

The following were implemented from source reading and tested only on fake models that
*mimic* these interfaces (`tests/fakes.py`). Each must be confirmed by `inspect` on the A100:

1. The hub `modeling_ouro.py` matches the copies I read in the parts used: `total_ut_steps`
   attribute, 3-tuple return, per-loop `norm`, and the attribute names
   `early_exit_threshold` / `early_exit_step` on `OuroForCausalLM`.
2. Ouro's native `generate` respects the patched `total_ut_steps` / disabled early exit
   (inspect compares it with the no-cache reference).
3. Huginn and Recurrent-* accept `num_steps` as a plain int, including `0`, together with
   `input_states`. The `transformers` version you install must still route the
   `num_steps` kwarg through `generate` (ExpC).
4. `attn.chunks` exists on the hub Raven files for Recurrent-* (it does in the GitHub
   conversion files).
5. The tokenizers of Ouro / Huginn / Llama-3 are fast tokenizers with offset mappings.
   Enough single-token items and categories should survive filtering; the resolved pool
   is saved to `results/<model>/stimuli/pool.json`.
6. Huginn's model-native state distribution via `initialize_state`. We seed it; the model
   itself draws an unseeded one.
7. **Shared by all adapters (added for the harder levels and the probe baselines):**
   * `model.get_input_embeddings()` returns the token embedding module (the A1
     input-embedding baseline; Raven multiplies by `emb_scale` afterwards, so the baseline
     is the unscaled embedding, which a standardised linear probe cannot tell apart).
   * The context limit is the smallest of `config.max_position_embeddings`,
     `config.block_size` and `config.n_positions`. Huginn's `freqs_cis` covers only
     `block_size` (4096); Ouro's config says 65536. Override with `options.max_context`
     in models.yaml.
