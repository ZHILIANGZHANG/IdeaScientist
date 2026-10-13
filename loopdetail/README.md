# loopdetail: does looping more erase detail information?

An experiment kit for looped / recurrent-depth language models. It asks whether running
more loops degrades a model's access to **detail** information in its context (exact
identity or value) relative to **gist** information (category or coarse value), and
whether the loss sits in the context positions' keys and values.

**Status.** Nothing in this kit has been run on a looped model yet. The adapters were
written from source code and tested on tiny fake looped models (`tests/fakes.py`). The
only real-model numbers are the non-looped Qwen3-0.6B CPU validation in
[CPU_VALIDATION.md](CPU_VALIDATION.md). Every Ouro / Huginn / Recurrent-Llama result is
**not yet run**. See [ADAPTERS.md](ADAPTERS.md) for the internals each adapter relies on
and which of them are still unverified.

## Hypotheses

* **H1** As the loop count t grows, retrieval of *detail* information from context
  degrades, or saturates earlier, relative to *gist* information.
* **H2** What is lost is detail (exact identity or value), not gist (category or coarse
  value). This predicts a loop × info-type interaction.
* **H3 (causal)** Replacing context-position K/V at loop t with loop-1 K/V restores detail
  retrieval.
* **Contrasts:** Ouro (no input re-injection), Huginn (re-injects the input every
  iteration), Recurrent-Llama-3.2 (looped, retrofitted) and its base Llama-3.2-1B, and
  Qwen3. For the non-looped models, "layer" plays the role of "loop", and only in the
  representation analyses (A1, A2).

## Pre-registered criteria

| claim | confirmed if | refuted if |
|---|---|---|
| H1/H2 behavioural (ExpB) | within the trained loop range, the `t × info_type` coefficient of `correct ~ log2(t) * detail + (1 \| context)` is **negative**, with the 95% interval excluding 0 (both VB mixed model and GEE), in the box family at N ≥ 8; and the detail curve falls, or flattens, while the gist curve does not (per-type GEE slopes; McNemar between adjacent t) | interaction CI includes 0 or is positive, or detail and gist curves move together |
| H2 representational (ExpA1/A2) | item-identity probe accuracy (and selectivity) on V/K at fact positions **decreases** over loops while category accuracy stays flat. In A2, the detail-swap / gist-swap distance ratio at fact and query positions decreases with loops, and both stay above the unrelated-swap control | identity is as decodable at late loops as at loop 1, or the ratios are flat |
| H3 causal (ExpH3) | at T = trained loops, `same_loop1` patching of context K/V raises **detail** accuracy over `none` (paired McNemar p < .05) more than it raises gist accuracy. `diff_loop1` must **hurt** both (otherwise the patch site is not causal and H3 is untestable). `noise` must hurt | `same_loop1` does not help detail, or helps gist as much |
| Architecture | the H1 interaction is stronger (more negative) for **Ouro** than for **Huginn**, because Huginn's input re-injection should protect detail. Recurrent-Llama vs Llama-3.2-1B, and Qwen3, serve as non-looped anchors | Huginn ≥ Ouro |

Ouro loops 5–8 are extrapolation, beyond the 4 loops it was trained with. They are
analysed separately (`extrapolated` column) and never pooled into the trained-range
test. Huginn and the retrofitted models were trained with a *sampled* recurrence count
(mean 32), so they have no hard extrapolation boundary.

## Design (all prefill-only, forced choice, hook-based)

* **Stimuli** (`stimuli/`), generated per tokenizer and seeded:
  * *box*: "Box KQ contains the apple." × N. A detail query ("Which exact object…",
    4 same-category candidates) and a gist query ("What kind of object…", 4 categories)
    are asked about the same fact.
  * *codes*: "The code of Alice is 4821." Detail = the exact code among 4 codes that
    differ in one non-leading digit. Gist = the thousand range. Scoring is multi-token
    (summed log-prob).
  * N ∈ {4, 8, 16, 32}. The design balances answer slot × query quartile and the
    answer frequency.
  * Every instance carries three minimal pairs (detail-swap, gist-swap, unrelated-swap)
    and a different-input control context with identical token positions.
  * Box answers must be one token for the model's own tokenizer. The resolved pool is
    saved in `stimuli/pool.json`.
* **Adapters** (`models/`): one interface for Ouro, Raven (Huginn and Recurrent-*) and HF
  decoders.
  * `readout_logits` runs exactly t loops.
  * `readout_loops` runs one forward and reads out every loop.
  * `capture` records pre-RoPE K, V and the layer output per (loop, layer).
  * `patch_kv` replaces K/V at chosen positions, loops, layers and heads.
  * `inspect` checks every structural assumption.
* **Experiments** (`experiments/`):
  * **sanity**:
    * S1: set-count vs capture readout.
    * S2: per-loop ARC trend.
    * S3: fp32 patch no-op.
    * S4: current-token probe.
    * S5: no-context baseline at chance.
  * **B**: per-loop accuracy and (calibrated) margins.
  * **A1**: probes at fact positions.
    * C chosen by group-CV on train only.
    * Hewitt–Liang control and selectivity.
    * Context-level bootstrap CIs.
    * Optional online-code MDL.
  * **A2**: minimal-pair sensitivity.
  * **H3**: loop-1 K/V patching.
    * Controls: different-input loop-1 and norm-matched noise.
    * Optional layer and head sweeps.
  * **C**: GSM8K at fixed t.
    * Regression set.
    * Automatic operand-provenance tagging, plus a 100-row CSV for human labels.
    * `agreement` command computes Cohen's κ.
* **Analysis** (`analysis/`):
  * Mixed-effects logistic model (statsmodels VB), with a GEE fallback.
  * Exact McNemar between adjacent t, GEE trend slopes.
  * PNG figures.

## Install (A100)

Ouro and the Raven models want different `transformers` versions, so use two
virtualenvs. Install torch for your CUDA first.

```bash
python -m venv ~/venvs/ld-ouro  && ~/venvs/ld-ouro/bin/pip install -r loopdetail/requirements-ouro.txt
python -m venv ~/venvs/ld-raven && ~/venvs/ld-raven/bin/pip install -r loopdetail/requirements-raven.txt
huggingface-cli login   # Llama-3.2-1B is gated (or pass --model-path to a local copy)
```

The code itself targets `transformers>=4.46` and `torch>=2.5`, and includes the shims
needed to run Ouro and Huginn under transformers 5.x. It has been **executed only under
transformers 5.19 / torch 2.14 (CPU)**.

## Quickstart (run from the repository root)

```bash
# 0. verify adapter assumptions on real weights (PASS/FAIL per check)
~/venvs/ld-ouro/bin/python -m loopdetail inspect --model ouro-1.4b
~/venvs/ld-raven/bin/python -m loopdetail inspect --model huginn
# 1. 5-minute end-to-end smoke (results meaningless; written under results/smoke/)
~/venvs/ld-ouro/bin/python -m loopdetail run --model ouro-1.4b --smoke --skip-repro
# 2. full run of one model (sanity must PASS first; resumable)
~/venvs/ld-ouro/bin/python -m loopdetail run --model ouro-1.4b --exp sanity,B,A1,A2,H3,C --analyze
# 3. everything, cheapest model first (inspect -> smoke -> full -> analyze per model)
VENV_OURO=~/venvs/ld-ouro VENV_RAVEN=~/venvs/ld-raven bash loopdetail/scripts/run_all_a100.sh
# after hand-labelling results/<model>/C/annotation_sample.csv:
python -m loopdetail agreement --csv loopdetail/results/ouro-1.4b/C/annotation_sample.csv
```

Useful flags:
* `--model-path` for a local checkpoint.
* `--models-yaml my.yaml` to add models.
* `--dtype float32`.
* `--chat` / `--no-chat`.
* `--force`, which ignores a failed sanity report.
* `--gsm8k-path DIR`, a local directory with `{train,test}.jsonl`.

## Expected cost on one A100

These are **unmeasured estimates** from parameter count × loops × prompt length; actual
times are logged in `results/<model>/run.log`.

| model | memory | full run |
|---|---|---|
| Ouro-1.4B | ~3 GB bf16; ~6 GB while S3 runs in fp32 | ~3–5 h |
| Ouro-2.6B | ~5 GB bf16; ~11 GB during S3 | ~5–8 h |
| Huginn-0125 (3.5B, 64-loop forwards) | ~7 GB bf16; ~14 GB while S3 runs in fp32 | ~10–16 h |
| Recurrent-Llama-3.2 | ~3 GB bf16 | ~4–7 h |
| Llama-3.2-1B / Qwen3 | < 8 GB | ~1–2 h each |

* Both 40 GB and 80 GB cards suffice; nothing is batched. S3 converts the model to fp32 in place and back (exact round trip). H3 keeps one prompt's context K/V for all loops on the GPU (about 1 GB for Huginn at N=16, less elsewhere).
* ExpC (generation) and ExpB on the codes family (4 forwards per candidate set) dominate
  the cost.
* To cut cost, lower `contexts_per_cell` in `configs/experiments.yaml` or override it
  per model under `experiments:` in `configs/models.yaml`.

## Output layout

```
results/<model>/
  run.log                      timestamps + per-stage durations
  stimuli/{pool.json, meta.json, box_N8.jsonl, ...}
  sanity/sanity_report.json    S1–S5 with PASS/FAIL/SKIPPED + overall
  B/rows/{family}_N{N}.jsonl   per item × info_type × t: correct, margin, margin_cal, logp, ...
  A1/features/{family}.pt      captured features (fp16) + labels;  A1/rows/{family}.jsonl probe cells
  A2/rows/...                  per item × variant × rep × site: [loop][layer] distances
  H3/rows/...                  per item × condition × target: correct, margin, delta_margin
  C/rows/t{t}.jsonl, C/annotation_sample.csv
  */summary.json               aggregated table + meta (config, git hash, timestamp, versions)
  analysis.json, figures/*.png
```

Each shard has a `.done` marker. Re-running skips finished shards; delete a shard and its
marker to recompute it.

## How to read the results

* **Behaviour (H1/H2):** start with `figures/B_accuracy_box.png` and
  `analysis.json → B_stats.box_interaction_trained_range`.
  * A negative `mixed_vb.coef` / `gee.coef` with an interval below 0 supports
    H1/H2.
  * Check `nocontext_acc` ≈ 0.25 and `margin_cal` before trusting raw accuracy.
* **Representations:** `A1_probe_*` heatmaps (loop × layer). Detail = item identity, gist =
  category.
  * At the item token, identity decoding at loop 1 / layer 0 is trivial (S4).
  * The question is whether it *declines* across loops in later layers. Selectivity is
    uninformative here. For identity the label is the word type itself. For category,
    a type-memorising control reached 0.97 on Qwen3 (CPU_VALIDATION.md). So compare
    accuracy across loops.
  * In A2, a shrinking detail-swap/gist-swap ratio across loops means the representation
    becomes less sensitive to the exact value relative to its category.
* **Causality:** in `H3_patch_box.png`, `same_loop1` should beat the black "none" line for
  detail only. `diff_loop1` falling toward chance confirms that the context K/V carries
  the facts.

## Known limitations

* **Unverified adapters.** Ouro, Huginn and Recurrent-* adapters are unverified on real
  weights (see ADAPTERS.md). Run `inspect` first.
* **Batch size 1 throughout.** Raven models ignore attention masks, and per-call hook
  counting assumes one call per (loop, layer). This is simple and exact but not fast.
* **Huginn's random initial state.**
  * We fix it per prompt length with a seed, so each run reflects one draw, not an
    expectation over draws. Use `options.state_seed` to vary it, or
    `state_init: zero`.
  * Native Huginn generation (ExpC) draws a fresh state per token.
* **Codes-family probes are asymmetric.** With digit-level tokenizers the detail label
  (last digit) is the current token, while the gist label (first digit) lies 3 tokens
  back. The box family is the primary probe target.
* **Templates.**
  * English only, base-model completion format by default.
  * The box template uses "the {item}" (not "a {item}") to avoid an a/an cue.
* **Chat format.** Thinking models use their chat template with thinking disabled where
  the template supports it. Their forced-choice behaviour may differ from base models.
* **GSM8K tagging.** The detail-error tagger is a heuristic (operand provenance). Use the
  human-annotation CSV and `agreement` before relying on it.
* **S2 checks a trend, not exact numbers.** It verifies that ARC accuracy rises from t=1
  to the trained loop count. It does not reproduce the published figures, because the
  evaluation details differ.
