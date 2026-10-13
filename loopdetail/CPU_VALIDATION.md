# CPU validation (sandbox, no GPU, no Hugging Face access)

What was run here and what it shows. **None of this is evidence about loops.** The only
real model available was Qwen3-0.6B, which is non-looped. Its numbers are a *small-scale
non-looped reference*: they check that the tasks are valid and that the pipeline works on
a real transformer. The fake-model runs only show that the code executes end to end.
Their numbers are meaningless (random weights).

Environment: 4 CPU threads, torch 2.14.1 (CPU), transformers 5.19.0, Python 3.13, fp32.
bf16 was slower on this CPU. The kit targets transformers 4.51 / 4.56 on the A100; those
versions were **not** exercised here.

There are two real-model runs:

| run | design | directory |
|---|---|---|
| v1 | implicit gist, N ≤ 16 | `results_cpu_validation/qwen3-0.6b-v1-implicit/` |
| v2 | explicit gist, harder levels, calibration, knowledge control | `results_cpu_validation/qwen3-0.6b/` |

Only summaries and per-item rows are committed. Stimuli JSONL and probe features
(`*.pt`) are git-ignored and regenerable.

## 1. Unit tests (`python -m pytest loopdetail/tests`)

TESTS_LINE

The tests use two tiny random looped models (`tests/fakes.py`):

* an Ouro-style shared 2-layer stack looped T times, with a norm after every loop;
* a Huginn-style prelude → [adapter(cat) → 2-layer core] × r → coda, with fused `Wqkv`
  and a seeded random initial state.

**Adapters and patching:**

* call counts per loop; prelude/coda run once;
* capture shapes; set-loop-count readout == capture-based readout;
* bit-exact fp32 patch no-op; a different-input patch changes the logits;
* per-head and noise patches;
* loud errors for a wrong module path, a loop-state miscount, test-time noise, and an
  out-of-range patch.

**Stimuli:**

* single-token filtering; the balanced design for all four families;
* explicit and implicit families are paired (identical facts);
* filler neutrality; fact and category spans; token-length preservation in minimal pairs
  with fillers;
* far-quartile and N=128 generation (repeated non-queried items, unique queried item);
* confusable label modes;
* skipping levels that do not fit the context window;
* the calibration selection rule, including the all-at-ceiling case.

**Scoring and statistics:**

* multi-token scoring checked against a manual computation;
* the **torch probe matches sklearn** (probabilities within 2e-3; same CV-selected C and
  accuracy within 0.02);
* synthetic probe sanity, the control task, group folds and MDL;
* Wilson / McNemar (vs scipy), cluster bootstrap, and recovery of a negative loop ×
  info-type interaction (GEE and VB);
* the GSM8K tagger.

**Pipeline:** the full `--smoke` CLI with `calibrate,sanity,B,A1,A2,H3,C` + analyze +
resume on both fakes. This checks that the calibrated levels are used, that the
category-known tables exist, and that A1 rows carry the embedding / loop-1-layer-0
baselines and MDL.

## 2. `inspect` on real Qwen3-0.6B: all PASS (unchanged since v1)

* Each of the 28 layers' k/v projections is called exactly once.
* The own-value patch no-op diff is 0.
* Native `generate` matches the no-cache greedy reference on 8 tokens.

## 3. v2: Qwen3-0.6B, explicit gist + difficulty ladder (non-looped reference)

```
python -m loopdetail run --model qwen3-0.6b --model-path <local Qwen3-0.6B> \
  --profile cpu_validation --exp calibrate,sanity,B --dtype float32 \
  --out loopdetail/results_cpu_validation --analyze
```

V2_WALLTIME

### Calibration (`calibration.json`; box_explicit, 12 pilot contexts per level, T = 1 forward)

V2_CALIB

**First pass (log only, run stopped early).** Before the confusable levels were added, a
first pass with the plain ladder gave:

| level | box_explicit detail / gist | codes_explicit detail / gist |
|---|---|---|
| N4 | 1.00 / 1.00 | 1.00 / 1.00 |
| N32 | 1.00 / 1.00 | – |
| N64 | 1.00 / 0.83 | – |
| N128 (1467 tokens) | 1.00 / 0.92 | – |
| N32 with 2 fillers | 1.00 / 1.00 | – |

That pass also exposed a bug: the selection rule picked the *easiest* level when every
level was at ceiling. It now picks the hardest level and flags it.

**Takeaway: more facts and longer distance alone do not move Qwen3-0.6B off detail
ceiling.** Label interference does (`labels: confusable`). This is why the default ladder
now ends in N64_conf, N128_conf and N128_conf_fill1.

### Sanity: V2_SANITY

### ExpB (selected levels; 16 contexts per cell; chance 0.25)

V2_B

## 4. v1: Qwen3-0.6B, implicit gist, N ≤ 16 (earlier design, kept for reference)

* **Final-layer accuracy, 32 contexts per cell.**
  * Box detail 1.00 at every N.
  * Box gist: 0.78 (N=4), 0.97 (N=8), 1.00 (N=16).
  * Codes detail 1.00 at every N.
  * Codes gist: 0.88, 0.78, 0.75.
  * No-context accuracy 0.19–0.38.
  * **Detail was at ceiling, and gist depended on world / numeric knowledge.** These are
    the two problems the v2 design addresses.
* **A1 probes, box, item token; 144 contexts.**
  * Item identity accuracy is ~1.00 at most layers, with a dip to 0.81–0.85 in K/V at
    layers 13–17. Category accuracy is 1.00 at every layer.
  * The Hewitt–Liang control reached 0.97 on category, so selectivity was
    uninformative. It is now demoted to a secondary measure; the primary references are
    the input-embedding and loop-1/layer-0 baselines and MDL.
  * v1 used the old probe code: per-C batched fits only, 46 CPU-min.
* **A2 minimal pairs.**
  * The fact-position detail/gist distance ratio is 0.78–0.92.
  * At the query position, both swap types are indistinguishable from the unrelated
    control until about layer 14, then exceed it.
* **Wall time:** 81 min (sanity 2.3, B 26.9, A1 46.1, A2 5.3). This was over the 40-min
  target.

## 5. `--smoke` CLI runs on the fake looped models (pipeline check; results meaningless)

SMOKE_LINE

## What could not be validated here

* **Ouro, Huginn and Recurrent-\*.** No weights and no HF access. ADAPTERS.md lists the
  assumptions `inspect` must confirm.
* **S2 (ARC) and the HF `datasets` code paths.** Never executed.
* **ExpC on a real model.** Only the fake models and the tagger tests exercised it.
* **v2 coverage.** v2 ran only box_explicit and box on CPU; codes_explicit only at N4 in
  the first pass. A1, A2 and H3 on the new levels ran only on the fakes. The new
  batched torch probe was checked against sklearn on synthetic data, not on Qwen
  features.
