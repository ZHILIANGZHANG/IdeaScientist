# CPU validation (sandbox, no GPU, no Hugging Face access)

What was run here and what it shows. **None of this is evidence about loops.** The only
real model available was Qwen3-0.6B, which is non-looped. Its numbers below are a
*small-scale non-looped reference*: they check that the tasks are valid and that the
pipeline works on a real transformer. The fake-model runs only show that the code
executes end to end. Their numbers are meaningless (random weights).

Environment: 4 CPU threads, torch 2.14.1 (CPU), transformers 5.19.0, Python 3.13.
The kit targets transformers 4.51 / 4.56 on the A100 (see README), and those versions
were **not** exercised here.

## 1. Unit tests: fake looped models (`tests/`, 51 tests, all pass)

`python -m pytest loopdetail/tests` runs two tiny random looped models (`tests/fakes.py`):

* an Ouro-style shared 2-layer stack looped T times, with a norm after every loop;
* a Huginn-style prelude → [adapter(cat) → 2-layer core] × r → coda, with fused `Wqkv`
  and a seeded random initial state.

The tests check:

* every recurrent k/v projection is called exactly T times, and prelude/coda once;
* capture shapes and dtypes;
* set-loop-count readout equals capture-based readout at every t;
* the fp32 patch no-op is bit-exact (`torch.equal`);
* patching with another input's K/V changes the logits;
* per-head patching touches only the selected head;
* the noise patch matches the per-head norm;
* loud errors for a wrong module path, a loop-state miscount, a test-time-noise config,
  and a patch at an out-of-range loop;
* single-token pool filtering, the balanced design (answer slot / query quartile /
  answer frequency), the minimal-pair and control-context invariants, and token-length
  preservation;
* multi-token scoring vs a manual computation;
* the probe pipeline on synthetic features (separable → >0.95, noise → chance), the
  control task, group folds and MDL;
* Wilson / McNemar (vs scipy), cluster bootstrap, and recovery of a negative loop ×
  info-type interaction on simulated data (GEE and VB mixed model);
* GSM8K answer extraction and operand tagging, the ExpC regression set and Cohen's κ;
* the full `--smoke` CLI pipeline (all six experiments + analyze + resume) on both
  fakes (`test_pipeline.py`).

## 2. `inspect` on real Qwen3-0.6B: all PASS

```
[PASS] each recurrent k/v projection called exactly T=1 times (28 layers x k, v)
[PASS] patching every k/v with its own value is a no-op: max diff 0
[INFO] native generate vs reference no-cache greedy (8 tokens): 8/8 tokens identical
```

On the fakes, `inspect` also passes the loop-specific checks: T calls per loop,
prelude/coda once, set == capture readout, the loop count changes the output, and the
no-op patch.

## 3. Real run: Qwen3-0.6B (non-looped reference, small scale)

Command (fp32, because bf16 was slower on this CPU: 1.7 s vs 5.3 s for a 300-token
prefill):

```
python -m loopdetail run --model qwen3-0.6b --model-path <local Qwen3-0.6B> \
  --profile cpu_validation --exp sanity,B,A1,A2 --dtype float32 \
  --out loopdetail/results_cpu_validation --analyze
```

Outputs are in `results_cpu_validation/qwen3-0.6b/`. `A1/features/*.pt` (95 MB) is not
committed; re-running the command recreates it.

**Wall time was 81 min, over the 40-min target in the spec.** Sanity took 2.3 min, B
26.9 min, A1 46.1 min and A2 5.3 min. A1 is dominated by fitting 168 probe cells
(task + control, 5-fold CV, L-BFGS) on CPU; on the A100 the probes run on the GPU. To
get under 40 min on CPU, use `A1.layers: every:4+last` and `cv_folds: 3`.

**Provenance.** The recorded git hash is `82a98a0-dirty`. The results were produced from
the uncommitted working tree that became the loopdetail commit. Two things changed
afterwards: plotting, and NaN-safe A2 ratios (the A2 summary was regenerated).

### Tokenizer pool

* 8 categories × 12 single-token items survived: fruit, animal, vehicle, clothing,
  furniture, vegetable, drink, metal.
* tool, instrument and flower were dropped: fewer than 10 single-token items survived.
* The resolved pool is in `stimuli/pool.json`.

### Sanity (`sanity/sanity_report.json`): overall PASS

* **S1: N/A** (non-looped).
* **S2: skipped** (no ARC target for this model, and no HF access).
* **S3: PASS.** The patch no-op diff is 0.0 in fp32; patching other values changes the
  logits by 1.65.
* **S4: PASS.** Item identity from V at layer 0 at the item token: acc 1.00 (92 classes,
  60 test samples).
* **S5: PASS**, but weak: n = 12 per cell, no-context accuracy 0.17–0.33. The full
  profile now uses 80 per cell. The better estimate comes from ExpB: no-context accuracy
  0.19–0.38 over 32 items per cell.

### ExpB: final-layer forced choice (task validity). 32 contexts per cell, chance 0.25

| family | info | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| box | detail | 1.00 [0.89, 1.00] | 1.00 [0.89, 1.00] | 1.00 [0.89, 1.00] |
| box | gist | 0.78 [0.61, 0.89] | 0.97 [0.84, 0.99] | 1.00 [0.89, 1.00] |
| codes | detail | 1.00 [0.89, 1.00] | 1.00 [0.89, 1.00] | 1.00 [0.89, 1.00] |
| codes | gist | 0.88 [0.72, 0.95] | 0.78 [0.61, 0.89] | 0.75 [0.58, 0.87] |

95% Wilson CIs in brackets.

* Mean raw margins: box detail ≈ 10.2–11.1 nats, box gist 2.3–4.2; codes detail 7.0–7.9,
  codes gist 0.4–1.0.
* Calibrated (minus no-context) accuracy: box gist 0.84 / 0.94 / 1.00; codes gist 0.88 /
  0.59 / 0.56.

**Reading.** Both query types are well above chance at every N, and the no-context
baseline is near chance, so the tasks are valid for this model.

Two things to watch on the looped models:

1. For a 0.6B model, *gist* is the harder query here (lower accuracy and much smaller
   margins). Detail is at ceiling up to N = 16. A loop-wise *detail* decline would
   therefore not be explained by a harder detail format. Ceiling effects on detail at
   small N are likely, so the N = 16/32 cells matter most.
2. For codes, calibration *lowers* gist accuracy at N ≥ 8. The thousand-range answers
   have strong no-context priors. Report raw and calibrated scores side by side, as the
   summaries do.

### ExpA1: probes at the item token (box; 144 contexts, 576 facts, 25% of contexts held out)

| rep | task | layers 0–8 | 12 | 14 | 16 | 20–27 |
|---|---|---|---|---|---|---|
| V | detail (item, 96 classes) | 0.99–1.00 | 0.94 | 0.81 | 0.84 | 0.99–1.00 |
| V | gist (category, 8 classes) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| K | detail | 1.00 | 0.93 | 0.85 | 0.84 | 0.99–1.00 |
| hidden | detail | 1.00 | 0.95 | 0.93 | 0.94 | 0.99–1.00 |

* The majority baseline is 0.04 for detail and 0.17 for gist.
* Bootstrap CIs are in `A1/rows/box.jsonl`. For example, V detail at layer 14 is 0.81
  [0.74, 0.88].

**Reading.** In this non-looped model, item identity at the item token is near-perfectly
linearly decodable at almost every layer, with a mid-stack dip at layers 13–17.

**Hewitt–Liang selectivity is ~0** for detail (the label *is* the word type) and only
0.01–0.26 for gist. The control task (a random category per item type) reaches 0.97, so
category is decodable mostly through type memorisation. This is a property of probing a
word-type attribute at that word's own position. Interpret A1 by comparing **accuracy
across loops**, not selectivity (README notes this).

### ExpA2: minimal-pair sensitivity (box; 48 contexts × 2 query types)

Normalised L2 between base and edit (hidden state), layer 0 → 27:

* **Fact position.** Detail-swap 1.12 → 0.58, gist-swap 1.21 → 0.74, unrelated-swap
  0.02–0.09. The edit is local, and a different-category edit moves the representation
  more than a same-category edit (ratio 0.78–0.92).
* **Query position.** Both swaps are indistinguishable from the unrelated control until
  about layer 14, then rise to 0.39 (detail) and 0.43 (gist) at layer 27, vs 0.10 for
  the unrelated swap. So the queried fact reaches the query position in the second half
  of the stack.
* **Detail/gist ratio at the query position.** About 0.6–0.9 across layers 16–27
  (detail query) and 0.62–0.74 (gist query).

In the looped models, the same quantities are indexed by loop. H2 predicts this ratio
falls with loops.

## 4. `--smoke` CLI runs on the fake looped models (pipeline check; results meaningless)

`python -m loopdetail run --model fake-{ouro,huginn} --smoke --skip-repro --exp
sanity,B,A1,A2,H3,C --gsm8k-path <local GSM8K> --analyze` completed end to end:

* sanity (S1/S3/S4/S5 PASS, S2 skipped), then B, A1 (with MDL), A2, H3 (all four
  conditions + layer and head sweeps), C (real GSM8K questions, random-weight
  generations), and analyze;
* wall time on 4 CPU threads: fake-ouro 246 s, fake-huginn 333 s. A1 (probe fitting, with MDL) takes about 3 min of that.

Fake-model accuracies are chance-level noise by construction and are not reported.

## What could not be validated here

* Anything about Ouro, Huginn or Recurrent-*: no weights, no HF access. ADAPTERS.md lists
  each assumption that `inspect` must confirm on the A100.
* S2 (ARC per-loop trend) and the HF `datasets` paths: never executed. Their parsing is
  unit-free code.
* ExpC on a real model: generation quality, few-shot format and stop strings. Only the
  fake models and the tagger unit tests exercised this path. On Qwen3-0.6B, `inspect`
  checked that native HF `generate` matches the no-cache greedy reference.
* bf16 numerics of S1 on real looped models. The fakes are fp32, where the agreement is
  exact.
