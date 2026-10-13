#!/usr/bin/env bash
# Run the whole kit on one A100, cheapest model first. For every model:
# inspect (assumption checks) -> smoke (5-min pipeline check) -> full run -> analyze.
# Every stage is resumable: finished shards are skipped when the script is re-run.
#
# Two virtualenvs are expected (see README "Install"):
#   VENV_OURO  : transformers==4.56.2  (Ouro + Qwen3 / Llama-3.2 / OLMo-2 references)
#   VENV_RAVEN : transformers==4.51.0  (Huginn + Recurrent-*)
# Set them to the same path to try a single environment.
#
# Usage (from the repository root):
#   VENV_OURO=~/venvs/ld-ouro VENV_RAVEN=~/venvs/ld-raven bash loopdetail/scripts/run_all_a100.sh
#   MODELS="ouro-1.4b qwen3-0.6b" bash loopdetail/scripts/run_all_a100.sh   # subset
#   STAGES="inspect smoke" bash loopdetail/scripts/run_all_a100.sh           # no full runs
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV_OURO=${VENV_OURO:-}
VENV_RAVEN=${VENV_RAVEN:-$VENV_OURO}
MODELS=${MODELS:-"ouro-1.4b huginn recurrent-llama llama-3.2-1b ouro-2.6b qwen3-0.6b qwen3-1.7b"}
STAGES=${STAGES:-"inspect smoke full analyze"}
EXPS=${EXPS:-"sanity,B,A1,A2,H3,C"}
OUT=${OUT:-loopdetail/results}
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

# Rough, UNMEASURED estimates of the full run (A100-80GB, bf16, batch size 1), derived from
# parameter count x loops x prompt length. Treat them as order-of-magnitude guides only;
# real times are logged to results/<model>/run.log.
declare -A EST=(
  [ouro-1.4b]="~3-5 h (B ~1 h, A1 ~0.5 h, A2 ~0.5 h, H3 ~1 h, C ~1.5 h)"
  [ouro-2.6b]="~5-8 h"
  [huginn]="~10-16 h (64-loop forwards of a 3.5B model dominate B; C uses up to 32 steps)"
  [recurrent-llama]="~4-7 h"
  [llama-3.2-1b]="~1-2 h (single loop; no H3)"
  [qwen3-0.6b]="~1 h (single loop; no H3)"
  [qwen3-1.7b]="~1.5 h (single loop; no H3)"
)

venv_for() {
  case "$1" in
    huginn|recurrent-*) echo "$VENV_RAVEN" ;;
    *) echo "$VENV_OURO" ;;
  esac
}

run_py() {  # run_py <venv> <args...>
  local venv=$1; shift
  if [[ -n "$venv" ]]; then "$venv/bin/python" -m loopdetail "$@"; else python -m loopdetail "$@"; fi
}

for model in $MODELS; do
  venv=$(venv_for "$model")
  echo "================ $model  (estimated full run: ${EST[$model]:-unknown}) ================"
  for stage in $STAGES; do
    log="$LOG_DIR/${model}_${stage}.log"
    start=$(date +%s)
    echo "[$(date +%H:%M:%S)] $model: $stage -> $log"
    case "$stage" in
      inspect) run_py "$venv" inspect --model "$model" --out-json "$OUT/$model.inspect.json" >"$log" 2>&1 ;;
      smoke)   run_py "$venv" run --model "$model" --smoke --skip-repro --exp "$EXPS" --out "$OUT" >"$log" 2>&1 ;;
      full)    run_py "$venv" run --model "$model" --exp "$EXPS" --out "$OUT" >"$log" 2>&1 ;;
      analyze) run_py "$venv" analyze --model "$model" --out "$OUT" >"$log" 2>&1 ;;
      *) echo "unknown stage $stage"; exit 2 ;;
    esac
    echo "[$(date +%H:%M:%S)] $model: $stage done in $(( ($(date +%s) - start) / 60 )) min"
  done
done
