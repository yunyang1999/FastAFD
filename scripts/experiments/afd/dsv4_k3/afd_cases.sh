#!/usr/bin/env bash
# RUN_SCRIPT for submit_afd_hybrid.sbatch: one AFD server + batch sweep per case.
#
#   CASES="k3:16:16:8/16/32/64 dsv4_pro:24:8:8/16/32/64 dsv4_flash:28:4:16/32/64/128"
#   (model:A:F:per-attention-GPU batch sizes; model in dsv4_flash, dsv4_pro, k3)
#
# Required env: DSV4K3_META_DIR (HF checkpoint dirs, or config/tokenizer-only
#   copies, named DeepSeek-V4-Flash, DeepSeek-V4-Pro, Kimi-K3), AFD_OUT_DIR
# Optional env: DSV4K3_WORK_DIR (config-only model dirs, default $AFD_OUT_DIR/models),
#   LAYERS (reduced depth, default full), DUMMY_WEIGHT (1), PROMPT_LEN (1024),
#   MAX_TOKENS (64), MEMORY_RATIO (0.85), CASE_TIMEOUT (s, 3600),
#   K3_TOKENIZER_PATH (tokenizer dir for K3 when tiktoken is unavailable)
# With DUMMY_WEIGHT=0 the model dirs must be the real checkpoints (MODEL_DIR_<model>).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DSV4K3_META_DIR:?}" "${AFD_OUT_DIR:?}" "${CASES:?}"
WORK="${DSV4K3_WORK_DIR:-$AFD_OUT_DIR/models}"
mkdir -p "$WORK" "$AFD_OUT_DIR"
export PROMPT_LEN="${PROMPT_LEN:-1024}" MAX_TOKENS="${MAX_TOKENS:-64}" MEMORY_RATIO="${MEMORY_RATIO:-0.85}"
export DUMMY_WEIGHT="${DUMMY_WEIGHT:-1}" WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-2400}"
declare -A SRC=([dsv4_flash]=DeepSeek-V4-Flash [dsv4_pro]=DeepSeek-V4-Pro [k3]=Kimi-K3)
for spec in $CASES; do
  IFS=: read -r m A F BS <<< "$spec"
  depth="${LAYERS:-0}"
  mdir_var="MODEL_DIR_$m"
  if [[ -n "${!mdir_var:-}" ]]; then
    mdir="${!mdir_var}"
  else
    mdir="$WORK/${m}_L${depth}"
    python "$SCRIPT_DIR/make_model_dir.py" --src "$DSV4K3_META_DIR/${SRC[$m]}" --dst "$mdir" --layers "$depth" >/dev/null
  fi
  if [[ $m == k3 && -n "${K3_TOKENIZER_PATH:-}" ]]; then
    export MINISGL_TOKENIZER_PATH="$K3_TOKENIZER_PATH"
  else
    unset MINISGL_TOKENIZER_PATH
  fi
  tag="${m}_L${depth}_${A}a${F}f"
  echo "===== $tag bs=${BS//\//,} $(date -u +%T)"
  AFD_ATTN_DP_SIZE="$A" AFD_MLP_DP_SIZE="$F" BATCH_SIZES="${BS//\//,}" MODEL_PATH="$mdir" RUN_DIR="$AFD_OUT_DIR/$tag" \
    timeout "${CASE_TIMEOUT:-3600}" bash "$SCRIPT_DIR/run_afd_hybrid_bench.sh" > "$AFD_OUT_DIR/$tag.out" 2>&1
  echo "rc=$? $(date -u +%T)"
  tail -12 "$AFD_OUT_DIR/$tag.out"
  sleep 20
done
