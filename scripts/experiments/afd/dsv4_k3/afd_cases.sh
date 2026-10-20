#!/usr/bin/env bash
# RUN_SCRIPT for submit_afd_hybrid.sbatch: one AFD server + batch sweep per case.
#
#   CASES="k3:16:16:8/16/32/64 dsv4_pro:24:8:8/16/32/64 dsv4_flash:28:4:16/32/64/128:1 k3:16:16:128:2:2"
#   (model:A:F:per-attention-GPU batch sizes[:microbatches, default 2[:attention TP, default 1]];
#    A counts attention GPUs (A/TP DP groups); model in dsv4_flash, dsv4_pro, k3)
#
# Required env: DSV4K3_META_DIR (HF checkpoint dirs, or config/tokenizer-only
#   copies, named DeepSeek-V4-Flash, DeepSeek-V4-Pro, Kimi-K3), AFD_OUT_DIR
# Optional env: DSV4K3_WORK_DIR (config-only model dirs, default $AFD_OUT_DIR/models),
#   LAYERS (reduced depth, default full), DUMMY_WEIGHT (1), PROMPT_LEN (1024),
#   MAX_TOKENS (64), MEMORY_RATIO (0.85), CASE_TIMEOUT (s, 3600),
#   K3_TOKENIZER_PATH (tokenizer dir for K3 when tiktoken is unavailable),
#   SWEEP_PROMPT_FILE (real text prompts for the sweep), E2E_OUT_DIR (+ ISL, OSL, COMMIT,
#   KV_DTYPE, MOE_PRECISION, NOTE: write e2e_point.py JSONs per batch size after each case)
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
  IFS=: read -r m A F BS MB TP <<< "$spec"
  MB="${MB:-${AFD_NUM_MB:-2}}"
  TP="${TP:-${AFD_ATTN_TP_SIZE:-1}}"
  if (( A % TP != 0 )); then echo "A=$A not a multiple of TP=$TP" >&2; continue; fi
  DP=$(( A / TP ))
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
  tag="${m}_L${depth}_${A}a${F}f_mb${MB}$( (( TP > 1 )) && echo "_tp${TP}" )${CASE_TAG_SUFFIX:-}_bs${BS//\//-}"
  echo "===== $tag bs=${BS//\//,} $(date -u +%T)"
  AFD_NUM_MB="$MB" AFD_ATTN_DP_SIZE="$DP" AFD_ATTN_TP_SIZE="$TP" AFD_MLP_DP_SIZE="$F" BATCH_SIZES="${BS//\//,}" MODEL_PATH="$mdir" RUN_DIR="$AFD_OUT_DIR/$tag" \
    timeout "${CASE_TIMEOUT:-3600}" bash "$SCRIPT_DIR/run_afd_hybrid_bench.sh" > "$AFD_OUT_DIR/$tag.out" 2>&1
  rc=$?
  echo "rc=$rc $(date -u +%T)"
  tail -12 "$AFD_OUT_DIR/$tag.out"
  if [[ -n "${E2E_OUT_DIR:-}" && -s "$AFD_OUT_DIR/$tag/step_times.json" ]]; then
    # one fastafd.e2e-step-time.v1 JSON per batch size (ISL/OSL describe the workload;
    # PROMPT_LEN is set so the timed steps run at context ISL + OSL/2)
    declare -A KEY=([dsv4_flash]=deepseek_v4_flash [dsv4_pro]=deepseek_v4_pro [k3]=kimi_k3)
    python "$SCRIPT_DIR/e2e_point.py" --kind afd --run-dir "$AFD_OUT_DIR/$tag" --out-dir "$E2E_OUT_DIR" \
      --model "${KEY[$m]}" --weights "$( [[ "$DUMMY_WEIGHT" == 1 ]] && echo dummy || echo real )" \
      --isl "${ISL:-$PROMPT_LEN}" --osl "${OSL:-$MAX_TOKENS}" --software fastafd --commit "${COMMIT:-}" \
      --a "$A" --f "$F" --mb "$MB" --attn-tp "$TP" --nodes "${SLURM_JOB_NUM_NODES:-0}" \
      --kv-dtype "${KV_DTYPE:-fp8_e4m3}" --moe-precision "${MOE_PRECISION:-mxfp4 (MegaMoE M2N stage)}" \
      --attention-kernels "${ATTN_KERNELS:-FastAFD sglang-path decode kernels}" --job-id "${SLURM_JOB_ID:-}" \
      --case "$tag" --note "${NOTE:-}" || echo "e2e_point failed for $tag" >&2
  fi
  sleep 20
done
