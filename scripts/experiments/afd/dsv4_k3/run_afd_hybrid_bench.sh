#!/usr/bin/env bash
# DeepSeek-V4 / Kimi-K3 AFD step-time sweep on an existing Ray cluster.
# Run on the Ray head node with the conda env active (see submit_afd_hybrid.sbatch).
#
# Required env:
#   MODEL_PATH        model dir (config.json + tokenizer; dummy weights by default)
#   AFD_ATTN_DP_SIZE  attention workers (A), AFD_MLP_DP_SIZE expert workers (F)
#   RUN_DIR           output dir
# Optional env:
#   AFD_ATTN_TP_SIZE (attention TP per DP group, default 1; AFD_ATTN_DP_SIZE then
#   counts DP groups and the attention GPUs are DP*TP), SWEEP_PROMPT_FILE (real
#   text prompts for the sweep, one per line, distinct per request),
#   BATCH_SIZES (per attention GPU, default 8,16,32,64), PROMPT_LEN (1024),
#   MAX_TOKENS (64), AFD_NUM_MB (2), AFD_DECODE_GRAPH_BS (per-MB buckets),
#   MEMORY_RATIO (0.85), MAX_SEQ_LEN (PROMPT_LEN+MAX_TOKENS+64), DUMMY_WEIGHT (1),
#   AFD_MAX_BATCHED_TOKENS (2048; prefill chunk and MegaMoE bucket), MINISGL_TOKENIZER_PATH, RAY_ADDRESS (auto),
#   MINISGL_AFD_ATTN_NODES / MINISGL_AFD_MLP_NODES (explicit placement),
#   EXTRA_SERVER_ARGS (appended to the server --extra-args),
#   GREEDY_OUT (fixed-prompt greedy tokens instead of the sweep),
#   PROMPT_FILE + NUM_PROMPTS (real text prompts instead of the sweep, cut to
#   PROMPT_LEN tokens) and ROUTING_OUT / ROUTING_NOTE (merge the per-rank
#   MINISGL_DUMP_EXPERT_LOAD dumps into one per-layer routing file)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
: "${MODEL_PATH:?}" "${AFD_ATTN_DP_SIZE:?}" "${AFD_MLP_DP_SIZE:?}" "${RUN_DIR:?}"
BATCH_SIZES="${BATCH_SIZES:-8,16,32,64}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
MAX_TOKENS="${MAX_TOKENS:-64}"
AFD_NUM_MB="${AFD_NUM_MB:-2}"
MEMORY_RATIO="${MEMORY_RATIO:-0.85}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + MAX_TOKENS + 64))}"
DUMMY_WEIGHT="${DUMMY_WEIGHT:-1}"
AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-2048}"  # also the MegaMoE dispatch bucket
AFD_ATTN_TP_SIZE="${AFD_ATTN_TP_SIZE:-1}"
ATTN_GPUS=$(( AFD_ATTN_DP_SIZE * AFD_ATTN_TP_SIZE ))
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
PORT="${PORT:-$((20000 + RANDOM % 20000))}"

# batch sizes are per attention GPU; a TP group of TP GPUs serves TP*b requests
MAX_BS=$(python -c "print(max(int(x) for x in '$BATCH_SIZES'.split(',')))")
PER_MB=$(( (MAX_BS * AFD_ATTN_TP_SIZE + AFD_NUM_MB - 1) / AFD_NUM_MB ))
if [[ -z "${AFD_DECODE_GRAPH_BS:-}" ]]; then
  AFD_DECODE_GRAPH_BS=$(python -c "
bs=[int(x) * $AFD_ATTN_TP_SIZE for x in '$BATCH_SIZES'.split(',')]; mb=$AFD_NUM_MB
print(','.join(str(v) for v in sorted({max(1,(b+mb-1)//mb) for b in bs})))")
fi
# --afd-max-running-requests is global (split evenly over the attention DP ranks)
NUM_REQ_TOTAL=$(( MAX_BS * ATTN_GPUS ))
MAX_RUNNING=$NUM_REQ_TOTAL

ARCH=$(python -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print(c.get('model_type',''))")
case "$ARCH" in
  deepseek_v4) ATTN_BACKEND=dsv4; PAGE_SIZE=256 ;;
  kimi_k3|kimi_linear) ATTN_BACKEND=kimi; PAGE_SIZE=64 ;;
  *) echo "unsupported model_type $ARCH" >&2; exit 2 ;;
esac
# whole pages: the scheduler maps every page of a request into the page table
MAX_SEQ_LEN=$(( (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE ))

mkdir -p "$RUN_DIR/ray_logs"
# Placement: attention ranks fill the GPU nodes (sorted by IP) first, expert
# ranks follow (small dev shapes may put both roles on one node).
if [[ -z "${MINISGL_AFD_ATTN_NODES:-}" ]]; then
  eval "$(python - <<PY
import ray
ray.init(address="$RAY_ADDRESS", logging_level="ERROR")
nodes = sorted({n["NodeManagerAddress"] for n in ray.nodes()
                if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0})
gpn = int(${AFD_GPUS_PER_NODE:-4})
tp = int($AFD_ATTN_TP_SIZE)
slots = [ip for ip in nodes for _ in range(gpn)]
A, F = int($AFD_ATTN_DP_SIZE), int($AFD_MLP_DP_SIZE)
if len(slots) < A * tp + F:
    raise SystemExit(f"need {A * tp + F} GPUs, Ray has {len(slots)} on {nodes}")
# one node entry per attention DP group (its TP ranks share the node)
attn_groups = [slots[i * tp] for i in range(A)]
print("export MINISGL_AFD_ATTN_NODES=" + ",".join(attn_groups))
print("export MINISGL_AFD_MLP_NODES=" + ",".join(slots[A * tp:A * tp + F]))
PY
)"
fi
echo "attn nodes: $MINISGL_AFD_ATTN_NODES"
echo "mlp nodes:  $MINISGL_AFD_MLP_NODES"
export MINISGL_AFD_MOE_BACKEND=megamoe_m2n
export MINISGL_NVTX_CPU_TRACE=1
export MINISGL_ROPE_TABLE_LEN="${MINISGL_ROPE_TABLE_LEN:-$(( MAX_SEQ_LEN + 128 ))}"
export MINISGL_AFD_MAX_COMM_TOKENS="${MINISGL_AFD_MAX_COMM_TOKENS:-$AFD_MAX_BATCHED_TOKENS}"

EXTRA="--attention-backend $ATTN_BACKEND --memory-ratio $MEMORY_RATIO"
[[ "$DUMMY_WEIGHT" == "1" ]] && EXTRA="$EXTRA --dummy-weight"
EXTRA="$EXTRA ${EXTRA_SERVER_ARGS:-}"

echo "AFD hybrid bench: model=$MODEL_PATH arch=$ARCH A=$AFD_ATTN_DP_SIZE x TP$AFD_ATTN_TP_SIZE F=$AFD_MLP_DP_SIZE mb=$AFD_NUM_MB"
echo "  batch_sizes=$BATCH_SIZES graph_bs=$AFD_DECODE_GRAPH_BS prompt=$PROMPT_LEN max_tokens=$MAX_TOKENS"

bash "$REPO_ROOT/scripts/kill_afd_ray_tasks.sh" >/dev/null 2>&1 || true  # leftovers of an earlier case
RAY_DEDUP_LOGS=0 "$REPO_ROOT/scripts/serve/fastafd_server.sh" \
  --model "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --launch-backend ray --ray-address "$RAY_ADDRESS" \
  --ray-log-dir "$RUN_DIR/ray_logs" --afd-report-dir "$RUN_DIR/ray_logs" \
  --afd-attn-dp-size "$AFD_ATTN_DP_SIZE" --afd-mlp-dp-size "$AFD_MLP_DP_SIZE" \
  --afd-attn-tp-size "$AFD_ATTN_TP_SIZE" --afd-mlp-tp-size 1 --afd-mlp-ep-size "$AFD_MLP_DP_SIZE" \
  --afd-batch-size "$NUM_REQ_TOTAL" --afd-max-running-requests "$MAX_RUNNING" \
  --afd-max-new-tokens "$MAX_TOKENS" --afd-max-batched-tokens "$AFD_MAX_BATCHED_TOKENS" \
  --cache-type naive --page-size "$PAGE_SIZE" --cuda-graph-max-bs "$PER_MB" \
  --afd-decode-graph-bs "$AFD_DECODE_GRAPH_BS" --afd-num-mb "$AFD_NUM_MB" \
  --afd-device-comm-num-sms 4 --max-seq-len-override "$MAX_SEQ_LEN" \
  --extra-args "$EXTRA" > "$RUN_DIR/afd.log" 2>&1 &
SERVER_PID=$!
cleanup_ray_workers() {  # hung workers of a killed server keep their GPUs otherwise
  bash "$REPO_ROOT/scripts/kill_afd_ray_tasks.sh" >/dev/null 2>&1 || true
}
stop_server() {
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 2; done
  kill -KILL "$SERVER_PID" 2>/dev/null || true
  cleanup_ray_workers
}
trap stop_server EXIT

deadline=$((SECONDS + ${WAIT_READY_TIMEOUT:-3600}))
until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited" >&2; tail -80 "$RUN_DIR/afd.log" >&2; exit 2; fi
  if grep -qs "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log; then
    echo "worker failure during startup:" >&2; grep -hs -A3 "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log | tail -20 >&2
    exit 3
  fi
  if (( SECONDS > deadline )); then echo "server not ready" >&2; tail -80 "$RUN_DIR/afd.log" >&2; exit 1; fi
  sleep 5
done
echo "server ready after ${SECONDS}s"

if [[ -n "${GREEDY_OUT:-}" ]]; then  # fixed-prompt greedy tokens instead of the sweep
  python "$SCRIPT_DIR/greedy_tokens.py" run --url "http://127.0.0.1:$PORT" --max-tokens "$MAX_TOKENS" --out "$GREEDY_OUT"
  rc=$?
  stop_server
  trap - EXIT
  exit $rc
fi

if [[ -n "${PROMPT_FILE:-}" ]]; then  # real text prompts (routing capture / sanity) instead of the sweep
  python "$SCRIPT_DIR/prompt_file_client.py" --url "http://127.0.0.1:$PORT" --prompt-file "$PROMPT_FILE" \
    --tokenizer "${MINISGL_TOKENIZER_PATH:-$MODEL_PATH}" --num-prompts "${NUM_PROMPTS:-1152}" \
    --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" --timeout "${CLIENT_TIMEOUT:-7200}" \
    --out "$RUN_DIR/prompt_file_client.json" > "$RUN_DIR/client.log" 2>&1 &
  CLIENT_PID=$!
  while kill -0 "$CLIENT_PID" 2>/dev/null; do
    if grep -qs "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log; then
      echo "worker failure:" >&2; grep -hs -A3 "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log | tail -20 >&2
      kill "$CLIENT_PID" 2>/dev/null; exit 3
    fi
    sleep 5
  done
  wait "$CLIENT_PID" || { echo "client failed" >&2; tail -20 "$RUN_DIR/client.log" >&2; exit 4; }
  cat "$RUN_DIR/client.log"
  stop_server
  trap - EXIT
  if [[ -n "${ROUTING_OUT:-}" ]]; then  # per-rank MINISGL_DUMP_EXPERT_LOAD dumps -> one per-layer file
    python "$SCRIPT_DIR/merge_expert_load.py" --dumps "$RUN_DIR/ray_logs/expert_load_attention*.json" \
      --out "$ROUTING_OUT" --note "${ROUTING_NOTE:-}"
  fi
  exit 0
fi

python "$SCRIPT_DIR/bench_client.py" --url "http://127.0.0.1:$PORT" \
  --tokenizer "${MINISGL_TOKENIZER_PATH:-$MODEL_PATH}" --attn-gpus "$ATTN_GPUS" \
  --batch-sizes "$BATCH_SIZES" --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
  ${SWEEP_PROMPT_FILE:+--prompt-file "$SWEEP_PROMPT_FILE"} \
  --timeout "${CLIENT_TIMEOUT:-1800}" --out "$RUN_DIR/sweep.json" > "$RUN_DIR/client.log" 2>&1 &
CLIENT_PID=$!
# a worker whose hot loop died never answers; stop waiting on the client then
while kill -0 "$CLIENT_PID" 2>/dev/null; do
  if grep -qs "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log; then
    echo "worker failure:" >&2; grep -hs -A3 "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log | tail -20 >&2
    kill "$CLIENT_PID" 2>/dev/null; exit 3
  fi
  sleep 5
done
wait "$CLIENT_PID" || { echo "client failed" >&2; tail -20 "$RUN_DIR/client.log" >&2; exit 4; }
cat "$RUN_DIR/client.log"

stop_server
trap - EXIT

python "$SCRIPT_DIR/analyze_step_times.py" --trace "$RUN_DIR/ray_logs/*coordinator*_nvtx_cpu.json" \
  --ag-trace "$RUN_DIR/ray_logs/attention_dp0_rank0_nvtx_cpu.json" \
  --sweep "$RUN_DIR/sweep.json" --out "$RUN_DIR/step_times.json" | tee "$RUN_DIR/step_times.log"
