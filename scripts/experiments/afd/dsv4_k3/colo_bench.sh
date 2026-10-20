#!/usr/bin/env bash
# FastAFD colocated (one node, TP = EP) step-time sweep with the scheduler step trace,
# one fastafd.e2e-step-time.v1 JSON per batch size. Run with the FastAFD env active.
#
# Required env: MODEL_PATH, RUN_DIR, E2E_OUT_DIR
# Optional env: MODEL_KEY (deepseek_v4_flash), WEIGHTS (real; dummy adds --dummy-weight),
#   TP (4), ISL (4096), OSL (1024), MAX_TOKENS (64), PROMPT_LEN (ISL + OSL/2 - MAX_TOKENS/2),
#   BATCH_SIZES (8,16,32,64,128,192,256), PROMPT_FILE (real prompts; synthetic otherwise),
#   MEMORY_RATIO (0.85), EXTRA_SERVER_ARGS, COMMIT, WAIT_READY_TIMEOUT (3600)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"  # this tree's minisgl
: "${MODEL_PATH:?}" "${RUN_DIR:?}" "${E2E_OUT_DIR:?}"
MODEL_KEY="${MODEL_KEY:-deepseek_v4_flash}"; WEIGHTS="${WEIGHTS:-real}"; TP="${TP:-4}"
ISL="${ISL:-4096}"; OSL="${OSL:-1024}"; MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_LEN="${PROMPT_LEN:-$(( ISL + OSL / 2 - MAX_TOKENS / 2 ))}"
BATCH_SIZES="${BATCH_SIZES:-8,16,32,64,128,192,256}"
MEMORY_RATIO="${MEMORY_RATIO:-0.85}"
PORT="${PORT:-$((22000 + RANDOM % 8000))}"
mkdir -p "$RUN_DIR" "$E2E_OUT_DIR"
ARCH=$(python -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print(c.get('model_type',''))")
case "$ARCH" in
  deepseek_v4) ATTN_BACKEND=dsv4; PAGE_SIZE=256 ;;
  kimi_k3|kimi_linear) ATTN_BACKEND=kimi; PAGE_SIZE=64 ;;
  *) echo "unsupported model_type $ARCH" >&2; exit 2 ;;
esac
MAX_BS=$(python -c "print(max(int(x) for x in '$BATCH_SIZES'.split(',')))")
MAX_RUN=$(( MAX_BS * TP ))
MAX_SEQ=$(( (PROMPT_LEN + MAX_TOKENS + 64 + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE ))
GRAPH_BS=$(python -c "print(','.join(str(int(b) * $TP) for b in '$BATCH_SIZES'.split(',')))")
EXTRA="${EXTRA_SERVER_ARGS:-}"
[[ "$WEIGHTS" == "dummy" ]] && EXTRA="$EXTRA --dummy-weight"
export MINISGL_SCHED_STEP_TRACE="$RUN_DIR/sched_trace.json"
export MINISGL_ROPE_TABLE_LEN="${MINISGL_ROPE_TABLE_LEN:-$(( MAX_SEQ + 128 ))}"
echo "FastAFD colocated: model=$MODEL_PATH tp=$TP isl=$ISL osl=$OSL prompt=$PROMPT_LEN bs=$BATCH_SIZES max_seq=$MAX_SEQ"
python -m minisgl --model-path "$MODEL_PATH" --tensor-parallel-size "$TP" --attention-backend "$ATTN_BACKEND" \
  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUN" --cuda-graph-bs "$GRAPH_BS" \
  --max-seq-len-override "$MAX_SEQ" --memory-ratio "$MEMORY_RATIO" --cache-type naive \
  --host 127.0.0.1 --port "$PORT" $EXTRA > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
stop_server() { kill -TERM "$SERVER_PID" 2>/dev/null || true; for _ in $(seq 1 45); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 2; done; kill -KILL "$SERVER_PID" 2>/dev/null || true; }
trap stop_server EXIT
deadline=$((SECONDS + ${WAIT_READY_TIMEOUT:-3600}))
until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited" >&2; tail -60 "$RUN_DIR/server.log" >&2; exit 2; fi
  if (( SECONDS > deadline )); then echo "server not ready" >&2; tail -60 "$RUN_DIR/server.log" >&2; exit 1; fi
  sleep 5
done
echo "server ready after ${SECONDS}s"
python "$SCRIPT_DIR/bench_client.py" --url "http://127.0.0.1:$PORT" --tokenizer "${MINISGL_TOKENIZER_PATH:-$MODEL_PATH}" \
  --attn-gpus "$TP" --batch-sizes "$BATCH_SIZES" --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
  ${PROMPT_FILE:+--prompt-file "$PROMPT_FILE"} --timeout "${CLIENT_TIMEOUT:-7200}" \
  --out "$RUN_DIR/sweep.json" 2>&1 | tee "$RUN_DIR/client.log"
rc=${PIPESTATUS[0]}
stop_server; trap - EXIT   # the scheduler flushes the step trace at exit
sleep 3
(( rc == 0 )) || { echo "client rc=$rc" >&2; exit $rc; }
[[ -s "$MINISGL_SCHED_STEP_TRACE" ]] || { echo "no scheduler step trace" >&2; exit 5; }
python "$SCRIPT_DIR/e2e_point.py" --kind colo --run-dir "$RUN_DIR" --out-dir "$E2E_OUT_DIR" --model "$MODEL_KEY" \
  --weights "$WEIGHTS" --isl "$ISL" --osl "$OSL" --software fastafd --commit "${COMMIT:-}" --tp "$TP" --ep "$TP" --nodes 1 \
  --kv-dtype "${KV_DTYPE:-fp8_e4m3}" --moe-precision "${MOE_PRECISION:-mxfp4 routed (colocated MegaMoE), fp8 shared}" \
  --attention-kernels "${ATTN_KERNELS:-FastAFD sglang-path decode kernels}" --job-id "${SLURM_JOB_ID:-}" \
  --case "$(basename "$RUN_DIR")" --note "${NOTE:-}" \
  --prompts "$( [[ -n "${PROMPT_FILE:-}" ]] && echo "real text (prompt file, distinct per request)" || echo "synthetic repeated token" )"
