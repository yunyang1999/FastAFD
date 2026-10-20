#!/usr/bin/env bash
# sglang 0.5.18 colocated baseline on one node: launch the server (TP or TP+EP),
# sweep the per-GPU batch sizes with sgl_bench_client.py and write one
# fastafd.e2e-step-time.v1 JSON per batch size.
#
# Required env: SGL_ENV_SETUP (bash file that activates the sglang env),
#   MODEL_PATH, RUN_DIR, E2E_OUT_DIR, PROMPT_FILE
# Optional env: MODEL_KEY (deepseek_v4_flash), WEIGHTS (real), TP (4), EP (0 = TP-sharded
#   experts; 4 = --ep-size 4), MOE_A2A (none), ISL (4096), OSL (1024), MAX_TOKENS (64),
#   PROMPT_LEN (ISL + OSL/2 - MAX_TOKENS/2), BATCH_SIZES (8,16,32,64,128,192,256),
#   MEM_FRACTION (0.85), MOE_RUNNER (deep_gemm), EXTRA_SGL_ARGS, KV_DTYPE (fp8_e4m3),
#   WAIT_READY_TIMEOUT (3600)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SGL_ENV_SETUP:?}" "${MODEL_PATH:?}" "${RUN_DIR:?}" "${E2E_OUT_DIR:?}" "${PROMPT_FILE:?}"
MODEL_KEY="${MODEL_KEY:-deepseek_v4_flash}"; WEIGHTS="${WEIGHTS:-real}"
TP="${TP:-4}"; EP="${EP:-0}"; MOE_A2A="${MOE_A2A:-none}"
ISL="${ISL:-4096}"; OSL="${OSL:-1024}"; MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_LEN="${PROMPT_LEN:-$(( ISL + OSL / 2 - MAX_TOKENS / 2 ))}"
BATCH_SIZES="${BATCH_SIZES:-8,16,32,64,128,192,256}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"; KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
# sglang's auto MoE runner for the FP8 block-quant experts on SM100 is the Triton fused MoE,
# which fails the hidden-size check at graph capture for DeepSeek-V4-Flash; DeepGEMM is the
# production FP8 grouped-GEMM runner for the DeepSeek family.
MOE_RUNNER="${MOE_RUNNER:-deep_gemm}"
PORT="${PORT:-$((21000 + RANDOM % 9000))}"
mkdir -p "$RUN_DIR" "$E2E_OUT_DIR"
# shellcheck source=/dev/null
source "$SGL_ENV_SETUP"
MAX_BS=$(python -c "print(max(int(x) for x in '$BATCH_SIZES'.split(',')))")
MAX_RUN=$(( MAX_BS * TP ))
CTX=$(( PROMPT_LEN + MAX_TOKENS + 256 ))
GRAPH_BS=$(python -c "print(' '.join(str(int(b) * $TP) for b in '$BATCH_SIZES'.split(',')))")
EP_ARGS=""
(( EP > 1 )) && EP_ARGS="--ep-size $EP"
[[ "$MOE_A2A" != "none" ]] && EP_ARGS="$EP_ARGS --moe-a2a-backend $MOE_A2A"
echo "sglang colocated: model=$MODEL_PATH tp=$TP ep=$EP a2a=$MOE_A2A isl=$ISL osl=$OSL prompt=$PROMPT_LEN bs=$BATCH_SIZES ctx=$CTX"
python -m sglang.launch_server --model-path "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --tp "$TP" $EP_ARGS --kv-cache-dtype "$KV_DTYPE" --mem-fraction-static "$MEM_FRACTION" \
  --moe-runner-backend "$MOE_RUNNER" \
  --max-running-requests "$MAX_RUN" --context-length "$CTX" --cuda-graph-max-bs-decode "$MAX_RUN" \
  --cuda-graph-bs-decode $GRAPH_BS --decode-log-interval 10 --log-level info \
  ${EXTRA_SGL_ARGS:-} > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
stop_server() { kill -TERM "$SERVER_PID" 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 2; done; kill -KILL "$SERVER_PID" 2>/dev/null || true; }
trap stop_server EXIT
deadline=$((SECONDS + ${WAIT_READY_TIMEOUT:-3600}))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited" >&2; tail -60 "$RUN_DIR/server.log" >&2; exit 2; fi
  if (( SECONDS > deadline )); then echo "server not ready" >&2; tail -60 "$RUN_DIR/server.log" >&2; exit 1; fi
  sleep 5
done
echo "server ready after ${SECONDS}s"
curl -fsS "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" 2>/dev/null || true
SGL_VERSION=$(python -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "0.5.18")
python "$SCRIPT_DIR/sgl_bench_client.py" --url "http://127.0.0.1:$PORT" --tokenizer "$MODEL_PATH" --gpus "$TP" \
  --batch-sizes "$BATCH_SIZES" --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" --prompt-file "$PROMPT_FILE" \
  --server-log "$RUN_DIR/server.log" --out "$RUN_DIR/sgl_sweep.json" 2>&1 | tee "$RUN_DIR/client.log"
rc=${PIPESTATUS[0]}
stop_server; trap - EXIT
(( rc == 0 )) || { echo "client rc=$rc" >&2; exit $rc; }
DEP_NOTE="sglang $SGL_VERSION, --tp $TP${EP_ARGS:+ $EP_ARGS}, kv $KV_DTYPE, moe_runner $MOE_RUNNER"
python "$SCRIPT_DIR/e2e_point.py" --kind sglang --run-dir "$RUN_DIR" --out-dir "$E2E_OUT_DIR" --model "$MODEL_KEY" \
  --weights "$WEIGHTS" --isl "$ISL" --osl "$OSL" --software sglang --version "$SGL_VERSION" --tp "$TP" --ep "$EP" \
  --nodes 1 --kv-dtype "$KV_DTYPE" --moe-precision "${MOE_PRECISION:-fp8 block-128 (checkpoint)}" \
  --attention-kernels "sglang dsv4 backend (FlashMLA sparse decode, DeepGEMM indexer)" \
  --job-id "${SLURM_JOB_ID:-}" --case "$(basename "$RUN_DIR")" --note "$DEP_NOTE" \
  --prompts "real text (prompt file, distinct per request)" \
  --extra "{\"server_flags\": \"--tp $TP $EP_ARGS --kv-cache-dtype $KV_DTYPE --mem-fraction-static $MEM_FRACTION --moe-runner-backend $MOE_RUNNER --cuda-graph-bs-decode ... ${EXTRA_SGL_ARGS:-}\"}"
