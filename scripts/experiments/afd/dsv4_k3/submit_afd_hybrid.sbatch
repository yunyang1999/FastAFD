#!/usr/bin/env bash
# Ray cluster bootstrap for the DeepSeek-V4 / Kimi-K3 AFD runs on GB200.
#
# Node 0 of the allocation is a GPU-less Ray head (it runs the AFD driver /
# coordinator / HTTP server); nodes 1.. join as 4-GPU workers. RUN_SCRIPT then
# runs on the head with RAY_ADDRESS / MINISGL_RAY_ADDRESS exported; it may call
# run_afd_hybrid_bench.sh any number of times (one server per call).
#
# Required env: CONDA_ROOT, FASTAFD_ROOT, RUN_SCRIPT (path of a bash script)
# Optional env: CONDA_ENV (default minisgl-cuda130), ENV_SETUP (a bash file
#   sourced on every node after conda activation, e.g. cache directories)
#
#   CONDA_ROOT=... FASTAFD_ROOT=... RUN_SCRIPT=... \
#   sbatch -N $((1 + GPU_NODES)) --segment=$((1 + GPU_NODES)) --account=... \
#     --partition=... --time=01:00:00 --job-name=... -o LOG submit_afd_hybrid.sbatch
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail
: "${CONDA_ROOT:?}" "${FASTAFD_ROOT:?}" "${RUN_SCRIPT:?}"
export CONDA_ROOT FASTAFD_ROOT RUN_SCRIPT
export CONDA_ENV="${CONDA_ENV:-minisgl-cuda130}"
export ENV_SETUP="${ENV_SETUP:-}"

nnodes=${SLURM_JOB_NUM_NODES:?not running under Slurm}
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node=${nodes[0]}
cpus_per_node=${CPUS_PER_NODE:-140}
gpus_per_node=${GPUS_PER_NODE:-4}
# outside Ray's default worker-port range (10002-19999)
ray_port=$((30000 + SLURM_JOB_ID % 4000))
dash_port=$((35000 + SLURM_JOB_ID % 4000))
head_addr="${head_node}:${ray_port}"
export SLURM_CPU_BIND=none
echo "job=${SLURM_JOB_ID} nodes=${nnodes} head=${head_node} ray=${head_addr} start=$(date -u +%FT%TZ)"

activate() {
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  if [[ -n "${ENV_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_SETUP}"
  fi
  set -u
  export PYTHONPATH="${FASTAFD_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
  export OPENBLAS_NUM_THREADS=1
}

head_pid=""
workers_pid=""
cleanup() {
  echo "=== tearing down ray ($(date -u +%FT%TZ)) ==="
  [[ -n "${workers_pid}" ]] && kill "${workers_pid}" 2>/dev/null || true
  [[ -n "${head_pid}" ]] && kill "${head_pid}" 2>/dev/null || true
  srun --overlap --ntasks="${nnodes}" --ntasks-per-node=1 --export=ALL bash -c "
    $(declare -f activate); activate
    ray stop --force >/dev/null 2>&1 || true
  " || true
}
trap cleanup EXIT

srun --ntasks="${nnodes}" --ntasks-per-node=1 --export=ALL bash -c "
  $(declare -f activate); activate
  ray stop --force >/dev/null 2>&1 || true
"

# The ray daemons must live inside a kept-alive srun step: Slurm kills a step's
# cgroup (daemons included) once its srun returns.
srun --nodes=1 --ntasks=1 -w "${head_node}" --export=ALL bash -c "
  $(declare -f activate); activate
  ray start --head --port=${ray_port} --dashboard-port=${dash_port} --dashboard-host=0.0.0.0 \
    --num-gpus=0 --num-cpus=${cpus_per_node} --temp-dir=/tmp/ray-\${SLURM_JOB_ID}
  sleep infinity
" &
head_pid=$!
sleep 10
if (( nnodes > 1 )); then
  srun --nodes=$((nnodes - 1)) --ntasks=$((nnodes - 1)) --ntasks-per-node=1 \
    --exclude="${head_node}" --export=ALL bash -c "
    $(declare -f activate); activate
    ray start --address=${head_addr} --num-gpus=${gpus_per_node} --num-cpus=${cpus_per_node} \
      --temp-dir=/tmp/ray-\${SLURM_JOB_ID}
    sleep infinity
  " &
  workers_pid=$!
fi

srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" --export=ALL bash -c "
  $(declare -f activate); activate
  python - <<'PY'
import sys, time
import ray
ray.init(address='${head_addr}', logging_level='ERROR')
deadline = time.time() + 180
alive = []
while time.time() < deadline:
    alive = [n for n in ray.nodes() if n.get('Alive')]
    if len(alive) >= ${nnodes}:
        print(f'ray: {len(alive)}/${nnodes} nodes alive')
        sys.exit(0)
    time.sleep(2)
sys.exit(f'ray: only {len(alive)}/${nnodes} nodes joined')
PY
"

srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" --export=ALL bash -c "
  $(declare -f activate); activate
  cd '${FASTAFD_ROOT}'
  export RAY_ADDRESS='${head_addr}' MINISGL_RAY_ADDRESS='${head_addr}' AFD_GPUS_PER_NODE=${gpus_per_node}
  bash '${RUN_SCRIPT}'
"
echo "end=$(date -u +%FT%TZ)"
