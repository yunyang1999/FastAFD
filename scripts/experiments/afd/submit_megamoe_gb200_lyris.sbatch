#!/usr/bin/env bash
# MegaMoE measurement on a GB200 NVL72 domain (ASE pre-cluster "lyris").
#
# Sibling of submit_megamoe_m2n_b200.sbatch. Four things differ from the B200
# site and each one is load-bearing:
#
#   1. A node is 4 GPUs, not 8. A 72-GPU NVL72 service unit is 18 nodes, so the
#      runners launch a multi-node torchrun rendezvous instead of --standalone.
#   2. GPUs are not GRES-managed here (`Gres=(null)`), so the allocation is
#      whole nodes via -N/--exclusive; --gpus= would fail.
#   3. There is no site SGLang container for aarch64. The measurement runs in
#      the conda environment FastAFD's own README validates on aarch64/CUDA 13,
#      so CONTAINER_IMAGE is replaced by CONDA_ROOT/CONDA_ENV.
#   4. TopologyPlugin=topology/block with BlockSize=18 and SwitchType=
#      switch/nvidia_imex: Slurm packs the allocation into one NVLink block and
#      provisions the IMEX channel itself. --segment makes that explicit rather
#      than relying on the packing heuristic, because a split allocation would
#      not be one NVLink domain and MegaMoE's symmetric buffer would fail there.
#
# Node count is deliberately NOT fixed here -- pass -N on the sbatch line:
#   -N1  (4 GPUs)   smoke / colocated ep<=4
#   -N2  (8 GPUs)   the B200-comparable 4A4F cross-check
#   -N18 (72 GPUs)  the NVL72 service unit
#
#SBATCH --account=general_sa
#SBATCH --partition=gb200-backfill
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=general_sa-afd:megamoe

set -euo pipefail

measurement_system=${MEASUREMENT_SYSTEM:-gb200}
run_script=${RUN_SCRIPT:-run_megamoe_m2n_model_benchmark.sh}
source_root=${SOURCE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source_root=$(readlink -f "${source_root}")
host_root=${HOST_ROOT:-$(dirname "${source_root}")}
host_root=$(readlink -f "${host_root}")
conda_root=${CONDA_ROOT:-${host_root}/miniforge3}
conda_env=${CONDA_ENV:-minisgl-cuda130}

case "${source_root}" in
  "${host_root}"/*) ;;
  *)
    echo "SOURCE_ROOT must be under HOST_ROOT: ${source_root}" >&2
    exit 2
    ;;
esac
case "${run_script}" in
  run_megamoe_m2n_model_benchmark.sh|run_megamoe_m2n_model_smoke_suite.sh|run_megamoe_m2n_model_suite.sh|run_megamoe_m2n_model_case_matrix.sh|run_megamoe_colocated_model_benchmark.sh|run_megamoe_colocated_model_smoke_suite.sh|run_megamoe_colocated_model_suite.sh|run_megamoe_colocated_model_case_matrix.sh) ;;
  *)
    echo "Unsupported RUN_SCRIPT: ${run_script}" >&2
    exit 2
    ;;
esac

# Same provenance contract as the B200 sbatch: every result JSON records the
# exact tree it was produced from.
source_commit=$(git -C "${source_root}" rev-parse HEAD)
source_branch=$(git -C "${source_root}" branch --show-current)
source_diff_sha256=$(git -C "${source_root}" diff --binary | sha256sum | cut -d' ' -f1)
if [[ -n $(git -C "${source_root}" status --porcelain) ]]; then
  source_dirty=true
else
  source_dirty=false
fi
source_tree_sha256=$(
  cd "${source_root}"
  git ls-files --cached --others --exclude-standard -z |
    sort -z |
    while IFS= read -r -d '' path; do
      printf '%s\0' "${path}"
      if [[ -L "${path}" ]]; then
        readlink "${path}" | sha256sum
      else
        sha256sum <"${path}"
      fi
    done |
    sha256sum |
    cut -d' ' -f1
)

nnodes=${SLURM_JOB_NUM_NODES:?not running under Slurm}
gpus_per_node=${GPUS_PER_NODE:-4}
total_gpus=$((nnodes * gpus_per_node))

# One NVLink domain, or the symmetric buffer cannot be allocated. topology/block
# should already guarantee this; verify rather than assume, because a silently
# split allocation would surface as an opaque rendezvous or CUDA IPC failure
# tens of minutes into a sweep.
blocks=$(sinfo -h -N -n "${SLURM_JOB_NODELIST}" -o '%f' |
         tr ',' '\n' | grep '^nvlblk' | sort -u)
if [[ $(wc -l <<<"${blocks}") -ne 1 ]]; then
  echo "allocation spans multiple NVLink blocks, not one NVL72 domain:" >&2
  echo "${blocks}" >&2
  exit 2
fi
echo "NVLink domain: ${blocks} (${nnodes} nodes, ${total_gpus} GPUs)"

# Rendezvous: node 0 of the allocation, port derived from the job id so two
# concurrent jobs on the same node cannot collide.
rdzv_host=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
rdzv_port=$((20000 + SLURM_JOB_ID % 20000))

results_dir=${RESULTS_DIR:-${host_root}/results/megamoe-gb200}
mkdir -p "${results_dir}" "${host_root}/runtime_home" "${host_root}/cache"

# The B200 wrapper needed this on NUMA-split allocations; harmless here and
# keeps the failure mode from reappearing on a differently-partitioned node.
export SLURM_CPU_BIND=none

srun \
  --ntasks-per-node=1 \
  --export=ALL \
  bash -c "
    set -euo pipefail
    # conda-forge's cuda-nvcc activate.d hook appends to NVCC_PREPEND_FLAGS
    # without initialising it, which aborts under \`set -u\`. Relax nounset
    # across activation only.
    set +u
    source '${conda_root}/etc/profile.d/conda.sh'
    conda activate '${conda_env}'
    set -u
    export WORKSPACE='${host_root}'
    export FASTAFD_ROOT='${source_root}'
    export PYTHON_DEPS_ROOT='${PYTHON_DEPS_ROOT:-${host_root}/python_deps}'
    export RESULTS_DIR='${results_dir}'
    export MEASUREMENT_SYSTEM='${measurement_system}'
    export MEASUREMENT_CONTAINER_IMAGE='conda:${conda_env}'
    export MEASUREMENT_SOURCE_COMMIT='${source_commit}'
    export MEASUREMENT_SOURCE_BRANCH='${source_branch}'
    export MEASUREMENT_SOURCE_DIRTY='${source_dirty}'
    export MEASUREMENT_SOURCE_DIFF_SHA256='${source_diff_sha256}'
    export MEASUREMENT_SOURCE_TREE_SHA256='${source_tree_sha256}'
    export TOTAL_GPUS='${total_gpus}'
    export MEGAMOE_NNODES='${nnodes}'
    export MEGAMOE_NPROC_PER_NODE='${gpus_per_node}'
    export MEGAMOE_NODE_RANK=\"\${SLURM_NODEID}\"
    export MEGAMOE_RDZV_ENDPOINT='${rdzv_host}:${rdzv_port}'
    export MEGAMOE_RDZV_ID_PREFIX='${SLURM_JOB_ID}'
    exec bash '${source_root}/scripts/experiments/afd/${run_script}'
  "
