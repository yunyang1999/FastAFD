#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/workspace}
fastafd_root=${FASTAFD_ROOT:-${workspace}/FastAFD-megamoe-multimodel}
sglang_python_root=${SGLANG_PYTHON_ROOT:-/sgl-workspace/sglang/python}
results_dir=${RESULTS_DIR:-${workspace}/results/megamoe-colocated-multimodel}
model=${MODEL:?set MODEL to an FP4 key from megamoe_model_profiles.py}
ep_size=${EP_SIZE:-8}
tokens_per_rank=${TOKENS_PER_RANK:-64}
mtp_nextn=${MTP_NEXTN:-0}
layers=${LAYERS:-0}
routing=${ROUTING:-balanced}
hot_expert_fraction=${HOT_EXPERT_FRACTION:-}
replay_distribution=${REPLAY_DISTRIBUTION:-}
backend=${BACKEND:-mega}
measurement_order=${MEASUREMENT_ORDER:-grouped}
weight_slots=${WEIGHT_SLOTS:-2}
warmups=${WARMUPS:-30}
iterations=${ITERATIONS:-30}

mkdir -p \
  "${workspace}/runtime_home" \
  "${workspace}/cache/deepgemm-multimodel" \
  "${results_dir}"
export HOME="${workspace}/runtime_home"
export XDG_CACHE_HOME="${workspace}/cache"
export PYTHONUNBUFFERED=1
export MINISGL_DEEPGEMM_BUILD_DIR="${workspace}/cache/deepgemm-multimodel"
export PYTHONPATH="${workspace}/python_deps:${sglang_python_root}:${fastafd_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export MEASUREMENT_CONTAINER_IMAGE=${MEASUREMENT_CONTAINER_IMAGE:-unknown}

suffix="${model}_${ep_size}ep_t${tokens_per_rank}_n${mtp_nextn}_${routing}_${backend}"

args=()
if [[ -n "${hot_expert_fraction}" ]]; then
  args+=(--hot-expert-fraction "${hot_expert_fraction}")
fi
if [[ -n "${replay_distribution}" ]]; then
  args+=(--replay-distribution "${replay_distribution}")
fi
# shellcheck source=./_torchrun_launch.sh
source "$(dirname "${BASH_SOURCE[0]}")/_torchrun_launch.sh"
megamoe_torchrun_cmd "${ep_size}" "${suffix}"

"${MEGAMOE_LAUNCH[@]}" \
  "${fastafd_root}/scripts/experiments/afd/benchmark_megamoe_colocated_models.py" \
  --model "${model}" \
  --output "${results_dir}/${suffix}.json" \
  --ep-size "${ep_size}" \
  --tokens-per-rank "${tokens_per_rank}" \
  --mtp-nextn "${mtp_nextn}" \
  --layers "${layers}" \
  --routing "${routing}" \
  --backend "${backend}" \
  --measurement-order "${measurement_order}" \
  --weight-slots "${weight_slots}" \
  --warmups "${warmups}" \
  --iterations "${iterations}" \
  "${args[@]}"
