#!/usr/bin/env bash
# Shared torchrun launch prefix for the MegaMoE measurement runners.
#
# The B200 measurement campaign only ever needed one node, so both runners
# launched `torchrun --standalone`. A GB200 NVL72 domain is 18 nodes of 4 GPUs,
# so an AFD service unit that spans the rack -- 56A16F, 64A8F, 68A4F -- needs a
# multi-node rendezvous. MegaMoE's symmetric buffer is allocated over the
# NVLink/MNNVL fabric, so those ranks are still one NVLink domain; only the
# process launcher changes.
#
# Single-node behaviour is unchanged and bit-identical when MEGAMOE_NNODES is
# unset or 1, so the committed B200 points stay reproducible from this tree.
#
# Contract, set by the submitting sbatch:
#   MEGAMOE_NNODES           number of nodes in the allocation
#   MEGAMOE_NPROC_PER_NODE   GPUs per node (4 on GB200)
#   MEGAMOE_NODE_RANK        this node's index, from SLURM_NODEID
#   MEGAMOE_RDZV_ENDPOINT    host:port of the rendezvous, identical on all nodes
#   MEGAMOE_RDZV_ID_PREFIX   optional namespace, defaults to the Slurm job id
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/_torchrun_launch.sh"
#   megamoe_torchrun_cmd <world_size> <case_tag>
#   "${MEGAMOE_LAUNCH[@]}" script.py --args

megamoe_torchrun_cmd() {
  local world=$1
  local tag=$2
  local nnodes=${MEGAMOE_NNODES:-1}

  if [[ ! ${world} =~ ^[1-9][0-9]*$ ]]; then
    echo "megamoe_torchrun_cmd: world size must be a positive integer: ${world}" >&2
    return 2
  fi

  if (( nnodes <= 1 )); then
    MEGAMOE_LAUNCH=(torchrun --standalone --nproc-per-node="${world}")
    return 0
  fi

  local nproc=${MEGAMOE_NPROC_PER_NODE:?MEGAMOE_NPROC_PER_NODE is required when MEGAMOE_NNODES > 1}
  : "${MEGAMOE_NODE_RANK:?MEGAMOE_NODE_RANK is required when MEGAMOE_NNODES > 1}"
  : "${MEGAMOE_RDZV_ENDPOINT:?MEGAMOE_RDZV_ENDPOINT is required when MEGAMOE_NNODES > 1}"

  if (( nnodes * nproc != world )); then
    # Refuse rather than silently measure a different topology: a mismatch here
    # would hang in rendezvous or, worse, run a world size the profile key does
    # not describe.
    echo "megamoe_torchrun_cmd: MEGAMOE_NNODES(${nnodes}) * MEGAMOE_NPROC_PER_NODE(${nproc})" \
         "!= world size ${world}; the A:F split must be node-aligned" >&2
    return 2
  fi

  # One rendezvous per case. A suite runs many torchrun invocations back to
  # back; reusing a single id lets a stale store from the previous case be
  # picked up by the next one. The tag is computed identically on every node.
  local rdzv_id="${MEGAMOE_RDZV_ID_PREFIX:-${SLURM_JOB_ID:-megamoe}}-${tag}"

  MEGAMOE_LAUNCH=(
    torchrun
    --nnodes="${nnodes}"
    --node-rank="${MEGAMOE_NODE_RANK}"
    --nproc-per-node="${nproc}"
    --rdzv-backend=c10d
    --rdzv-endpoint="${MEGAMOE_RDZV_ENDPOINT}"
    --rdzv-id="${rdzv_id}"
    --rdzv-conf="timeout=${MEGAMOE_RDZV_TIMEOUT:-1800}"
    # A silent restart would re-run a case with a warm cache and a different
    # clock state; a measurement that needs a retry should fail and be resubmitted.
    --max-restarts=0
  )
  return 0
}
