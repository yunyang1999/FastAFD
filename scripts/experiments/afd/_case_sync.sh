#!/usr/bin/env bash
# Sourced by the case-matrix runners. Multi-node only.
#
# Every node runs the case loop itself and the nodes meet per case through
# torchrun's c10d rendezvous, whose store lives in node 0's agent. A case that
# fails (e.g. an OOM at the top of a grid) tears that store down while other
# nodes are still inside the case; without a barrier the nodes drift apart and
# every later case fails to rendezvous until the job times out. So each case
# attempt gets its own rendezvous port, all nodes meet on the shared filesystem
# after every attempt, and the nodes decide together whether it failed.
#
# Ports stay below the kernel's ephemeral range (32768+ on lyris): a store port
# in that range can already be held by an outgoing connection, and the other
# nodes then time out connecting to it.

case_sync_init() {
  CASE_SYNC_NNODES=${MEGAMOE_NNODES:-1}
  CASE_SYNC_IDX=0
  CASE_SYNC_FAILED=0
  if (( CASE_SYNC_NNODES > 1 )); then
    CASE_SYNC_DIR=${WORKSPACE:?WORKSPACE is required for multi-node case sync}/cache/case_sync/${SLURM_JOB_ID:?}
    CASE_SYNC_HOST=${MEGAMOE_RDZV_ENDPOINT%:*}
    CASE_SYNC_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
    mkdir -p "${CASE_SYNC_DIR}"
  fi
}

# Rendezvous endpoint for the current case attempt.
case_sync_endpoint() {
  if (( CASE_SYNC_NNODES > 1 )); then
    echo "${CASE_SYNC_HOST}:$(( CASE_SYNC_PORT + CASE_SYNC_IDX ))"
  else
    echo "${MEGAMOE_RDZV_ENDPOINT:-}"
  fi
}

# case_sync_barrier RC: wait until every node has finished the current attempt,
# then set CASE_SYNC_FAILED=1 if the attempt failed on ANY node, and advance.
case_sync_barrier() {
  local rc=$1
  CASE_SYNC_FAILED=$(( rc != 0 ))
  if (( CASE_SYNC_NNODES > 1 )); then
    echo "${rc}" > "${CASE_SYNC_DIR}/${CASE_SYNC_IDX}.${MEGAMOE_NODE_RANK:?}"
    local deadline=$(( SECONDS + ${CASE_SYNC_TIMEOUT:-1800} ))
    while (( $(find "${CASE_SYNC_DIR}" -maxdepth 1 -name "${CASE_SYNC_IDX}.*" | wc -l) < CASE_SYNC_NNODES )); do
      if (( SECONDS >= deadline )); then
        echo "case sync: timed out waiting for all nodes after attempt ${CASE_SYNC_IDX}" >&2
        exit 3
      fi
      sleep 2
    done
    if grep -qv '^0$' "${CASE_SYNC_DIR}/${CASE_SYNC_IDX}".*; then
      CASE_SYNC_FAILED=1
    fi
  fi
  CASE_SYNC_IDX=$(( CASE_SYNC_IDX + 1 ))
}
