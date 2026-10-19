#!/usr/bin/env bash
set -euo pipefail

here=$(dirname "${BASH_SOURCE[0]}")
runner=${here}/run_megamoe_m2n_model_benchmark.sh
: "${CASE_MATRIX:?Set CASE_MATRIX to a whitespace-separated case matrix}"
# Multi-node: per-attempt rendezvous port, a barrier after every attempt, and
# a shared verdict on whether it failed.
source "${here}/_case_sync.sh"
case_sync_init

while read -r -u 9 model split sequences nextn microbatches backend remainder; do
  if [[ -z ${model:-} || ${model:0:1} == "#" ]]; then
    continue
  fi
  if [[ -n ${remainder:-} ]]; then
    echo "Unexpected extra fields in CASE_MATRIX row: ${model} ${split} ${sequences} ${nextn} ${microbatches} ${backend} ${remainder}" >&2
    exit 2
  fi
  if [[ ! ${split:-} =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "Invalid A:F split in CASE_MATRIX: ${split:-<missing>}" >&2
    exit 2
  fi
  IFS=: read -r ag_size eg_size <<< "${split}"
  # CASE_ATTEMPTS (default 2): a failed attempt on any node is retried by all
  # nodes on a fresh port -- rendezvous connects occasionally time out.
  for (( attempt = 1; attempt <= ${CASE_ATTEMPTS:-2}; attempt++ )); do
    rc=0
    MODEL=${model} \
    AG_SIZE=${ag_size} \
    EG_SIZE=${eg_size} \
    SEQUENCES_PER_AG_RANK=${sequences} \
    MTP_NEXTN=${nextn} \
    MICROBATCHES=${microbatches} \
    BACKEND=${backend} \
    MEGAMOE_RDZV_ENDPOINT=$(case_sync_endpoint) \
    "${runner}" || rc=$?
    case_sync_barrier "${rc}"
    (( CASE_SYNC_FAILED )) || break
    echo "case attempt ${attempt} failed (rc=${rc} here): ${model} ${split} ${sequences} ${nextn} ${microbatches}" >&2
  done
  if (( CASE_SYNC_FAILED )); then
    # CASE_MATRIX_CONTINUE=1: record the failed row and move on; the barrier
    # keeps every node on the same row.
    if [[ ${CASE_MATRIX_CONTINUE:-0} == 1 ]]; then
      echo "case failed: ${model} ${split} ${sequences} ${nextn} ${microbatches}" >&2
    else
      exit 1
    fi
  fi
done 9< "${CASE_MATRIX}"
