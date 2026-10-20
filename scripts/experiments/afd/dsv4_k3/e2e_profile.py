"""Per-step GPU kernel decomposition of an AFD worker from a torch.profiler chrome trace.

The AFD workers write ``torch_profile_<role>_dp0_rank0.json`` (+ ``_meta.json``) when
run with ``MINISGL_AFD_TORCH_PROFILE=decode:<first>:<count>``; every profiled step's
launches are wrapped in ``record_function("afd_step[<id>]")``. Kernels are attributed
to steps through the CUDA runtime launch events (same correlation id) that fall inside
a step's annotation, then grouped by name into: attention core, attention-side norms /
RoPE / mHC / residual mixing ("norm"), router ("gate"), MegaMoE M2N kernels ("moe": on the
A side the dispatch/combine side of the stage, which also waits for the experts; on the F
side the expert kernel), other collectives ("comm": TP all-reduce), quantisation ("quant"),
dense GEMMs ("dense"), sampling, copies, other. Per step the GPU span (first kernel
start to last kernel end), the busy time (union of kernel intervals) and the idle time
(span - busy) are reported.

  python e2e_profile.py --trace torch_profile_attention_dp0_rank0.json --role attention \
      --step-ms 20.5 --out profile_attention_b64.json [--keep-steps 2:]

Colocated engine (``--role colo``): the scheduler writes ``torch_profile_sched[_bs<N>].json``
(``MINISGL_SCHED_TORCH_PROFILE``, steps ``sched_step[<i>]`` each ended by a device
synchronize), so kernels -- CUDA-graph replays included -- are attributed to steps by
timestamp. With ``MINISGL_NVTX_TORCH_RECORD=1`` (eager run) the model's NVTX ranges are
in the trace as well and every kernel is also attributed to its innermost label
(DSV4AttnCore, DSV4AttnIndex, DSV4Attn, DSV4AttnOut, DSV4Gate, DSV4Routed, DSV4Shared,
DSV4Layer_<l>, DSV4Head), which the output groups into the step buckets attention core /
attention projections + all-reduce / gate / routed MegaMoE / shared expert / mHC+norms /
lm_head / outside-layers (sampler, copies).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics

CATEGORIES = [  # first match wins
    ("moe", re.compile(r"mega_moe|megamoe|m2n|sm100_mega|MegaMoE", re.I)),
    ("attn", re.compile(r"fmha|mla|sparse_decode|kda|conv1d|causal_conv|chunk_|fused_recurrent|attn_res|"
                        r"paged_mqa|topk_main|topk_transform|_compress_decode|index_key|fused_norm_rope_indexer|"
                        r"q_indexer|_mla_query|_assemble_index|fmhaReduction|flash", re.I)),
    ("gate", re.compile(r"router|moe_fused_gate|radix|_route|hash_topk|sqrtsoftplus|gate_kernel", re.I)),
    ("comm", re.compile(r"dispatch|combine|a2a|all_reduce|allreduce|custom_all_reduce|nccl|AllReduce|"
                        r"reduce_scatter|all_gather", re.I)),
    ("quant", re.compile(r"quant|cast_fp8|per_token|fp8_wo_a", re.I)),
    ("norm", re.compile(r"norm|rope|hc_pre|hc_post|hc_head|mhc|sinkhorn|residual|situ|silu|swiglu|act", re.I)),
    ("dense", re.compile(r"nvjet|gemm|cutlass|blockscaled|Gemm|cublas|splitK|tgv", re.I)),
    ("sample", re.compile(r"sampl|softmax|argmax|multinomial|topk_|cumsum|sort|gumbel|exponential", re.I)),
    ("copy", re.compile(r"Memcpy|memcpy|Memset|memset|copy_|elementwise|fill|index_|gather|scatter|cat_|"
                        r"vectorized|unrolled|reduce_kernel|where", re.I)),
]


LABEL_PREFIXES = ("DSV4", "AFD_", "K3", "Kimi", "Layer_", "MHA")

BUCKETS = [  # innermost model label -> step bucket (colocated DSV4)
    ("attention_core", ("DSV4AttnCore", "DSV4AttnIndex")),
    ("attention_proj_allreduce", ("DSV4Attn", "DSV4AttnOut")),
    ("gate", ("DSV4Gate",)),
    ("routed_megamoe", ("DSV4Routed",)),
    ("shared_expert", ("DSV4Shared",)),
    ("mhc_norm_residual", ("DSV4Layer",)),
    ("lm_head", ("DSV4Head",)),
]


def bucket_of(label: str) -> str:
    for b, labs in BUCKETS:
        if label in labs:
            return b
    return "outside_layers" if label == "(none)" else f"other:{label}"


def categorize(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def union_len(intervals: list[tuple[float, float]]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--step-ms", type=float, default=None, help="unprofiled step period at this batch (for idle vs T)")
    ap.add_argument("--keep-steps", default="1:", help="python slice of the profiled steps to average (drop the first)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--attribute", choices=["auto", "corr", "ts"], default="auto",
                    help="kernel->step attribution: launch correlation (AFD workers) or timestamp (synced sched steps)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.load(open(args.trace))
    evs = data["traceEvents"] if isinstance(data, dict) else data
    meta_path = args.trace.replace(".json", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    steps = []  # (step_id, tid, ts, te)
    labels = []  # (label, tid, ts, te) model NVTX ranges mirrored as record_function
    runtime = {}  # correlation -> (ts, tid)
    kernels = []
    synced_steps = False
    step_rx = re.compile(r"(afd|sched)_step\[(\d+)\]")
    for e in evs:
        cat = e.get("cat", "")
        name = str(e.get("name", ""))
        if cat == "user_annotation" or (cat == "cpu_op" and (step_rx.match(name) or name.startswith(LABEL_PREFIXES))):
            m = step_rx.match(name)
            if m:
                steps.append((int(m.group(2)), e.get("tid"), e["ts"], e["ts"] + e.get("dur", 0)))
                synced_steps = synced_steps or m.group(1) == "sched"
            elif name.startswith(LABEL_PREFIXES):
                labels.append((re.sub(r"_\d+$", "", name), e.get("tid"), e["ts"], e["ts"] + e.get("dur", 0)))
        elif cat in ("cuda_runtime", "cuda_driver"):
            # the EG persistent MegaMoE kernel is launched through the driver API
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                runtime[corr] = (e["ts"], e.get("tid"))
        elif cat == "kernel":
            kernels.append(e)
    steps.sort(key=lambda s: s[2])
    labels.sort(key=lambda l: l[3] - l[2])  # innermost (shortest) first
    by_ts = args.attribute == "ts" or (args.attribute == "auto" and synced_steps)

    def step_of(k):
        if by_ts:  # steps end with a device synchronize: the kernel lies inside its step
            ts = k["ts"]
            for sid, _, s0, s1 in steps:
                if s0 <= ts <= s1:
                    return sid
            return None
        r = runtime.get(k.get("args", {}).get("correlation"))
        if r is None:
            return None
        ts, tid = r
        for sid, stid, s0, s1 in steps:
            if s0 <= ts <= s1 and (stid == tid or stid is None):
                return sid
        return None

    def label_of(k):
        r = runtime.get(k.get("args", {}).get("correlation"))
        if r is None:
            return None
        ts, tid = r
        for lab, ltid, l0, l1 in labels:
            if l0 <= ts <= l1 and (ltid == tid or ltid is None):
                return lab
        return None

    per_step: dict[int, dict] = {}
    unattributed = 0.0
    names: dict[str, list[float]] = {}
    for k in kernels:
        sid = step_of(k)
        dur = k.get("dur", 0) / 1e3  # us -> ms
        name = k["name"]
        if sid is None:
            unattributed += dur
            continue
        st = per_step.setdefault(sid, {"cats": {}, "intervals": [], "n_kernels": 0, "labels": {}})
        c = categorize(name)
        st["cats"][c] = st["cats"].get(c, 0.0) + dur
        st["intervals"].append((k["ts"] / 1e3, (k["ts"] + k.get("dur", 0)) / 1e3))
        st["n_kernels"] += 1
        names.setdefault(f"{c}|{name}", []).append(dur)
        if labels:
            lab = label_of(k) or "(none)"
            lc = st["labels"].setdefault(lab, {})
            lc[c] = lc.get(c, 0.0) + dur

    sids = sorted(per_step)
    sl = slice(*[int(x) if x else None for x in args.keep_steps.split(":")])
    kept = sids[sl] if sids else []
    rows = []
    for sid in kept:
        st = per_step[sid]
        span = max(e for _, e in st["intervals"]) - min(s for s, _ in st["intervals"])
        busy = union_len(st["intervals"])
        row = {"step_id": sid, "span_ms": span, "busy_ms": busy, "idle_ms": span - busy,
               "kernel_sum_ms": sum(st["cats"].values()), "n_kernels": st["n_kernels"]}
        row.update({f"{c}_ms": v for c, v in st["cats"].items()})
        rows.append(row)
    keys = sorted({k for r in rows for k in r if k != "step_id"})
    med = {k: statistics.median([r.get(k, 0.0) for r in rows]) for k in keys} if rows else {}
    # per-label and per-bucket medians (ms per step, by kernel category)
    label_med: dict[str, dict[str, float]] = {}
    all_labels = sorted({lab for sid in kept for lab in per_step[sid]["labels"]})
    for lab in all_labels:
        cats = sorted({c for sid in kept for c in per_step[sid]["labels"].get(lab, {})})
        label_med[lab] = {c: statistics.median([per_step[sid]["labels"].get(lab, {}).get(c, 0.0) for sid in kept]) for c in cats}
        label_med[lab]["total"] = statistics.median([sum(per_step[sid]["labels"].get(lab, {}).values()) for sid in kept])
    bucket_med: dict[str, dict[str, float]] = {}
    if all_labels:
        for sid in kept:
            per_step[sid]["buckets"] = {}
            for lab, cats in per_step[sid]["labels"].items():
                b = bucket_of(lab)
                bc = per_step[sid]["buckets"].setdefault(b, {})
                for c, v in cats.items():
                    bc[c] = bc.get(c, 0.0) + v
        for b in sorted({b for sid in kept for b in per_step[sid]["buckets"]}):
            cats = sorted({c for sid in kept for c in per_step[sid]["buckets"].get(b, {})})
            bucket_med[b] = {c: statistics.median([per_step[sid]["buckets"].get(b, {}).get(c, 0.0) for sid in kept]) for c in cats}
            bucket_med[b]["total"] = statistics.median([sum(per_step[sid]["buckets"].get(b, {}).values()) for sid in kept])
    top = sorted(((sum(v) / max(1, len(kept)), k, len(v)) for k, v in names.items()), reverse=True)[: args.top]
    out = {
        "schema": "fastafd.e2e-profile.v1",
        "role": args.role,
        "trace_meta": {k: v for k, v in meta.items() if k != "key_averages"},
        "steps_profiled": sids,
        "steps_used": kept,
        "step_ms_unprofiled": args.step_ms,
        "median": med,
        "per_step": rows,
        "unattributed_kernel_ms_total": unattributed,
        "attribution": "timestamp" if by_ts else "correlation",
        "median_by_label": label_med,
        "median_by_bucket": bucket_med,
        "top_kernels_ms_per_step": [{"category": k.split("|")[0], "name": k.split("|", 1)[1][:120],
                                     "ms_per_step": round(v, 4), "calls": n} for v, k, n in top],
    }
    if args.step_ms and med:
        out["median"]["wait_outside_span_ms"] = args.step_ms - med.get("span_ms", 0.0)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps({"role": args.role, "steps": len(kept), **{k: round(v, 3) for k, v in med.items()}}))


if __name__ == "__main__":
    main()
