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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.load(open(args.trace))
    evs = data["traceEvents"] if isinstance(data, dict) else data
    meta_path = args.trace.replace(".json", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    steps = []  # (step_id, tid, ts, te)
    runtime = {}  # correlation -> (ts, tid)
    kernels = []
    for e in evs:
        cat = e.get("cat", "")
        if cat == "user_annotation" or (cat == "cpu_op" and str(e.get("name", "")).startswith("afd_step[")):
            m = re.match(r"afd_step\[(\d+)\]", str(e.get("name", "")))
            if m:
                steps.append((int(m.group(1)), e.get("tid"), e["ts"], e["ts"] + e.get("dur", 0)))
        elif cat in ("cuda_runtime", "cuda_driver"):
            # the EG persistent MegaMoE kernel is launched through the driver API
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                runtime[corr] = (e["ts"], e.get("tid"))
        elif cat == "kernel":
            kernels.append(e)
    steps.sort(key=lambda s: s[2])

    def step_of(corr):
        r = runtime.get(corr)
        if r is None:
            return None
        ts, tid = r
        for sid, stid, s0, s1 in steps:
            if s0 <= ts <= s1 and (stid == tid or stid is None):
                return sid
        return None

    per_step: dict[int, dict] = {}
    unattributed = 0.0
    names: dict[str, list[float]] = {}
    for k in kernels:
        sid = step_of(k.get("args", {}).get("correlation"))
        dur = k.get("dur", 0) / 1e3  # us -> ms
        name = k["name"]
        if sid is None:
            unattributed += dur
            continue
        st = per_step.setdefault(sid, {"cats": {}, "intervals": [], "n_kernels": 0})
        c = categorize(name)
        st["cats"][c] = st["cats"].get(c, 0.0) + dur
        st["intervals"].append((k["ts"] / 1e3, (k["ts"] + k.get("dur", 0)) / 1e3))
        st["n_kernels"] += 1
        names.setdefault(f"{c}|{name}", []).append(dur)

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
