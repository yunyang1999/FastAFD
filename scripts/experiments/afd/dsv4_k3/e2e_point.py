"""Write one ``fastafd.e2e-step-time.v1`` JSON per measured batch size.

Sources (``--kind``):
  afd      an AFD sweep run dir (run_afd_hybrid_bench.sh): sweep.json, step_times.json,
           ray_logs/*coordinator*_nvtx_cpu.json (coordinator host time), worker logs (memory)
  colo     a FastAFD colocated run dir (colo_bench.sh): sweep.json + sched_trace.json
           (MINISGL_SCHED_STEP_TRACE) + server.log (memory)
  sglang   an sglang run dir (sgl_colocated_bench.sh): sgl_sweep.json (client ITL series)

  python e2e_point.py --kind afd --run-dir RUN --out-dir RESULTS --model deepseek_v4_flash \
      --weights real --isl 4096 --osl 1024 --a 28 --f 4 --mb 2 --attn-tp 1 --nodes 8 \
      --software fastafd --commit abc1234 --job-id 123 --kv-dtype fp8_e4m3 --moe-precision mxfp4

Schema (one file per point):
  schema, generated_at, system, model, weights, deployment{kind, software, label, a, f,
  microbatches, attn_tp, tp, ep, gpus, nodes}, workload{isl, osl, prompt_tokens,
  generated_tokens, context_at_measurement, batch_per_gpu, concurrent_requests, sampling,
  prompts}, step_ms{median, p50, p90, mean, min, max, n, skipped, source}, throughput{
  tokens_per_s_total, tokens_per_s_per_gpu, tokens_per_s_per_attention_gpu}, host{
  coordinator_busy_ms_median, coordinator_us_per_request} (AFD), memory{...}, software{...},
  run{job_id, run_dir, case, note}
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import statistics

SCHEMA = "fastafd.e2e-step-time.v1"


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))]


def stats(deltas: list[float], skip: int, source: str) -> dict:
    core = deltas[skip: len(deltas) - skip] if len(deltas) > 2 * skip + 2 else list(deltas)
    srt = sorted(core)
    return {
        "median": statistics.median(core) if core else None,
        "p50": pct(srt, 0.5),
        "p90": pct(srt, 0.9),
        "mean": statistics.fmean(core) if core else None,
        "min": srt[0] if srt else None,
        "max": srt[-1] if srt else None,
        "n": len(core),
        "skipped": skip,
        "source": source,
        "all_ms": [round(x, 4) for x in deltas],
    }


def grep_last(path: str, pattern: str) -> re.Match | None:
    last = None
    if not os.path.exists(path):
        return None
    for line in open(path, errors="replace"):
        m = re.search(pattern, line)
        if m:
            last = m
    return last


def afd_memory(run_dir: str) -> dict:
    mem = {}
    att = sorted(glob.glob(os.path.join(run_dir, "ray_logs", "attention_dp0_rank0.log"))) or sorted(
        glob.glob(os.path.join(run_dir, "ray_logs", "attention_rank0.log")))
    mlp = sorted(glob.glob(os.path.join(run_dir, "ray_logs", "mlp_dp0_rank0.log"))) or sorted(
        glob.glob(os.path.join(run_dir, "ray_logs", "mlp_rank0.log")))
    if att:
        m = grep_last(att[0], r"afd_hybrid_ag_model loaded cuda_alloc_gib=([\d.]+)(?: reserved_gib=([\d.]+) free_gib=([\d.]+))?")
        if m:
            mem["attention_weights_alloc_gib"] = float(m.group(1))
        m = grep_last(att[0], r"warmup:done .*free_gib=[\d.]+->([\d.]+) reserved_gib=([\d.]+)")
        if m:
            mem["attention_free_gib_after_capture"] = float(m.group(1))
            mem["attention_reserved_gib_after_capture"] = float(m.group(2))
    if mlp:
        m = grep_last(mlp[0], r"afd_runtime:init_done .*cuda_alloc_gib=([\d.]+) reserved_gib=([\d.]+) free_gib=([\d.]+)")
        if m:
            mem["expert_alloc_gib"] = float(m.group(1))
            mem["expert_reserved_gib"] = float(m.group(2))
            mem["expert_free_gib"] = float(m.group(3))
    log = os.path.join(run_dir, "afd.log")
    m = grep_last(log, r"hybrid cache: (\d+) B/token, fixed ([\d.]+) GiB, attention weights ([\d.]+) GiB")
    if m:
        mem["kv_bytes_per_token"] = int(m.group(1))
        mem["attention_fixed_state_gib"] = float(m.group(2))
        mem["attention_weights_est_gib"] = float(m.group(3))
    m = grep_last(log, r"allocating (\d+) pages \(([\d.]+) GiB")
    if m:
        mem["attention_kv_pages"] = int(m.group(1))
        mem["attention_kv_gib"] = float(m.group(2))
    return mem


def coordinator_host_time(run_dir: str, runs: list[dict], skip: int) -> dict[int, dict]:
    """Per batch-size run: coordinator host busy time per decode step (ms), i.e. the
    sum of the coordinator's own ranges of that step minus the time spent waiting for
    worker replies inside CompleteCollect."""
    paths = sorted(glob.glob(os.path.join(run_dir, "ray_logs", "*coordinator*_nvtx_cpu.json")))
    if not paths:
        return {}
    evs = []
    for p in paths:
        d = json.load(open(p))
        evs.extend(d.get("events", d if isinstance(d, list) else []))
    by_id = {e["id"]: e for e in evs}
    busy: dict[int, float] = {}
    start_ns: dict[int, int] = {}
    phase: dict[int, str] = {}
    for e in evs:
        sid = e.get("step_id")
        if sid is None:
            continue
        dur = e["duration_ns"] / 1e6
        if e["depth"] == 0:
            busy[sid] = busy.get(sid, 0.0) + dur
            if e["name"].startswith("AFD_Coordinator_ScheduleDps"):
                start_ns[sid] = e["start_time_ns"]
                phase[sid] = e.get("fields", {}).get("phase", "")
        elif e["depth"] == 1 and e["name"].startswith("AFD_Coordinator_RecvWorkerReply"):
            parent = by_id.get(e.get("parent_id"))
            if parent is not None and parent["name"].startswith("AFD_Coordinator_CompleteCollect"):
                busy[sid] = busy.get(sid, 0.0) - dur
    out = {}
    for run in runs:
        t0, t1 = run["start"] * 1e9, run["end"] * 1e9
        sids = sorted(s for s, t in start_ns.items() if t0 <= t <= t1 and phase.get(s) == "decode")
        core = sids[skip: len(sids) - skip] if len(sids) > 2 * skip + 2 else sids
        vals = [busy[s] for s in core if s in busy]
        if vals:
            out[run["per_attn_gpu_bs"]] = {
                "coordinator_busy_ms_median": statistics.median(vals),
                "coordinator_busy_ms_mean": statistics.fmean(vals),
                "coordinator_us_per_request": statistics.median(vals) * 1e3 / run["num_requests"],
                "n": len(vals),
            }
    return out


def load_afd(run_dir: str, skip: int) -> tuple[list[dict], dict, dict]:
    sweep = json.load(open(os.path.join(run_dir, "sweep.json")))
    st_path = os.path.join(run_dir, "step_times.json")
    rows = json.load(open(st_path))["rows"] if os.path.exists(st_path) else []
    points = []
    host = coordinator_host_time(run_dir, sweep["runs"], skip)
    for run in sweep["runs"]:
        row = next((r for r in rows if r["per_attn_gpu_bs"] == run["per_attn_gpu_bs"]), None)
        if row is None or row.get("step_ms_median") is None:
            continue
        deltas = row.get("step_ms_all")
        s = stats(deltas, skip, row.get("source", "coordinator")) if deltas else {
            "median": row["step_ms_median"], "p50": row.get("step_ms_p50"), "p90": row.get("step_ms_p90"),
            "mean": row.get("step_ms_mean"), "min": row.get("step_ms_min"), "max": row.get("step_ms_max"),
            "n": row["samples"], "skipped": skip, "source": row.get("source", "coordinator")}
        observed = row.get("reqs_per_step")
        if observed and observed != run["num_requests"]:
            print(f"warning b={run['per_attn_gpu_bs']}: steady decode steps ran {observed} requests, "
                  f"{run['num_requests']} requested", flush=True)
        points.append({"bs": run["per_attn_gpu_bs"], "num_requests": run["num_requests"],
                       "observed_requests": observed,
                       "generated_tokens": run.get("generated_tokens"), "step_ms": s,
                       "host": host.get(run["per_attn_gpu_bs"])})
    return points, sweep, afd_memory(run_dir)


def load_colo(run_dir: str, skip: int) -> tuple[list[dict], dict, dict]:
    sweep = json.load(open(os.path.join(run_dir, "sweep.json")))
    tr = json.load(open(os.path.join(run_dir, "sched_trace.json")))
    steps = tr["steps"]  # [start_time_ns, phase, num_reqs, num_tokens]
    points = []
    for run in sweep["runs"]:
        t0, t1 = run["start"] * 1e9, run["end"] * 1e9
        dec = [s for s in steps if t0 <= s[0] <= t1 and s[1] == "decode" and s[2] == run["num_requests"]]
        starts = [s[0] for s in dec]
        deltas = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
        if not deltas:
            continue
        points.append({"bs": run["per_attn_gpu_bs"], "num_requests": run["num_requests"],
                       "generated_tokens": run.get("generated_tokens"),
                       "step_ms": stats(deltas, skip, "scheduler trace"), "host": None})
    mem = {}
    log = os.path.join(run_dir, "server.log")
    m = grep_last(log, r"Allocating (\d+) tokens for KV cache, K \+ V = ([\d.]+) GiB")
    if m:
        mem["kv_tokens"] = int(m.group(1))
        mem["kv_gib"] = float(m.group(2))
    mem.update(sweep.get("memory", {}))
    return points, sweep, mem


def load_sglang(run_dir: str, skip: int) -> tuple[list[dict], dict, dict]:
    sweep = json.load(open(os.path.join(run_dir, "sgl_sweep.json")))
    points = []
    for run in sweep["runs"]:
        deltas = run.get("step_ms_all") or []
        if not deltas:
            continue
        s = stats(deltas, skip, "client inter-token latency (median over requests per token)")
        s["server_log_median_ms"] = run.get("server_log_median_ms", run.get("server_log_mean_ms"))
        if run.get("server_queue_max", 0) > 0 or (run.get("server_running_max") and run["server_running_max"] < run["num_requests"]):
            print(f"skip b={run['per_gpu_bs']}: server queued requests (max running {run.get('server_running_max')}, "
                  f"queue {run.get('server_queue_max')}) -- batch did not fit", flush=True)
            continue
        points.append({"bs": run["per_gpu_bs"], "num_requests": run["num_requests"],
                       "generated_tokens": run.get("generated_tokens"), "step_ms": s, "host": None,
                       "memory": run.get("memory")})
    return points, sweep, sweep.get("memory", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["afd", "colo", "sglang"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True, help="deepseek_v4_flash | kimi_k3 | qwen3_235b")
    ap.add_argument("--weights", choices=["real", "dummy"], required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--software", choices=["fastafd", "sglang"], required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--version", default="")
    ap.add_argument("--a", type=int, default=0, help="attention GPUs (AFD)")
    ap.add_argument("--f", type=int, default=0, help="expert GPUs (AFD)")
    ap.add_argument("--mb", type=int, default=0)
    ap.add_argument("--attn-tp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=0, help="colocated TP")
    ap.add_argument("--ep", type=int, default=0, help="colocated EP (MoE)")
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--kv-dtype", default="")
    ap.add_argument("--moe-precision", default="")
    ap.add_argument("--attention-kernels", default="")
    ap.add_argument("--job-id", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--prompts", default="", help="description of the prompt source")
    ap.add_argument("--extra", default="{}", help="JSON merged into software{}")
    args = ap.parse_args()

    loader = {"afd": load_afd, "colo": load_colo, "sglang": load_sglang}[args.kind]
    points, sweep, mem = loader(args.run_dir, args.skip)
    if not points:
        raise SystemExit(f"{args.run_dir}: no measured points")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.kind == "afd":
        gpus = args.a + args.f
        label = f"{args.a}A{args.f}F m{args.mb}" + (f" attn-TP{args.attn_tp}" if args.attn_tp > 1 else "")
        dep_key = f"{args.a}a{args.f}f_mb{args.mb}" + (f"_tp{args.attn_tp}" if args.attn_tp > 1 else "")
        deployment = {"kind": "afd", "software": args.software, "label": label, "a": args.a, "f": args.f,
                      "microbatches": args.mb, "attn_tp": args.attn_tp, "gpus": gpus, "nodes": args.nodes}
    else:
        gpus = args.tp
        ep = args.ep or args.tp
        label = f"colocated TP{args.tp}" + (f" EP{ep}" if ep != args.tp or args.ep else "")
        dep_key = f"colo_tp{args.tp}" + (f"_ep{ep}" if args.ep else "")
        deployment = {"kind": "colocated", "software": args.software, "label": label, "tp": args.tp, "ep": ep,
                      "gpus": gpus, "nodes": args.nodes or 1}
    temp = sweep.get("temperature", 1.0)
    try:
        sampling = "greedy" if float(temp) == 0 else f"temperature {temp}"
    except (TypeError, ValueError):
        sampling = str(temp)
    prompts = args.prompts or ("real text (prompt file)" if sweep.get("prompt_file") else "synthetic repeated token")
    written = []
    for p in points:
        bs = p["bs"]
        med = p["step_ms"]["median"]
        tok_s = p["num_requests"] / (med / 1e3) if med else None
        thr = {"tokens_per_s_total": tok_s, "tokens_per_s_per_gpu": tok_s / gpus if tok_s else None}
        if args.kind == "afd":
            thr["tokens_per_s_per_attention_gpu"] = tok_s / args.a if tok_s else None
        rec = {
            "schema": SCHEMA,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "system": "GB200 NVL72 (lyris), 4 GPUs per node",
            "model": args.model,
            "weights": args.weights,
            "deployment": deployment,
            "workload": {
                "isl": args.isl, "osl": args.osl,
                "prompt_tokens": sweep.get("prompt_len"),
                "generated_tokens": sweep.get("max_tokens"),
                "context_at_measurement": (sweep.get("prompt_len") or 0) + (sweep.get("max_tokens") or 0) // 2,
                "batch_per_gpu": bs,
                "concurrent_requests": p["num_requests"],
                "concurrent_requests_observed": p.get("observed_requests"),
                "sampling": sampling,
                "prompts": prompts,
            },
            "step_ms": p["step_ms"],
            "throughput": thr,
            "host": p.get("host"),
            "memory": dict(mem, **(p.get("memory") or {})),
            "software": dict({
                "stack": args.software, "commit": args.commit, "version": args.version,
                "kv_cache_dtype": args.kv_dtype, "moe_precision": args.moe_precision,
                "attention_kernels": args.attention_kernels,
            }, **json.loads(args.extra)),
            "run": {"job_id": args.job_id, "run_dir": os.path.basename(os.path.normpath(args.run_dir)),
                    "case": args.case, "note": args.note},
        }
        name = f"{args.model}_{args.software}_{dep_key}_isl{args.isl}_osl{args.osl}_b{bs}.json"
        path = os.path.join(args.out_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, path)
        written.append(name)
        print(f"{name}: step {med:.2f} ms (p90 {p['step_ms']['p90']:.2f}, n={p['step_ms']['n']}) "
              f"tok/s/GPU {thr['tokens_per_s_per_gpu']:.0f}", flush=True)
    print(f"wrote {len(written)} points to {args.out_dir}")


if __name__ == "__main__":
    main()
