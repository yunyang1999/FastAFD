"""Decode step time per batch size from the AFD coordinator CPU trace.

Per-step latency = wall-clock delta between consecutive
``AFD_Coordinator_ScheduleDps`` starts in the decode phase (the coordinator runs
one step ahead, so in steady state this is the step time). Steps are attributed
to the bench_client batch run whose [start, end] window contains them; the first
``--skip`` and last ``--skip`` decode deltas of each run (pipeline ramp / drain)
are dropped.

  python analyze_step_times.py --trace RUN/ray_logs/*coordinator*_nvtx_cpu.json \
      --sweep RUN/sweep.json --out RUN/step_times.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics


def load_events(paths):
    events = []
    for p in paths:
        data = json.load(open(p))
        evs = data.get("events", data if isinstance(data, list) else [])
        events.extend(evs)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="coordinator *_nvtx_cpu.json (glob ok)")
    ap.add_argument("--ag-trace", default="", help="fallback: attention worker *_nvtx_cpu.json (glob ok)")
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sweep = json.load(open(args.sweep))
    paths = sorted(glob.glob(args.trace))
    source = "coordinator"
    if paths:
        events = [
            e for e in load_events(paths)
            if str(e.get("name", "")).startswith("AFD_Coordinator_ScheduleDps")
        ]
    else:
        # Fallback: one attention worker's per-step materialize events. Prefill
        # is scheduled first, so the last (max_tokens - 1) steps of each batch
        # run are its decode steps.
        ag = sorted(glob.glob(args.ag_trace)) if args.ag_trace else []
        if not ag:
            raise SystemExit(f"no trace matches {args.trace} (and no --ag-trace)")
        events = [e for e in load_events(ag[:1]) if str(e.get("name", "")).startswith("AFD_AG_Materialize")]
        source = "attention-worker"
    events.sort(key=lambda e: e["start_time_ns"])
    rows = []
    for run in sweep["runs"]:
        t0, t1 = run["start"] * 1e9, run["end"] * 1e9
        win = [e for e in events if t0 <= e["start_time_ns"] <= t1]
        if source == "coordinator":
            dec = [e for e in win if e.get("fields", {}).get("phase") == "decode"]
        else:
            dec = win[-max(1, int(sweep.get("max_tokens", 2)) - 1):]
        starts = [e["start_time_ns"] for e in dec]
        deltas = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
        core = deltas[args.skip : len(deltas) - args.skip] if len(deltas) > 2 * args.skip + 2 else deltas
        row = {
            "per_attn_gpu_bs": run["per_attn_gpu_bs"],
            "num_requests": run["num_requests"],
            "decode_steps": len(starts),
            "step_ms_median": statistics.median(core) if core else None,
            "step_ms_mean": statistics.fmean(core) if core else None,
            "step_ms_min": min(core) if core else None,
            "step_ms_max": max(core) if core else None,
            "samples": len(core),
            "source": source,
        }
        if row["step_ms_median"]:
            row["tokens_per_s_per_attn_gpu"] = run["per_attn_gpu_bs"] / (row["step_ms_median"] / 1e3)
        rows.append(row)
        print(json.dumps(row))
    json.dump({"prompt_len": sweep.get("prompt_len"), "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
