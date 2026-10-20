"""Summarise the end-to-end campaign points (fastafd.e2e-step-time.v1 JSONs).

  python e2e_summary.py RESULTS_DIR [--csv summary.csv] [--md step_time_tables.md]

Writes a flat CSV (one row per point), Markdown step-time tables (per model and
deployment: batch per GPU vs step time at each ISL), the coordinator host-time check
(eta = coordinator busy time per request per step, from the AFD points) and the
profiler decompositions found in ``profile_*.json``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

COLS = ["model", "weights", "software", "deployment", "a", "f", "microbatches", "attn_tp", "tp", "ep", "gpus", "nodes",
        "isl", "osl", "context_at_measurement", "batch_per_gpu", "concurrent_requests", "step_ms_median", "step_ms_p50",
        "step_ms_p90", "step_ms_mean", "step_ms_n", "tokens_per_s_per_gpu", "tokens_per_s_per_attention_gpu",
        "coordinator_busy_ms", "coordinator_us_per_request", "attention_gpu_mem_gib", "expert_gpu_mem_gib",
        "gpu_used_gib_max", "kv_cache_dtype", "moe_precision", "commit", "job_id", "file"]


def load(results_dir: str) -> list[dict]:
    pts = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("schema") != "fastafd.e2e-step-time.v1":
            continue
        d["_file"] = os.path.basename(f)
        pts.append(d)
    return pts


def row(d: dict) -> dict:
    dep, w, s, t, mem = d["deployment"], d["workload"], d["step_ms"], d["throughput"], d.get("memory") or {}
    host = d.get("host") or {}
    att_mem = mem.get("attention_reserved_gib_after_capture") or mem.get("gpu_used_gib_max")
    return {
        "model": d["model"], "weights": d["weights"], "software": dep["software"], "deployment": dep["label"],
        "a": dep.get("a", ""), "f": dep.get("f", ""), "microbatches": dep.get("microbatches", ""),
        "attn_tp": dep.get("attn_tp", ""), "tp": dep.get("tp", ""), "ep": dep.get("ep", ""), "gpus": dep["gpus"],
        "nodes": dep.get("nodes", ""), "isl": w["isl"], "osl": w["osl"],
        "context_at_measurement": w.get("context_at_measurement", ""), "batch_per_gpu": w["batch_per_gpu"],
        "concurrent_requests": w["concurrent_requests"],
        "step_ms_median": round(s["median"], 3), "step_ms_p50": round(s["p50"], 3) if s.get("p50") else "",
        "step_ms_p90": round(s["p90"], 3) if s.get("p90") else "", "step_ms_mean": round(s["mean"], 3) if s.get("mean") else "",
        "step_ms_n": s["n"], "tokens_per_s_per_gpu": round(t["tokens_per_s_per_gpu"], 1),
        "tokens_per_s_per_attention_gpu": round(t["tokens_per_s_per_attention_gpu"], 1) if t.get("tokens_per_s_per_attention_gpu") else "",
        "coordinator_busy_ms": round(host["coordinator_busy_ms_median"], 3) if host else "",
        "coordinator_us_per_request": round(host["coordinator_us_per_request"], 3) if host else "",
        "attention_gpu_mem_gib": round(att_mem, 1) if att_mem else "",
        "expert_gpu_mem_gib": round(mem["expert_reserved_gib"], 1) if mem.get("expert_reserved_gib") else "",
        "gpu_used_gib_max": round(mem["gpu_used_gib_max"], 1) if mem.get("gpu_used_gib_max") else "",
        "kv_cache_dtype": d["software"].get("kv_cache_dtype", ""), "moe_precision": d["software"].get("moe_precision", ""),
        "commit": d["software"].get("commit", "") or d["software"].get("version", ""), "job_id": d["run"].get("job_id", ""),
        "file": d["_file"],
    }


def tables(pts: list[dict]) -> str:
    out = ["# End-to-end decode step time (ms) vs batch per GPU", ""]
    groups: dict[tuple, dict] = {}
    for d in pts:
        dep, w = d["deployment"], d["workload"]
        key = (d["model"], d["weights"], dep["software"], dep["label"], dep["gpus"], dep.get("nodes", ""))
        groups.setdefault(key, {}).setdefault((w["isl"], w["osl"]), {})[w["batch_per_gpu"]] = d
    for key in sorted(groups, key=lambda k: (k[0], k[2], k[1], -k[4] if isinstance(k[4], int) else 0, k[3])):
        model, weights, software, label, gpus, nodes = key
        isls = sorted(groups[key])
        bss = sorted({b for v in groups[key].values() for b in v})
        out.append(f"## {model} ({weights} weights), {software}, {label}, {gpus} GPUs" + (f", {nodes} nodes" if nodes else ""))
        out.append("")
        head = "| batch / GPU | requests |" + "".join(f" {i}K/{o // 1024}K step ms (p90) | tok/s/GPU |" for i, o in
                                                     ((isl // 1024, osl) for isl, osl in isls))
        out.append(head)
        out.append("|---:|---:|" + "---:|---:|" * len(isls))
        for b in bss:
            cells = []
            nreq = ""
            for isl in isls:
                d = groups[key][isl].get(b)
                if d is None:
                    cells.append(" - | - |")
                    continue
                nreq = d["workload"]["concurrent_requests"]
                s = d["step_ms"]
                p90 = f" ({s['p90']:.1f})" if s.get("p90") else ""
                cells.append(f" {s['median']:.1f}{p90} | {d['throughput']['tokens_per_s_per_gpu']:.0f} |")
            out.append(f"| {b} | {nreq} |" + "".join(cells))
        out.append("")
    # eta check
    afd = [d for d in pts if d["deployment"]["kind"] == "afd" and d.get("host")]
    if afd:
        out += ["## Coordinator host time (AFD points)", "",
                "eta = coordinator busy time per decode step / requests in the step (A x b).", "",
                "| model | deployment | ISL | batch / GPU | requests | step ms | coordinator busy ms | eta us/request |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
        etas = []
        for d in sorted(afd, key=lambda d: (d["model"], d["deployment"]["label"], d["workload"]["isl"], d["workload"]["batch_per_gpu"])):
            h, w = d["host"], d["workload"]
            etas.append(h["coordinator_us_per_request"])
            out.append(f"| {d['model']} | {d['deployment']['label']} | {w['isl']} | {w['batch_per_gpu']} | {w['concurrent_requests']} | "
                       f"{d['step_ms']['median']:.2f} | {h['coordinator_busy_ms_median']:.2f} | {h['coordinator_us_per_request']:.2f} |")
        xs = [d["workload"]["concurrent_requests"] for d in afd]
        ys = [d["host"]["coordinator_busy_ms_median"] for d in afd]
        if len(xs) >= 3 and len(set(xs)) > 1:
            n = len(xs)
            sx, sy, sxx, sxy = sum(xs), sum(ys), sum(x * x for x in xs), sum(x * y for x, y in zip(xs, ys))
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            icpt = (sy - slope * sx) / n
            out += ["", f"Linear fit over all AFD points: busy = {icpt:.2f} ms + {slope * 1e3:.2f} us x requests "
                        f"(median of per-point eta {statistics.median(etas):.2f} us; Qwen3-235B anchor 3.87 us/request/step)."]
        out.append("")
    return "\n".join(out)


def profiles(results_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(results_dir, "profile_*.json")))
    if not files:
        return ""
    out = ["## Profiler decomposition (torch.profiler, dp0 worker of each role; ms per decode step, medians over the profiled steps)", "",
           "| file | role | steps | step ms (unprofiled) | GPU span | busy | idle in span | attn | dense | norm | gate | quant/comm | moe | sample | copy | other |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in files:
        d = json.load(open(f))
        m = d.get("median", {})
        g = lambda k: f"{m.get(k, 0.0):.2f}" if m.get(k) is not None else "-"  # noqa: E731
        out.append(f"| {os.path.basename(f)} | {d['role']} | {len(d.get('steps_used', []))} | "
                   f"{d.get('step_ms_unprofiled') or '-'} | {g('span_ms')} | {g('busy_ms')} | {g('idle_ms')} | {g('attn_ms')} | "
                   f"{g('dense_ms')} | {g('norm_ms')} | {g('gate_ms')} | {g('quant_ms')} | {g('moe_ms')} | {g('sample_ms')} | "
                   f"{g('copy_ms')} | {g('other_ms')} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--md", default="step_time_tables.md")
    a = ap.parse_args()
    pts = load(a.results_dir)
    rows = sorted((row(d) for d in pts), key=lambda r: (r["model"], r["software"], r["deployment"], r["isl"], r["batch_per_gpu"]))
    with open(os.path.join(a.results_dir, a.csv), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    md = tables(pts) + "\n" + profiles(a.results_dir)
    with open(os.path.join(a.results_dir, a.md), "w") as f:
        f.write(md)
    print(md)
    print(f"{len(rows)} points -> {a.csv}, {a.md}")


if __name__ == "__main__":
    main()
