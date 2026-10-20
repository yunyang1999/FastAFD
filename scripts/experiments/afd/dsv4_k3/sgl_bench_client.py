"""Decode step time of an sglang server from client-side inter-token latency.

For each per-GPU batch size ``b`` it sends ``b * --gpus`` streaming ``/generate``
requests at once (distinct real-text prompts of exactly ``--prompt-len`` tokens
from ``--prompt-file``, ``ignore_eos``, ``--max-tokens`` new tokens) and records
the arrival time of every output token. All requests decode in one batch, so the
step time series is the delta between the median (over requests) arrival times of
consecutive tokens; the first/last ``--skip`` steps are dropped for the summary.
The server's decode log lines in the same window (``--server-log``, "Decode batch
... gen throughput") give an independent mean step time.

  python sgl_bench_client.py --url http://127.0.0.1:PORT --tokenizer DIR --gpus 4 \
      --batch-sizes 8,16,32 --prompt-len 4576 --max-tokens 64 --prompt-file prompts.txt --out sgl_sweep.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request

from bench_client import real_prompt_ids


def stream_group(url: str, ids: list[list[int]], max_tokens: int, temperature: float, timeout: float,
                 arrivals: dict, base_index: int, errors: list) -> None:
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_tokens, "ignore_eos": True, "temperature": temperature,
                            "stream_interval": 1},
        "stream": True,
    }
    req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                d = json.loads(body)
                if "error" in d:
                    errors.append(d["error"])
                    return
                idx = base_index + int(d.get("index", 0))
                k = int(d["meta_info"]["completion_tokens"])
                arrivals.setdefault(idx, {})[k] = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        errors.append(repr(e))


def gpu_mem_sampler(stop: threading.Event, out: dict) -> None:
    peak = 0.0
    while not stop.is_set():
        try:
            txt = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            vals = [float(x) for x in txt.split() if x.strip()]
            if vals:
                peak = max(peak, max(vals) / 1024.0)
        except Exception:  # noqa: BLE001
            pass
        stop.wait(3.0)
    out["gpu_used_gib_max"] = peak


def server_log_steps(path: str, t0: float, t1: float, num_requests: int) -> tuple[float | None, int, int]:
    """From the server's "Decode batch ... #running-req: N ... gen throughput (token/s): X ...
    #queue-req: Q" lines with a timestamp in [t0, t1] (wall clock): the median decode step
    (ms) = N / X over lines with N == the run's request count (first interval after prefill
    excluded), the largest N seen and the largest queue depth (Q > 0 means the batch did not
    fit: requests waited and the running batch was smaller than requested)."""
    if not path:
        return None, 0, 0
    vals, max_run, max_q = [], 0, 0
    try:
        for line in open(path, errors="replace"):
            if "Decode batch" not in line or "gen throughput" not in line:
                continue
            m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
            n = re.search(r"#running-req: (\d+)", line)
            g = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
            q = re.search(r"#queue-req: (\d+)", line)
            if not (m and n and g):
                continue
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            if not (t0 - 1 <= ts <= t1 + 1):
                continue
            max_run = max(max_run, int(n.group(1)))
            max_q = max(max_q, int(q.group(1)) if q else 0)
            if int(n.group(1)) == num_requests and float(g.group(1)) > 0:
                vals.append(int(n.group(1)) / float(g.group(1)) * 1e3)
    except OSError:
        return None, 0, 0
    vals = vals[1:] if len(vals) > 2 else vals
    return (statistics.median(vals) if vals else None), max_run, max_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--batch-sizes", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=32, help="requests per streaming connection")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bss = [int(x) for x in args.batch_sizes.split(",") if x]
    n_max = max(bss) * args.gpus
    prompts = real_prompt_ids(args.tokenizer, args.prompt_file, n_max, args.prompt_len)
    log = {"prompt_len": args.prompt_len, "max_tokens": args.max_tokens, "gpus": args.gpus,
           "prompt_file": args.prompt_file, "temperature": args.temperature, "runs": []}
    for bs in bss:
        n = bs * args.gpus
        arrivals: dict = {}
        errors: list = []
        stop = threading.Event()
        mem: dict = {}
        sampler = threading.Thread(target=gpu_mem_sampler, args=(stop, mem), daemon=True)
        sampler.start()
        threads = []
        t0 = time.time()
        p0 = time.perf_counter()
        for g0 in range(0, n, args.group):
            ids = prompts[g0: min(n, g0 + args.group)]
            th = threading.Thread(target=stream_group, args=(args.url, ids, args.max_tokens, args.temperature,
                                                                args.timeout, arrivals, g0, errors), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        t1 = time.time()
        stop.set()
        sampler.join(timeout=15)
        if errors:
            print(f"bs={bs}: {len(errors)} errors, first: {errors[0]}", flush=True)
        # token k arrival = median over requests; step k = arrival(k) - arrival(k-1)
        ks = range(2, args.max_tokens + 1)
        med = []
        for k in ks:
            vals = [a[k] - p0 for a in arrivals.values() if k in a]
            if len(vals) >= max(1, n // 2):
                med.append(statistics.median(vals))
        deltas = [(b - a) * 1e3 for a, b in zip(med, med[1:])]
        core = deltas[args.skip: len(deltas) - args.skip] if len(deltas) > 2 * args.skip + 2 else deltas
        ntok = sum(max(a) for a in arrivals.values()) if arrivals else 0
        srv_ms, srv_run, srv_q = server_log_steps(args.server_log, t0, t1, n)
        run = {"per_gpu_bs": bs, "num_requests": n, "completed_requests": len(arrivals), "start": t0, "end": t1,
               "generated_tokens": ntok, "step_ms_all": [round(x, 4) for x in deltas],
               "step_ms_median": statistics.median(core) if core else None,
               "server_log_median_ms": srv_ms, "server_running_max": srv_run, "server_queue_max": srv_q,
               "memory": mem, "errors": errors[:3]}
        log["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "step_ms_all"}), flush=True)
        with open(args.out, "w") as f:
            json.dump(log, f, indent=1)
        time.sleep(3.0)


if __name__ == "__main__":
    main()
