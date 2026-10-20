"""Merge per-rank expert-load dumps (MINISGL_DUMP_EXPERT_LOAD) into one per-layer file.

  python merge_expert_load.py --dumps 'RUN/ray_logs/expert_load_attention_*.json' \
      --out real_routing_<model>_perlayer.json --note "..."

Every attention DP rank dispatches into the same expert-parallel group, so the
per-rank counts are summed. Output schema ``fastafd.expert-load.v2``:
counts_per_layer[L][E], counts_total[E] and CV summaries (per layer and over
all layers).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def cv(v) -> float:
    m = sum(v) / len(v)
    return st.pstdev(v) / m if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="glob of expert_load_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    files = sorted(glob.glob(args.dumps))
    if not files:
        raise SystemExit(f"no dumps match {args.dumps}")
    per_layer = None
    steps = []
    for fn in files:
        d = json.load(open(fn))
        cl = d["counts_per_layer"]
        steps.append(int(d.get("steps_recorded", 0)))
        if per_layer is None:
            per_layer = [[0] * len(cl[0]) for _ in cl]
        for li, row in enumerate(cl):
            for e, c in enumerate(row):
                per_layer[li][e] += int(c)
    n_l, n_e = len(per_layer), len(per_layer[0])
    total = [sum(per_layer[li][e] for li in range(n_l)) for e in range(n_e)]
    active = [li for li in range(n_l) if sum(per_layer[li])]
    layer_cvs = [cv(per_layer[li]) for li in active]
    out = {
        "schema": "fastafd.expert-load.v2",
        "note": args.note or f"per-layer routing merged from {len(files)} per-rank dumps",
        "num_experts": n_e,
        "num_layers": n_l,
        "source_dumps": [os.path.basename(f) for f in files],
        "steps_recorded_per_dump": steps,
        "counts_per_layer": per_layer,
        "counts_total": total,
        "summary": {
            "assignments": sum(total),
            "aggregated_cv_percent": cv(total) * 100,
            "per_layer_cv_mean_percent": st.mean(layer_cvs) * 100 if layer_cvs else None,
            "per_layer_cv_median_percent": st.median(layer_cvs) * 100 if layer_cvs else None,
            "per_layer_max_over_mean_mean": st.mean(
                max(per_layer[li]) / (sum(per_layer[li]) / n_e) for li in active
            ) if active else None,
        },
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, args.out)
    print(json.dumps({"out": args.out, "dumps": len(files), **out["summary"]}))


if __name__ == "__main__":
    main()
