"""Collect AFD step-time rows into one table.

Reads afd_cases.sh job logs ("===== <tag> bs=..." headers followed by the
analyzer's JSON rows) and/or per-case step_times.json files.

  python collect_results.py JOB_LOG [JOB_LOG ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def rows_from_log(path: str) -> list[dict]:
    rows, tag = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"^===== (\S+) bs=", line)
        if m:
            tag = m.group(1)
            continue
        if line.startswith("{") and "step_ms_median" in line and tag:
            r = json.loads(line)
            r["case"] = tag
            rows.append(r)
    return rows


def rows_from_step_times(path: str) -> list[dict]:
    case = os.path.basename(os.path.dirname(path))
    return [dict(r, case=case) for r in json.load(open(path))["rows"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="job logs or step_times.json files / globs")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for pattern in a.inputs:
        for p in sorted(glob.glob(pattern)):
            rows += rows_from_step_times(p) if p.endswith(".json") else rows_from_log(p)
    seen, uniq = set(), []
    for r in rows:
        key = (re.sub(r"_bs[\d-]+$", "", r["case"]), r["per_attn_gpu_bs"])
        if key in seen or r.get("step_ms_median") is None:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda r: (re.sub(r"_bs[\d-]+$", "", r["case"]), r["per_attn_gpu_bs"]))
    print("| case | batch / attn GPU | decode step ms (median) | tok/s per attn GPU | samples |")
    print("|---|---:|---:|---:|---:|")
    for r in uniq:
        case = re.sub(r"_bs[\d-]+$", "", r["case"])
        print(f"| {case} | {r['per_attn_gpu_bs']} | {r['step_ms_median']:.1f} | "
              f"{r.get('tokens_per_s_per_attn_gpu', 0):.0f} | {r['samples']} |")
    if a.json:
        json.dump(uniq, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
