"""Tables for the attention-side kernel work: per-layer A-side time before/after
(tests/dsv4_k3/bench_aside.py JSONs) and attention-core time vs the sglang
0.5.18 kernel benchmark (afd.decode-attention-benchmark.v2 JSONs).

  python compare_aside.py --before DIR --after DIR --after-core DIR --sglang DIR --out table.md

Per-layer numbers are us per layer for each attention layer type and the
model average weighted by the full model's layer mix.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

MODEL_KEY = {"dsv4_flash": "deepseek_v4_flash", "k3": "kimi_k3"}


def load_bench(d: str, core: bool) -> dict:
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue  # other JSONs in the run directory (numerics etc.)
        for r in rows:
            if not isinstance(r, dict) or "model" not in r:
                continue
            if bool(r.get("mode") == "core") != core:
                continue
            out[(r["model"], r["tp"], r["bs"], r["ctx"])] = r
    return out


def load_sglang(d: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            x = json.load(open(f))
        except Exception:
            continue
        w, t = x.get("workload", {}), x.get("topology", {})
        if w.get("q_len_per_req") != 1 or not x.get("stable", True):
            continue
        key = (x.get("model_profile", {}).get("key"), x.get("layer_type"), t.get("attn_tp_size"), w.get("batch_size"))
        out[key + (w.get("seqlen"),)] = x
    return out


def sgl_core(sg: dict, model: str, lt: str, tp: int, b: int, ctx: int) -> float | None:
    x = sg.get((MODEL_KEY[model], lt, tp, b, ctx)) or sg.get((MODEL_KEY[model], lt, tp, b, None))
    if x is None:
        return None
    comps = x.get("components") or {}
    if model == "k3" and lt == "mla":
        # the kernel sglang would run here: trtllm-gen if measured, else the CuTe-DSL MLA
        for k in ("mla_trtllm_gen", "mla_cute_dsl"):
            if k in comps:
                return comps[k]["p50_ms"] * 1e3
    return x["stage_cuda"]["p50_ms"] * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--after-core", required=True)
    ap.add_argument("--sglang", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    before, after = load_bench(args.before, False), load_bench(args.after, False)
    core = load_bench(args.after_core, True)
    sg = load_sglang(args.sglang)
    lines = ["# Attention-side per-layer decode time (us / layer)", ""]
    lines += ["## Full A-side per layer (attention + mHC / residuals + norms + router + A-side MoE parts)", ""]
    lines += ["| model | TP | ctx | B | before avg | after avg | speed-up | after per type |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key in sorted(after):
        m, tp, b, ctx = key
        a = after[key]
        bf = before.get(key)
        ba = bf["model_avg"]["aside_us"] if bf else None
        aa = a["model_avg"]["aside_us"]
        per = ", ".join(f"{t} {v['aside_us']:.0f}" for t, v in a["per_type"].items())
        lines.append(f"| {m} | {tp} | {ctx} | {b} | {ba:.0f} | {aa:.0f} | {ba / aa:.2f}x | {per} |" if ba else
                     f"| {m} | {tp} | {ctx} | {b} | - | {aa:.0f} | - | {per} |")
    lines += ["", "## Attention core vs sglang 0.5.18 (same kernels / boundary as the sglang kernel benchmark)", ""]
    lines += ["| model | TP | ctx | B | type | FastAFD core | sglang core | ratio |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    mix_avg = defaultdict(lambda: [0.0, 0.0, 0])
    for key in sorted(core):
        m, tp, b, ctx = key
        r = core[key]
        mix = after.get(key, {}).get("layer_mix") or ({"swa": 2, "c4": 21, "c128": 20} if m == "dsv4_flash" else {"kda": 69, "mla": 24})
        ok = True
        for lt, v in r["per_type"].items():
            s = sgl_core(sg, m, lt, tp, b, ctx)
            c = v["core_us"]
            ratio = f"{c / s:.2f}" if s else "-"
            lines.append(f"| {m} | {tp} | {ctx} | {b} | {lt} | {c:.1f} | {s:.1f} | {ratio} |" if s else
                         f"| {m} | {tp} | {ctx} | {b} | {lt} | {c:.1f} | - | - |")
            if s is None:
                ok = False
            else:
                mix_avg[key][0] += c * mix[lt]
                mix_avg[key][1] += s * mix[lt]
        mix_avg[key][2] = ok
    lines += ["", "Model-weighted attention core (us / layer):", ""]
    lines += ["| model | TP | ctx | B | FastAFD | sglang | ratio |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, (c, s, ok) in sorted(mix_avg.items()):
        if ok and s:
            m, tp, b, ctx = key
            n = sum((after.get(key, {}).get("layer_mix") or {}).values()) or (43 if m == "dsv4_flash" else 93)
            lines.append(f"| {m} | {tp} | {ctx} | {b} | {c / n:.1f} | {s / n:.1f} | {c / s:.2f} |")
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, args.out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
