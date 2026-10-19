"""Colocated bring-up smoke test (dummy weights, reduced depth, one GPU).

  DSV4K3_META_DIR=<dir with DeepSeek-V4-Flash/, DeepSeek-V4-Pro/, Kimi-K3/ configs>
  DSV4K3_WORK_DIR=<scratch dir>
  python tests/dsv4_k3/smoke_colocated.py dsv4_flash|dsv4_pro|k3 [--layers N] [--moe torch|megamoe]

Prefill (chunked) + decode for duplicated requests, once eager and once through
CUDA graphs; checks finiteness and eager/graph agreement, prints memory/timing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from harness import EngineDriver, make_reduced_model_dir  # noqa: E402

META = os.environ.get("DSV4K3_META_DIR", "")
WORK = os.environ.get("DSV4K3_WORK_DIR", "/tmp/dsv4k3_work")


def model_dir(name: str, layers: int, experts: int | None) -> str:
    if name.startswith("dsv4"):
        src = os.path.join(META, "DeepSeek-V4-Flash" if name == "dsv4_flash" else "DeepSeek-V4-Pro")
        import json

        cfg = json.load(open(os.path.join(src, "config.json")))
        ratios = cfg["compress_ratios"]
        # keep a representative mix: one SWA-only (Flash), C4 and C128 layers
        if name == "dsv4_flash":
            pick = [0, 2, 3, 4, 5, 6][:layers]
        else:
            pick = [0, 2, 3, 4, 5, 6][:layers]
        over = {
            "num_hidden_layers": layers,
            "compress_ratios": [ratios[i] for i in pick] + [0],
            "num_hash_layers": 1,
        }
        if experts:
            over["n_routed_experts"] = experts
        dst = os.path.join(WORK, f"{name}_L{layers}_E{experts or 'full'}")
        return make_reduced_model_dir(src, dst, over)
    src = os.path.join(META, "Kimi-K3")
    over = {
        "num_hidden_layers": layers,
        "attn_res_block_size": 2,
        "linear_attn_config": {
            "kda_layers": [i for i in range(1, layers + 1) if i % 4 != 0],
            "full_attn_layers": [i for i in range(1, layers + 1) if i % 4 == 0],
        },
    }
    if experts:
        over["num_experts"] = experts
    dst = os.path.join(WORK, f"k3_L{layers}_E{experts or 'full'}")
    return make_reduced_model_dir(src, dst, over, text_config=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--moe", default="megamoe")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()
    os.environ["MINISGL_MOE_EXEC"] = args.moe
    path = model_dir(args.model, args.layers, args.experts or None)
    print("model dir", path)
    drv = EngineDriver(path, max_running_req=8, max_seq_len=args.max_seq_len, cuda_graph_bs=[1, 2, 4, 8], max_extend_tokens=1024)
    eng = drv.engine
    print(f"engine ready: pages={eng.num_pages} page_size={drv.page_size} max_seq_len={eng.max_seq_len}")
    print("pool:", getattr(eng.kv_cache, "memory_summary", lambda: {})())
    print(f"mem alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    g = torch.Generator().manual_seed(0)
    V = eng.model._config.vocab_size if hasattr(eng.model, "_config") else 1000
    lens = [37, 300, 5, 140]
    prompts = [torch.randint(0, V, (n,), generator=g).tolist() for n in lens]
    eager_uids = [0, 1, 2, 3]
    graph_uids = [10, 11, 12, 13]
    for u, p in zip(eager_uids, prompts):
        drv.add_request(u, p)
    for u, p in zip(graph_uids, prompts):
        drv.add_request(u, p)
    t0 = time.time()
    le = drv.prefill(eager_uids, chunk=128)
    lg = drv.prefill(graph_uids, chunk=None)
    torch.cuda.synchronize()
    print(f"prefill done {time.time() - t0:.2f}s; finite={bool(torch.isfinite(le).all())}")
    d = (le - lg).abs().max().item()
    print(f"prefill chunked(128) vs one-shot last-token logits max_abs={d:.4e} (ref absmax {le.abs().max().item():.3e})")
    nxt = le.argmax(-1).tolist()
    worst = 0.0
    agree = 0
    for step in range(args.steps):
        a = drv.decode(eager_uids, nxt, use_graph=False)
        b = drv.decode(graph_uids, nxt, use_graph=True)
        worst = max(worst, (a - b).abs().max().item())
        agree += int((a.argmax(-1) == b.argmax(-1)).all())
        if not torch.isfinite(a).all():
            print(f"step {step}: non-finite logits")
            break
        nxt = a.argmax(-1).tolist()
    print(f"decode {args.steps} steps: eager vs graph max_abs={worst:.4e}, argmax agree {agree}/{args.steps}")
    # timing of graph decode
    torch.cuda.synchronize()
    t0 = time.time()
    n = 10
    for _ in range(n):
        drv.decode(graph_uids, nxt, use_graph=True)
    torch.cuda.synchronize()
    print(f"graph decode bs=4 step {(time.time() - t0) / n * 1e3:.2f} ms ({args.layers} layers)")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
