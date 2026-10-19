"""Kernel-level profile of colocated decode steps (torch.profiler, CUDA activity).

  python tests/dsv4_k3/profile_decode.py dsv4_flash|k3 [--layers 4] [--bs 8] [--moe megamoe]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from harness import EngineDriver  # noqa: E402
from smoke_colocated import model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--moe", default="megamoe")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()
    os.environ["MINISGL_MOE_EXEC"] = args.moe
    path = model_dir(args.model, args.layers, args.experts or None)
    drv = EngineDriver(path, max_running_req=max(8, args.bs), max_seq_len=args.ctx + 64,
                       cuda_graph_bs=[args.bs], max_extend_tokens=4096, num_pages=None, memory_ratio=0.6)
    V = drv.engine.model._config.vocab_size
    g = torch.Generator().manual_seed(0)
    uids = list(range(args.bs))
    for u in uids:
        drv.add_request(u, torch.randint(0, V, (args.ctx,), generator=g).tolist())
    lg = drv.prefill(uids, chunk=2048)
    nxt = lg.argmax(-1).tolist()
    for _ in range(3):
        lg = drv.decode(uids, nxt, use_graph=args.graph)
        nxt = lg.argmax(-1).tolist()
    torch.cuda.synchronize()
    import time

    t0 = time.time()
    for _ in range(5):
        drv.decode(uids, nxt, use_graph=args.graph)
    torch.cuda.synchronize()
    print(f"decode step (graph={args.graph}) bs={args.bs} ctx={args.ctx}: {(time.time() - t0) / 5 * 1e3:.2f} ms")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            drv.decode(uids, nxt, use_graph=args.graph)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35, max_name_column_width=80))


if __name__ == "__main__":
    main()
