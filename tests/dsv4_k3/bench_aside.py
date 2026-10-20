"""Per-layer attention-side (AFD A side) decode timing on fixed shapes.

  python tests/dsv4_k3/bench_aside.py dsv4_flash|dsv4_pro|k3 --ctx 4096 \
      [--bs 8,32,128] [--tp 2] [--copies 3] [--out result.json]

A reduced model holds ``--copies`` layers of every attention type (DSV4: SWA,
C4, C128; K3: KDA, MLA) with dummy weights; distinct copies keep the weights
out of L2 between layers as in the full model. For every batch size a decode
batch of B requests at context C is faked (pages mapped, cache rows zero, no
prefill) and the following are timed by CUDA-graph replay, per layer:

  attn   the attention module (input projections .. output projection)
  aside  everything the attention stage runs per layer in an AFD step
         DSV4: hc_post of the previous MoE output + attention_part (hc_pre,
               norm, attention, hc_post, hc_pre, norm) + router
         K3:   residual add + attention_part (attention residuals, norm,
               attention, norm) + router + latent down projection + shared
               experts + post_combine (latent norm + up projection)

``model_avg`` weights the per-type times by the layer mix of the full model.
TP > 1 runs one process per rank (attention heads sharded, o_proj all-reduce).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from harness import make_reduced_model_dir  # noqa: E402

META = os.environ.get("DSV4K3_META_DIR", "")
WORK = os.environ.get("DSV4K3_WORK_DIR", "/tmp/dsv4k3_work")


def _src(name: str) -> str:
    return os.path.join(META, {"dsv4_flash": "DeepSeek-V4-Flash", "dsv4_pro": "DeepSeek-V4-Pro", "k3": "Kimi-K3"}[name])


def full_layer_mix(name: str) -> dict[str, int]:
    cfg = json.load(open(os.path.join(_src(name), "config.json")))
    if name.startswith("dsv4"):
        L = int(cfg["num_hidden_layers"])
        r = cfg["compress_ratios"][:L]
        return {"swa": r.count(0), "c4": r.count(4), "c128": r.count(128)}
    t = cfg.get("text_config", cfg)
    L = int(t["num_hidden_layers"])
    kda = len([x for x in t["linear_attn_config"]["kda_layers"] if x <= L])
    return {"kda": kda, "mla": L - kda}


def bench_model_dir(name: str, copies: int) -> tuple[str, dict[str, list[int]]]:
    """Reduced config with ``copies`` layers per attention type -> (dir, type -> layer ids)."""
    if name.startswith("dsv4"):
        types = {"swa": 0, "c4": 4, "c128": 128}
        ratios, ids = [], {}
        for t, r in types.items():
            ids[t] = list(range(len(ratios), len(ratios) + copies))
            ratios += [r] * copies
        over = {
            "num_hidden_layers": len(ratios),
            "compress_ratios": ratios + [0],
            "num_hash_layers": 0,
            "n_routed_experts": 16,
        }
        dst = os.path.join(WORK, f"bench_{name}_x{copies}")
        return make_reduced_model_dir(_src(name), dst, over), ids
    # K3: dense layer 0, then `copies` KDA and `copies` MLA layers
    kda = list(range(1, 1 + copies))
    mla = list(range(1 + copies, 1 + 2 * copies))
    L = 1 + 2 * copies
    over = {
        "num_hidden_layers": L,
        "first_k_dense_replace": 1,
        "num_experts": 32,
        "linear_attn_config": {
            "kda_layers": [i + 1 for i in [0] + kda],
            "full_attn_layers": [i + 1 for i in mla],
        },
    }
    dst = os.path.join(WORK, f"bench_k3_x{copies}")
    return make_reduced_model_dir(_src(name), dst, over, text_config=True), {"kda": kda, "mla": mla}


# ---------------------------------------------------------------------------


def build_engine(path: str, rank: int, tp: int, max_bs: int, ctx: int):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine
    from minisgl.scheduler import SchedulerConfig

    from minisgl.attention import model_attention_backend
    from minisgl.models import ModelConfig
    from minisgl.utils import cached_load_hf_config

    mc = ModelConfig.from_hf(cached_load_hf_config(path))
    P = model_attention_backend(mc)[1]
    pages = max_bs * ((ctx + 2 + P - 1) // P) + 8
    cfg = SchedulerConfig(
        model_path=path,
        tp_info=DistributedInfo(rank, tp),
        dtype=torch.bfloat16,
        max_running_req=max_bs,
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
        use_dummy_weight=True,
        max_seq_len_override=ctx + 64,
        num_page_override=pages,
        max_extend_tokens=2048,
        memory_ratio=0.9,
        cache_type="naive",
        offline_mode=True,
    )
    return Engine(cfg)


def make_decode_batch(eng, B: int, C: int):
    """B requests at context C (token C is decoded); pages of request i are
    contiguous so compressed groups never straddle pages."""
    from minisgl.core import Batch, Req, SamplingParams

    P = eng.ctx.page_size
    L = C + 1
    np_ = (L + P - 1) // P
    dev = eng.device
    pt = eng.page_table
    reqs = []
    for i in range(B):
        base = i * np_ * P
        pt[i, :L] = torch.arange(base, base + L, dtype=torch.int32, device=dev)
        reqs.append(
            Req(
                input_ids=torch.zeros(L, dtype=torch.int32),
                table_idx=i, cached_len=C, output_len=1, uid=i,
                sampling_params=SamplingParams(), cache_handle=None,  # type: ignore[arg-type]
            )
        )
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = list(reqs)
    g = torch.Generator().manual_seed(B)
    V = int(eng.model._config.vocab_size)
    batch.input_ids = torch.randint(0, V, (B,), generator=g, dtype=torch.int32).to(dev)
    batch.positions = torch.full((B,), C, dtype=torch.int32, device=dev)
    batch.out_loc = pt[torch.arange(B, device=dev), batch.positions.long()]
    return batch


def time_graph(fn, warmup: int, iters: int, trials: int) -> tuple[float, float]:
    """Capture ``fn`` in a CUDA graph; mean replay time (us) median over trials, CV."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    res = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        b.synchronize()
        res.append(a.elapsed_time(b) / iters * 1e3)
    del g
    med = statistics.median(res)
    cv = statistics.pstdev(res) / max(statistics.mean(res), 1e-9)
    return med, cv


def profile_fn(fn, label: str, steps: int = 3) -> None:
    """Kernel table of ``fn`` (eager) for a breakdown of the per-layer time."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    print(f"===== profile {label} ({steps} calls)", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=90), flush=True)


# ---------------------------------------------------------------------------


def dsv4_fns(model, ids: list[int], B: int, dev):
    from minisgl.kernel.dsv4_triton import hc_post_fused

    m = model.model
    a = m._args
    x0 = torch.randn(B, a.hc_mult, a.dim, device=dev, dtype=torch.bfloat16)
    h0 = torch.randn(B, a.dim, device=dev, dtype=torch.bfloat16)
    # residual of the previous layer (post / comb mixing weights of a real step)
    _, res0 = m.layers.op_list[ids[0]].attention_part(x0)
    layers = [m.layers.op_list[i] for i in ids]

    def attn():
        for L in layers:
            L.attn.forward(h0)

    def aside():
        for L in layers:
            x = hc_post_fused(h0, res0.x, res0.post, res0.comb)
            h, _ = L.attention_part(x)
            L.ffn.gate.forward(h)

    return attn, aside


def k3_fns(model, ids: list[int], B: int, dev, nb: int):
    from minisgl.models.kimi_k3 import K3Residual

    m = model.model
    a = m._args
    h0 = torch.randn(B, a.dim, device=dev, dtype=torch.bfloat16)
    buf = torch.randn(B, nb + 2, a.dim, device=dev, dtype=torch.bfloat16)  # nb block summaries + prefix slot
    y_lat = torch.randn(B, a.latent, device=dev, dtype=torch.bfloat16)
    layers = [m.layers.op_list[i] for i in ids]

    def attn():
        for L in layers:
            L.self_attn.forward(h0)

    def aside():
        for L in layers:
            res = K3Residual(buf=buf, nb=nb, prefix=h0 + h0)
            h, res = L.attention_part(res)
            moe = L.block_sparse_moe
            moe.gate.forward(h)
            moe.routed_expert_down_proj.forward(h)
            sh = moe.shared_experts.forward(h) if moe.shared_experts is not None else None
            moe.post_combine(y_lat, sh)

    return attn, aside


def run_rank(rank: int, args) -> list[dict] | None:
    path, ids = bench_model_dir(args.model, args.copies)
    bss = [int(b) for b in args.bs.split(",")]
    eng = build_engine(path, rank, args.tp, max(bss), args.ctx)
    dev = eng.device
    model = eng.model
    backend = eng.attn_backend
    is_k3 = args.model == "k3"
    if is_k3:
        cfg = json.load(open(os.path.join(_src("k3"), "config.json")))["text_config"]
        blk = int(cfg["attn_res_block_size"])
        nb = max(1, ((int(cfg["num_hidden_layers"]) + blk - 1) // blk) // 2)  # mid-depth block count
    mix = full_layer_mix(args.model)
    rows = []
    for B in bss:
        batch = make_decode_batch(eng, B, args.ctx)
        backend.init_capture_graph(args.ctx + 64, [B])
        backend.prepare_for_capture(batch)
        with eng.ctx.forward_batch(batch):
            per_type = {}
            for t, lids in ids.items():
                if is_k3:
                    attn, aside = k3_fns(model, lids, B, dev, nb)
                else:
                    attn, aside = dsv4_fns(model, lids, B, dev)
                if args.profile:
                    profile_fn(aside, f"{args.model} {t} B={B} C={args.ctx}")
                    continue
                ta, cva = time_graph(attn, args.warmup, args.iters, args.trials)
                tb, cvb = time_graph(aside, args.warmup, args.iters, args.trials)
                n = len(lids)
                per_type[t] = {"attn_us": ta / n, "aside_us": tb / n, "cv_attn": cva, "cv_aside": cvb}
        if args.profile:
            continue
        tot = sum(mix.values())
        avg = {
            k: sum(per_type[t][k] * c for t, c in mix.items()) / tot for k in ("attn_us", "aside_us")
        }
        row = {
            "model": args.model, "tp": args.tp, "rank": rank, "bs": B, "ctx": args.ctx,
            "impl": os.environ.get("MINISGL_ASIDE_IMPL_TAG", "minisgl"),
            "per_type": per_type, "layer_mix": mix, "model_avg": avg,
        }
        rows.append(row)
        if rank == 0:
            pt = " ".join(f"{t}={v['attn_us']:.1f}/{v['aside_us']:.1f}" for t, v in per_type.items())
            print(f"[bench_aside] {args.model} tp={args.tp} B={B} C={args.ctx}: attn/aside us per layer {pt} "
                  f"| avg {avg['attn_us']:.1f}/{avg['aside_us']:.1f}", flush=True)
        torch.cuda.synchronize()
    return rows


def _worker(rank: int, args, q) -> None:
    os.environ["MINISGL_DEVICE_INDEX"] = str(rank)
    try:
        rows = run_rank(rank, args)
        q.put((rank, rows, None))
    except Exception as e:  # pragma: no cover - surfaced by the parent
        import traceback

        q.put((rank, None, traceback.format_exc()))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["dsv4_flash", "dsv4_pro", "k3"])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--bs", default="8,32,128")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", action="store_true", help="kernel table of the aside path instead of timing")
    args = ap.parse_args()
    os.environ.setdefault("MINISGL_MOE_EXEC", "torch")
    if args.tp == 1:
        rows = run_rank(0, args)
    else:
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(r, args, q)) for r in range(args.tp)]
        for p in procs:
            p.start()
        got = {}
        for _ in procs:
            r, rows_r, err = q.get()
            if err:
                print(f"rank {r} failed:\n{err}", flush=True)
            got[r] = rows_r
        for p in procs:
            p.join()
        rows = got.get(0)
        if rows is None:
            sys.exit(1)
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
