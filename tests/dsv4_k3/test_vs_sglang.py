"""Cross-check minisgl's DSV4 / K3 kernels against sglang 0.5.18's kernels.

Run with a Python that has sglang installed (e.g. a separate sglang env) on one
GPU; minisgl's kernel files only need torch + triton and are loaded by path:

  python tests/dsv4_k3/test_vs_sglang.py

Same random inputs on both sides:
  * KDA prefill: q/k/v short conv (sglang causal_conv1d_fn) + gated delta rule
    with safe gate (chunk_kda), varlen batch with and without initial state;
    output, final recurrent state, final conv state.
  * KDA decode: causal_conv1d_update + fused_sigmoid_gating_delta_rule_update
    from a common state.
  * DSV4 mHC: mhc_pre (prenorm GEMM + Sinkhorn split + pre-mix) and mhc_post.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = []


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmp(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    got, ref = got.float(), ref.float()
    d = (got - ref).abs()
    rel = (d.norm() / ref.norm().clamp_min(1e-12)).item()
    ok = rel <= tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} "
          f"rel_l2={rel:.3e} (tol {tol})", flush=True)
    RESULTS.append(ok)


def kda_checks(dev) -> None:
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update

    mine = load("mine_kda", "python/minisgl/kernel/kda_triton.py")
    g = torch.Generator(device=dev).manual_seed(0)
    H, D, KW, LB = 8, 128, 4, -5.0
    HD = H * D
    lens = [37, 64, 5]
    N, T = len(lens), sum(lens)
    slots = 6
    idx = torch.tensor([4, 1, 2], dtype=torch.int32, device=dev)
    has_init = torch.tensor([1, 0, 1], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 37, 101, 106], dtype=torch.int32, device=dev)

    def rn(*shape, std=1.0):
        return torch.randn(*shape, device=dev, generator=g) * std

    x = rn(T, 3 * HD).bfloat16()
    w = rn(3 * HD, KW, std=0.5).bfloat16()  # bf16-representable for both sides
    g_raw = rn(T, HD).bfloat16()
    b_raw = rn(T, H).bfloat16()
    A_log = torch.log(torch.rand(H, device=dev, generator=g) * 15 + 1)
    dt_bias = rn(HD, std=0.1)
    conv0 = rn(slots, 3 * HD, KW - 1).bfloat16()
    ssm0 = rn(slots, H, D, D, std=0.05)
    for s, init in zip(idx.tolist(), has_init.tolist()):
        if not init:  # sglang expects zeroed slots for fresh requests
            conv0[s].zero_()
            ssm0[s].zero_()

    # ---------------- prefill ----------------
    conv_m, ssm_m = conv0.clone(), ssm0.clone()
    y = mine.short_conv_varlen(x, w.float(), conv_m, cu, idx, has_init, max(lens))
    q_m, k_m, v_m = y[:, :HD], y[:, HD : 2 * HD], y[:, 2 * HD :]
    o_m = mine.kda_recurrent(q_m, k_m, v_m, g_raw, b_raw, A_log, dt_bias, ssm_m, cu, idx, has_init,
                             num_heads=H, head_dim=D, scale=D**-0.5, lower_bound=LB)

    conv_s, ssm_s = conv0.clone(), ssm0.clone()
    xs = x.t()
    outs = []
    for j in range(3):
        sl = slice(j * HD, (j + 1) * HD)
        outs.append(causal_conv1d_fn(
            xs[sl], w[sl], None, conv_states=conv_s[:, sl], query_start_loc=cu, seq_lens_cpu=lens,
            cache_indices=idx, has_initial_state=has_init.bool(), activation="silu",
        ).t())
    q_s, k_s, v_s = (t.unflatten(-1, (H, D)).unsqueeze(0) for t in outs)
    cmp("KDA prefill short conv (q|k|v)", y, torch.cat(outs, -1), 1e-2)
    o_s = chunk_kda(q=q_s, k=k_s, v=v_s, g=g_raw.unflatten(-1, (H, D)).unsqueeze(0),
                    beta=b_raw.float().sigmoid().unsqueeze(0), initial_state=ssm_s, initial_state_indices=idx,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu.long(), A_log=A_log, dt_bias=dt_bias,
                    lower_bound=LB)
    if isinstance(o_s, tuple):
        o_s = o_s[0]
    cmp("KDA prefill output (chunk_kda)", o_m, o_s[0], 2e-2)
    cmp("KDA prefill final state", ssm_m[idx.long()], ssm_s[idx.long()], 2e-2)
    cmp("KDA prefill final conv state", conv_m[idx.long()], conv_s[idx.long()], 1e-2)

    # ---------------- decode (common start = minisgl post-prefill state) ----------------
    xd = rn(N, 3 * HD).bfloat16()
    gd = rn(N, HD).bfloat16()
    bd = rn(N, H).bfloat16()
    conv_m2, ssm_m2 = conv_m.clone(), ssm_m.clone()
    conv_s2, ssm_s2 = conv_m.clone(), ssm_m.clone()
    one = torch.ones(N, dtype=torch.int32, device=dev)
    cud = torch.arange(N + 1, dtype=torch.int32, device=dev)
    yd = mine.short_conv_decode(xd, w.float(), conv_m2, idx)
    od_m = mine.kda_recurrent(yd[:, :HD], yd[:, HD : 2 * HD], yd[:, 2 * HD :], gd, bd, A_log, dt_bias, ssm_m2,
                              cud, idx, one, num_heads=H, head_dim=D, scale=D**-0.5, lower_bound=LB)
    yd_s = causal_conv1d_update(xd, conv_s2, w, None, activation="silu", conv_state_indices=idx)
    cmp("KDA decode short conv", yd, yd_s, 1e-2)
    qd, kd, vd = (t.unflatten(-1, (H, D)).unsqueeze(0) for t in yd_s.split(HD, -1))
    od_s = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, dt_bias=dt_bias, q=qd, k=kd, v=vd, a=gd, b=bd.unsqueeze(0),
        initial_state_source=ssm_s2, initial_state_indices=idx, cu_seqlens=cud,
        use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, is_kda=True, lower_bound=LB,
    )
    cmp("KDA decode output", od_m, od_s.reshape(N, H, D), 2e-2)
    cmp("KDA decode state", ssm_m2[idx.long()], ssm_s2[idx.long()], 2e-2)


def mhc_checks(dev) -> None:
    import sglang.kernels.ops.layernorm.mhc as M

    # standalone use: no TP group / symmetric memory / DSA context parallelism
    M.get_tp_group = lambda: None
    M.is_allocation_symmetric = lambda: False
    M.use_symmetric_memory = lambda *a, **k: contextlib.nullcontext()
    M.is_dsa_prefill_cp_round_robin_split = lambda: False

    mine = load("mine_dsv4", "python/minisgl/kernel/dsv4_triton.py")
    g = torch.Generator(device=dev).manual_seed(1)
    T, HC, Dm, iters, eps = 33, 4, 4096, 20, 1e-6
    mix = (2 + HC) * HC
    res = (torch.randn(T, HC, Dm, device=dev, generator=g)).bfloat16()
    fn = torch.randn(mix, HC * Dm, device=dev, generator=g) / (HC * Dm) ** 0.5
    scale = torch.tensor([0.7, 0.3, 0.2], device=dev)
    base = torch.randn(mix, device=dev, generator=g) * 0.1
    mixes = mine.hc_prenorm_mixes(res, fn, eps)
    y_m, post_m, comb_m = mine.hc_pre_fused(res, mixes, scale, base, sinkhorn_iters=iters, eps=eps)
    _, post_k, comb_k = M.hc_split_sinkhorn(mixes.view(1, T, mix), scale, base, HC, iters, eps)
    cmp("mHC Sinkhorn post (hc_split_sinkhorn, same mixes)", post_m, post_k[0], 1e-4)
    cmp("mHC Sinkhorn comb (hc_split_sinkhorn, same mixes)", comb_m, comb_k[0], 1e-4)
    post_s, comb_s, y_s = M.mhc_pre(res, fn, scale, base, eps, eps, eps, 2.0, iters)
    cmp("mHC pre layer input (mhc_pre)", y_m, y_s, 1e-2)
    cmp("mHC pre post-mix (mhc_pre)", post_m, post_s.view(T, HC), 1e-2)
    cmp("mHC pre comb (mhc_pre)", comb_m, comb_s.view(T, HC, HC), 1e-2)
    o = torch.randn(T, Dm, device=dev, generator=g).bfloat16()
    out_m = mine.hc_post_fused(o, res, post_m, comb_m)
    out_s = M.mhc_post(o, res, post_m.view(T, HC, 1), comb_m)
    cmp("mHC post (mhc_post)", out_m, out_s, 1e-2)


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    for fn in (kda_checks, mhc_checks):
        try:
            fn(dev)
        except Exception as exc:  # report and continue with the next group
            import traceback

            traceback.print_exc()
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
            RESULTS.append(False)
    n_ok = sum(RESULTS)
    print(f"SGLANG CROSS-CHECK: {n_ok}/{len(RESULTS)} passed", flush=True)
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
