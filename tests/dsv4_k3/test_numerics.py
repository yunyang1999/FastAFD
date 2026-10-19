"""Numerics of the minisgl DSV4 / K3 implementations against plain-torch
transcriptions of the official references (tests/dsv4_k3/ref_*.py).

  DSV4K3_META_DIR=... DSV4K3_WORK_DIR=... python tests/dsv4_k3/test_numerics.py dsv4_flash|dsv4_pro|k3

Reduced depth, reduced expert count, BF16 non-expert weights (FP8 path checked
separately), torch MoE executor (exact model activation). Reports for each
check max |diff|, relative L2 error and argmax agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from harness import EngineDriver, make_reduced_model_dir  # noqa: E402

META = os.environ.get("DSV4K3_META_DIR", "")
WORK = os.environ.get("DSV4K3_WORK_DIR", "/tmp/dsv4k3_work")
RESULTS = []


def cmp(name, got, ref, *, logits=False):
    got, ref = got.float(), ref.float()
    d = (got - ref).abs()
    rel = (d.norm() / ref.norm().clamp_min(1e-12)).item()
    line = f"{name}: max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} rel_l2={rel:.3e}"
    agree = None
    if logits:
        agree = float((got.argmax(-1) == ref.argmax(-1)).float().mean().item())
        line += f" argmax_agree={agree:.3f}"
    print(line, flush=True)
    RESULTS.append({"name": name, "max_abs": d.max().item(), "rel_l2": rel, "argmax_agree": agree})
    return rel


def dsv4_dir(name, layers, experts, fp8):
    src = os.path.join(META, "DeepSeek-V4-Flash" if name == "dsv4_flash" else "DeepSeek-V4-Pro")
    cfg = json.load(open(os.path.join(src, "config.json")))
    ratios = cfg["compress_ratios"]
    pick = [0, 2, 3, 4, 5, 6, 7, 8][:layers]
    over = {"num_hidden_layers": layers, "compress_ratios": [ratios[i] for i in pick] + [0],
            "num_hash_layers": 1, "n_routed_experts": experts}
    if not fp8:
        over["quantization_config"] = None
    tag = f"{name}_L{layers}_E{experts}_{'fp8' if fp8 else 'bf16'}"
    d = make_reduced_model_dir(src, os.path.join(WORK, tag), over)
    if not fp8:
        c = json.load(open(os.path.join(d, "config.json")))
        c.pop("quantization_config", None)
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
    return d


def k3_dir(layers, experts):
    src = os.path.join(META, "Kimi-K3")
    over = {
        "num_hidden_layers": layers,
        "attn_res_block_size": 2,
        "num_experts": experts,
        "linear_attn_config": {
            "kda_layers": [i for i in range(1, layers + 1) if i % 4 != 0],
            "full_attn_layers": [i for i in range(1, layers + 1) if i % 4 == 0],
        },
    }
    return make_reduced_model_dir(src, os.path.join(WORK, f"k3_L{layers}_E{experts}_num"), over, text_config=True)


# ---------------------------------------------------------------------------
# weight export to reference parameter dicts
# ---------------------------------------------------------------------------


def _w(lin):
    w = lin.weight
    if w.dtype == torch.float8_e4m3fn:
        return lin._dequant_weight_to_bf16()
    return w


def dsv4_ref_params(model):
    P = {}
    m = model.model
    P["embed.weight"] = m.embed.weight
    P["norm.weight"] = m.norm.weight
    P["head.weight"] = model.lm_head.weight
    for k in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        P[k] = getattr(m, k)
    experts, shared = [], []
    for i, blk in enumerate(m.layers.op_list):
        pf = f"layers.{i}"
        for k in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            P[f"{pf}.{k}"] = getattr(blk, k)
        P[f"{pf}.attn_norm.weight"] = blk.attn_norm.weight
        P[f"{pf}.ffn_norm.weight"] = blk.ffn_norm.weight
        a = blk.attn
        ap = f"{pf}.attn"
        P[f"{ap}.wq_a.weight"] = _w(a.wq_a)
        P[f"{ap}.q_norm.weight"] = a.q_norm.weight
        P[f"{ap}.wq_b.weight"] = _w(a.wq_b)
        P[f"{ap}.wkv.weight"] = _w(a.wkv)
        P[f"{ap}.kv_norm.weight"] = a.kv_norm.weight
        P[f"{ap}.wo_a"] = a.wo_a
        P[f"{ap}.wo_b.weight"] = _w(a.wo_b)
        P[f"{ap}.attn_sink"] = a.attn_sink
        for cname, comp in (("compressor", a.compressor), ("indexer.compressor", a.indexer.compressor if a.indexer else None)):
            if comp is None:
                continue
            cp = f"{ap}.{cname}"
            P[f"{cp}.wkv.weight"] = comp.wkv.weight
            P[f"{cp}.wgate.weight"] = comp.wgate.weight
            P[f"{cp}.ape"] = comp.ape
            P[f"{cp}.norm.weight"] = comp.norm.weight
        if a.indexer is not None:
            P[f"{ap}.indexer.wq_b.weight"] = _w(a.indexer.wq_b)
            P[f"{ap}.indexer.weights_proj.weight"] = a.indexer.weights_proj.weight
        f = blk.ffn
        fp = f"{pf}.ffn"
        P[f"{fp}.gate.weight"] = f.gate.weight
        if f.gate.is_hash:
            P[f"{fp}.gate.tid2eid"] = f.gate.tid2eid
        else:
            P[f"{fp}.gate.bias"] = f.gate.bias
        experts.append(f.experts.bf16())
        shared.append((_w(f.shared_experts.gate_up), _w(f.shared_experts.down)))
    return P, experts, shared


def dsv4_ref_args(model):
    a = model.model._args
    return {
        "n_layers": a.n_layers, "n_heads": a.n_heads, "head_dim": a.head_dim, "rope_dim": a.rope_dim,
        "o_groups": a.o_groups, "o_lora_rank": a.o_lora_rank, "window": a.window, "ratios": a.ratios,
        "eps": a.eps, "index_n_heads": a.index_n_heads, "index_head_dim": a.index_head_dim,
        "index_topk": a.index_topk, "hc_mult": a.hc_mult, "hc_iters": a.hc_iters, "hc_eps": a.hc_eps,
        "topk": a.topk, "rsf": a.rsf, "swiglu_limit": a.swiglu_limit, "n_hash_layers": a.n_hash_layers,
        "rope_theta": a.rope_theta, "compress_rope_theta": a.compress_rope_theta, "rope_factor": a.rope_factor,
        "original_seq_len": a.original_seq_len, "beta_fast": a.beta_fast, "beta_slow": a.beta_slow,
    }


def k3_ref_params(model):
    P = {}
    m = model.model
    a = m._args
    P["model.embed_tokens.weight"] = m.embed_tokens.weight
    P["model.norm.weight"] = m.norm.weight
    P["model.output_attn_res_norm.weight"] = m.output_attn_res_norm.weight
    P["model.output_attn_res_proj.weight"] = m.output_attn_res_proj
    P["lm_head.weight"] = model.lm_head.weight
    experts = {}
    for i, layer in enumerate(m.layers.op_list):
        pf = f"model.layers.{i}"
        for k in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            P[f"{pf}.{k}.weight"] = getattr(layer, k).weight
        P[f"{pf}.self_attention_res_proj.weight"] = layer.self_attention_res_proj
        P[f"{pf}.mlp_res_proj.weight"] = layer.mlp_res_proj
        sa = layer.self_attn
        ap = f"{pf}.self_attn"
        if layer.is_kda:
            proj = a.kda_heads * a.kda_dim
            qkv = sa.qkv_proj.weight
            P[f"{ap}.q_proj.weight"], P[f"{ap}.k_proj.weight"], P[f"{ap}.v_proj.weight"] = qkv.split(proj, 0)
            cw = sa.conv_weight
            P[f"{ap}.q_conv1d.weight"], P[f"{ap}.k_conv1d.weight"], P[f"{ap}.v_conv1d.weight"] = cw.split(proj, 0)
            for k in ("f_a_proj", "f_b_proj", "b_proj", "g_proj", "o_proj"):
                P[f"{ap}.{k}.weight"] = getattr(sa, k).weight
            P[f"{ap}.A_log"] = sa.A_log
            P[f"{ap}.dt_bias"] = sa.dt_bias
            P[f"{ap}.o_norm.weight"] = sa.o_norm.weight
        else:
            for k in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "o_proj"):
                P[f"{ap}.{k}.weight"] = getattr(sa, k).weight
            P[f"{ap}.q_a_layernorm.weight"] = sa.q_a_layernorm.weight
            P[f"{ap}.kv_a_layernorm.weight"] = sa.kv_a_layernorm.weight
        if layer.block_sparse_moe is not None:
            mo = layer.block_sparse_moe
            mp = f"{pf}.block_sparse_moe"
            P[f"{mp}.gate.weight"] = mo.gate.weight
            P[f"{mp}.gate.e_score_correction_bias"] = mo.gate.e_score_correction_bias
            P[f"{mp}.routed_expert_down_proj.weight"] = mo.routed_expert_down_proj.weight
            P[f"{mp}.routed_expert_up_proj.weight"] = mo.routed_expert_up_proj.weight
            P[f"{mp}.routed_expert_norm.weight"] = mo.routed_expert_norm.weight
            gu = mo.shared_experts.gate_up_proj.weight
            P[f"{mp}.shared_experts.gate_proj.weight"], P[f"{mp}.shared_experts.up_proj.weight"] = gu.chunk(2, 0)
            P[f"{mp}.shared_experts.down_proj.weight"] = mo.shared_experts.down_proj.weight
            experts[i] = mo.experts.bf16()
        else:
            gu = layer.mlp.gate_up_proj.weight
            P[f"{pf}.mlp.gate_proj.weight"], P[f"{pf}.mlp.up_proj.weight"] = gu.chunk(2, 0)
            P[f"{pf}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight
    return P, experts


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def run_mine(drv, uid, prompt, steps, next_tokens, chunk=None, use_graph=False):
    """Returns (list of logits per call, list of tap lists per call)."""
    from minisgl.utils import debug_taps

    drv.add_request(uid, prompt)
    outs, taps = [], []
    debug_taps.enable()
    outs.append(drv.prefill([uid], chunk=chunk)[0])
    taps.append(list(debug_taps.TAPS))
    for s in range(steps):
        debug_taps.enable()
        outs.append(drv.decode([uid], [next_tokens[s]], use_graph=use_graph)[0])
        taps.append(list(debug_taps.TAPS))
    debug_taps.disable()
    return outs, taps


def tap_get(taps, name):
    for n, t in taps:
        if n == name:
            return t
    raise KeyError(name)


def numerics_dsv4(args):
    os.environ["MINISGL_MOE_EXEC"] = "torch"
    path = dsv4_dir(args.model, args.layers, args.experts, args.fp8)
    drv = EngineDriver(path, max_running_req=8, max_seq_len=1024, cuda_graph_bs=[1, 2, 4], max_extend_tokens=1024)
    from ref_dsv4 import RefDSV4, RefAttention, RefMoE

    model = drv.engine.model
    P, experts, shared = dsv4_ref_params(model)
    rargs = dsv4_ref_args(model)
    dev = drv.device
    V = model._config.vocab_size
    g = torch.Generator().manual_seed(1)
    lens = [int(x) for x in args.lens.split(",")]
    uid = 0
    for L in lens:
        prompt = torch.randint(0, V, (L,), generator=g).tolist()
        nxt = torch.randint(0, V, (args.steps,), generator=g).tolist()
        mine, taps = run_mine(drv, uid, prompt, args.steps, nxt)
        uid += 1
        ref = RefDSV4(P, rargs, experts, shared, max_seq_len=1024, dev=dev)
        toks = torch.tensor(prompt, device=dev)
        r_logits = [ref.forward(toks.view(1, -1), 0)[0]]
        for s in range(args.steps):
            r_logits.append(ref.forward(torch.tensor([[nxt[s]]], device=dev), L + s)[0])
        for s, (a, b) in enumerate(zip(mine, r_logits)):
            cmp(f"dsv4 L={L} full-forward logits call{s}", a, b, logits=True)
        # same, with the reference forced onto my routed-expert ids (removes
        # top-k flips from router near-ties, which random weights make common)
        import ref_dsv4

        ref_f = RefDSV4(P, rargs, experts, shared, max_seq_len=1024, dev=dev)
        for s in range(args.steps + 1):
            ref_dsv4.FORCE_IDS = {l: tap_get(taps[s], f"L{l}.topk_ids") for l in range(rargs["n_layers"])}
            if s == 0:
                rf = ref_f.forward(toks.view(1, -1), 0)[0]
            else:
                rf = ref_f.forward(torch.tensor([[nxt[s - 1]]], device=dev), L + s - 1)[0]
            cmp(f"dsv4 L={L} full-forward logits (forced routing) call{s}", mine[s], rf, logits=True)
        if args.fp8:  # reference with the FP8 activation quantisation of the FP8 GEMMs
            ref_dsv4.ACT_QUANT = True
            ref_f = RefDSV4(P, rargs, experts, shared, max_seq_len=1024, dev=dev)
            for s in range(args.steps + 1):
                ref_dsv4.FORCE_IDS = {l: tap_get(taps[s], f"L{l}.topk_ids") for l in range(rargs["n_layers"])}
                if s == 0:
                    rf = ref_f.forward(toks.view(1, -1), 0)[0]
                else:
                    rf = ref_f.forward(torch.tensor([[nxt[s - 1]]], device=dev), L + s - 1)[0]
                cmp(f"dsv4 L={L} full-forward logits (forced routing, ref FP8 act-quant) call{s}", mine[s], rf, logits=True)
            ref_dsv4.ACT_QUANT = False
        ref_dsv4.FORCE_IDS = None
        # module-level replay with my inputs
        for l in range(rargs["n_layers"]):
            ra = RefAttention(P, l, rargs, 1024, dev)
            rm = RefMoE(P, l, rargs, experts[l], shared[l])
            all_ids = prompt + nxt
            for c, tp in enumerate(taps):
                x = tap_get(tp, f"L{l}.attn_in")
                start = 0 if c == 0 else L + c - 1
                y = ra.forward(x.unsqueeze(0), start)[0]
                cmp(f"dsv4 L={L} layer{l} ratio{rargs['ratios'][l]} attention call{c}", tap_get(tp, f"L{l}.attn_out"), y)
                xm = tap_get(tp, f"L{l}.moe_in")
                ids = torch.tensor(all_ids[start : start + xm.shape[0]], device=dev)
                ym = rm.forward(xm.unsqueeze(0), ids.unsqueeze(0))[0]
                cmp(f"dsv4 L={L} layer{l} moe call{c}", tap_get(tp, f"L{l}.moe_out"), ym)
        # chunked prefill and graph decode consistency on the same prompt
        mine_c, _ = run_mine(drv, uid, prompt, args.steps, nxt, chunk=64)
        uid += 1
        mine_g, _ = run_mine(drv, uid, prompt, args.steps, nxt, use_graph=True)
        uid += 1
        for s in range(args.steps + 1):
            cmp(f"dsv4 L={L} chunked(64)-vs-oneshot call{s}", mine_c[s], mine[s], logits=True)
            cmp(f"dsv4 L={L} graph-vs-eager call{s}", mine_g[s], mine[s], logits=True)
        for u in list(drv.reqs):
            drv.free_tables.append(drv.reqs[u].table_idx)
        drv.reqs.clear()


def numerics_k3(args):
    os.environ["MINISGL_MOE_EXEC"] = "torch"
    path = k3_dir(args.layers, args.experts)
    drv = EngineDriver(path, max_running_req=8, max_seq_len=1024, cuda_graph_bs=[1, 2, 4], max_extend_tokens=1024,
                       num_pages=16384)
    from ref_kimi import RefKimiK3, RefKDA, RefMLA

    model = drv.engine.model
    P, experts = k3_ref_params(model)
    cfg = model._config.hf_text_config
    dev = drv.device
    V = model._config.vocab_size
    g = torch.Generator().manual_seed(1)
    lens = [int(x) for x in args.lens.split(",")]
    uid = 0
    for L in lens:
        prompt = torch.randint(0, V, (L,), generator=g).tolist()
        nxt = torch.randint(0, V, (args.steps,), generator=g).tolist()
        mine, taps = run_mine(drv, uid, prompt, args.steps, nxt)
        uid += 1
        import ref_kimi

        ref = RefKimiK3(P, cfg, experts, dev)
        ref_kimi.TAPS = []
        r_logits = [ref.forward(torch.tensor(prompt, device=dev))[0]]
        ref_taps = [list(ref_kimi.TAPS)]
        for s in range(args.steps):
            ref_kimi.TAPS = []
            r_logits.append(ref.forward(torch.tensor([nxt[s]], device=dev))[0])
            ref_taps.append(list(ref_kimi.TAPS))
        ref_kimi.TAPS = None
        for s, (a, b) in enumerate(zip(mine, r_logits)):
            cmp(f"k3 L={L} full-forward logits call{s}", a, b, logits=True)
        # same, with the reference forced onto my routed-expert ids
        ref_f = RefKimiK3(P, cfg, experts, dev)
        for s in range(args.steps + 1):
            ref_kimi.FORCE_IDS = {l: tap_get(taps[s], f"L{l}.topk_ids") for l in experts}
            ids_s = torch.tensor(prompt if s == 0 else [nxt[s - 1]], device=dev)
            cmp(f"k3 L={L} full-forward logits (forced routing) call{s}", mine[s], ref_f.forward(ids_s)[0], logits=True)
        ref_kimi.FORCE_IDS = None
        # per-layer divergence in the full forward + routing-set agreement
        for c in (0, 1):
            for l in range(cfg["num_hidden_layers"]):
                xm, xr = tap_get(taps[c], f"L{l}.moe_in"), tap_get(ref_taps[c], f"L{l}.moe_in")
                cmp(f"k3 L={L} call{c} full-forward moe_in layer{l}", xm, xr)
                mo = model.model.layers.op_list[l].block_sparse_moe
                if mo is not None:
                    im, _ = mo.gate.forward(xm)
                    ir, _ = mo.gate.forward(xr)
                    same = (im.sort(-1).values == ir.sort(-1).values).all(-1).float().mean().item()
                    print(f"k3 L={L} call{c} layer{l} routing-set agreement (own inputs) {same:.4f}", flush=True)
        kda1 = set(cfg["linear_attn_config"]["kda_layers"])
        for l in range(cfg["num_hidden_layers"]):
            pf = f"model.layers.{l}.self_attn"
            mod = RefKDA(P, pf, cfg, dev) if (l + 1) in kda1 else RefMLA(P, pf, cfg, dev)
            kind = "kda" if (l + 1) in kda1 else "mla"
            for c, tp in enumerate(taps):
                y = mod.forward(tap_get(tp, f"L{l}.attn_in"))
                cmp(f"k3 L={L} layer{l} {kind} call{c}", tap_get(tp, f"L{l}.attn_out"), y)
                if l in experts:
                    ym = ref.moe(tap_get(tp, f"L{l}.moe_in"), l)
                    cmp(f"k3 L={L} layer{l} latent-moe call{c}", tap_get(tp, f"L{l}.moe_out"), ym)
                else:
                    ym = ref.mlp(tap_get(tp, f"L{l}.moe_in"), f"model.layers.{l}.mlp")
                    cmp(f"k3 L={L} layer{l} dense-mlp call{c}", tap_get(tp, f"L{l}.moe_out"), ym)
        mine_c, _ = run_mine(drv, uid, prompt, args.steps, nxt, chunk=64)
        uid += 1
        mine_g, _ = run_mine(drv, uid, prompt, args.steps, nxt, use_graph=True)
        uid += 1
        for s in range(args.steps + 1):
            cmp(f"k3 L={L} chunked(64)-vs-oneshot call{s}", mine_c[s], mine[s], logits=True)
            cmp(f"k3 L={L} graph-vs-eager call{s}", mine_g[s], mine[s], logits=True)
        for u in list(drv.reqs):
            drv.free_tables.append(drv.reqs[u].table_idx)
        drv.reqs.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lens", default="9,150,300")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.model.startswith("dsv4"):
        numerics_dsv4(args)
    else:
        numerics_k3(args)
    if args.out:
        json.dump(RESULTS, open(args.out, "w"), indent=1)
    print("NUMERICS DONE")


if __name__ == "__main__":
    main()
