"""Checkpoint-loader tests for DeepSeek-V4 / Kimi-K3 (models/ckpt_loaders.py).

roundtrip MODEL
    Dummy-initialise a reduced model, export it in the official checkpoint
    format (the inverse of the loader, written here independently), load it
    back with ``load_hybrid_checkpoint`` / ``expert_layer_weights`` and require
    bit-exact equality. Values the checkpoint format cannot represent are
    canonicalised first (fp32 params stored as BF16, power-of-two UE8M0 FP8
    scales, FP8 storage of DSV4 ``wo_a``). Covers the colocated model, the AFD
    attention stage and the expert-worker slices (+ DSV4 shared slot).

real MODEL --shards DIR
    Load the downloaded shards of a real checkpoint into a truncated-depth
    model (``MINISGL_CKPT_LAYER_MAP`` picks the real layers), print weight
    statistics, check the expert-worker loader against the colocated load and
    run prefill + decode through the engine against the torch reference on the
    same (real) weights.

  DSV4K3_META_DIR=... DSV4K3_WORK_DIR=... python tests/dsv4_k3/test_ckpt_loader.py roundtrip dsv4_flash
  python tests/dsv4_k3/test_ckpt_loader.py real k3 --shards DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from harness import make_reduced_model_dir  # noqa: E402

META = os.environ.get("DSV4K3_META_DIR", "")
WORK = os.environ.get("DSV4K3_WORK_DIR", "/tmp/dsv4k3_work")
FAILS: list[str] = []

_DT = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
}


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


# ---------------------------------------------------------------------------
# minimal safetensors writer
# ---------------------------------------------------------------------------


def st_write(path: str, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        t, dt = tensors[name]
        b = t.detach().contiguous().cpu().view(torch.uint8).flatten()
        header[name] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + b.numel()]}
        blobs.append(b)
        off += b.numel()
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b.numpy().tobytes())


def write_checkpoint(out_dir: str, tensors: dict, group) -> None:
    files: dict[str, dict] = {}
    for name, v in tensors.items():
        files.setdefault(group(name), {})[name] = v
    weight_map = {}
    for i, (g, ts) in enumerate(sorted(files.items())):
        fn = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
        st_write(os.path.join(out_dir, fn), ts)
        weight_map.update({n: fn for n in ts})
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))


# ---------------------------------------------------------------------------
# canonicalisation + export (inverse loader)
# ---------------------------------------------------------------------------


def pow2_scale(s: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.round(torch.log2(s.float())))


def e8m0_bytes(s: torch.Tensor) -> torch.Tensor:
    return (torch.round(torch.log2(s.float())) + 127).to(torch.uint8)


def quant_fp8_pow2(w: torch.Tensor, block: int = 128):
    out, inp = w.shape
    v = w.float().view(out // block, block, inp // block, block)
    amax = v.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (v / s).to(torch.float8_e4m3fn)
    deq = (q.float() * s).view(out, inp).to(torch.bfloat16)
    return q.view(out, inp), s.view(out // block, inp // block), deq


def bf16_round(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(t.dtype)


def canon_and_export_dsv4(S: dict) -> tuple[dict, dict]:
    S = dict(S)
    T = {}
    for name in list(S):
        if name.endswith(".weight_scale"):
            S[name] = pow2_scale(S[name])
    for name, t in list(S.items()):
        if name == "model.embed.weight":
            T["embed.weight"] = (t, "BF16")
            continue
        if name == "lm_head.weight":
            T["head.weight"] = (t, "BF16")
            continue
        if name == "model.norm.weight":
            T["norm.weight"] = (t, "BF16")
            continue
        if name.startswith("model.hc_head_"):
            T[name[len("model.") :]] = (t, "F32")
            continue
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        assert m, name
        i, rest = int(m.group(1)), m.group(2)
        p = f"layers.{i}."
        if rest == "attn.wo_a":
            q, s, deq = quant_fp8_pow2(t)
            S[name] = deq
            T[p + "attn.wo_a.weight"] = (q, "F8_E4M3")
            T[p + "attn.wo_a.scale"] = (e8m0_bytes(s), "F8_E8M0")
            continue
        if rest.startswith("ffn.experts."):
            leaf = rest.rsplit(".", 1)[1]
            if leaf != "w13":
                continue
            w13, s13 = S[name], S[name + "_sf"]
            w2, s2 = S[name.replace("w13", "w2")], S[name.replace("w13", "w2") + "_sf"]
            inter = w13.shape[1] // 2
            for e in range(w13.shape[0]):
                q = f"{p}ffn.experts.{e}."
                T[q + "w1.weight"] = (w13[e, :inter].view(torch.int8), "I8")
                T[q + "w3.weight"] = (w13[e, inter:].view(torch.int8), "I8")
                T[q + "w1.scale"] = (s13[e, :inter], "F8_E8M0")
                T[q + "w3.scale"] = (s13[e, inter:], "F8_E8M0")
                T[q + "w2.weight"] = (w2[e].view(torch.int8), "I8")
                T[q + "w2.scale"] = (s2[e], "F8_E8M0")
            continue
        if rest.startswith("ffn.shared_experts.gate_up."):
            half = t.shape[0] // 2
            leaf = rest.rsplit(".", 1)[1]
            for w, part in (("w1", t[:half]), ("w3", t[half:])):
                if leaf == "weight":
                    T[f"{p}ffn.shared_experts.{w}.weight"] = (part, "F8_E4M3")
                else:
                    T[f"{p}ffn.shared_experts.{w}.scale"] = (e8m0_bytes(part), "F8_E8M0")
            continue
        rest_ck = rest.replace("ffn.shared_experts.down.", "ffn.shared_experts.w2.")
        if rest.endswith(".weight_scale"):
            T[p + rest_ck[: -len("weight_scale")] + "scale"] = (e8m0_bytes(t), "F8_E8M0")
        elif t.dtype == torch.float8_e4m3fn:
            T[p + rest_ck] = (t, "F8_E4M3")
        elif rest == "ffn.gate.weight" or re.search(r"compressor\.(wkv|wgate)\.weight$", rest):
            S[name] = bf16_round(t)
            T[p + rest_ck] = (t.to(torch.bfloat16), "BF16")
        elif rest.endswith("tid2eid"):
            T[p + rest_ck] = (t.long(), "I64")
        else:
            T[p + rest_ck] = (t, _DT[t.dtype])
    return S, T


def canon_and_export_k3(S: dict, cfg: dict) -> tuple[dict, dict]:
    S = dict(S)
    T = {}
    P = "language_model."
    for name, t in list(S.items()):
        if name == "lm_head.weight":
            T[P + name] = (t, "BF16")
            continue
        if name == "model.output_attn_res_proj":
            T[P + name + ".weight"] = (t, "BF16")
            continue
        if not name.startswith("model.layers."):
            T[P + name] = (t, _DT[t.dtype])
            continue
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        i, rest = int(m.group(1)), m.group(2)
        p = f"{P}model.layers.{i}."
        if rest in ("self_attention_res_proj", "mlp_res_proj"):
            T[p + rest + ".weight"] = (t, "BF16")
        elif rest == "self_attn.qkv_proj.weight":
            for c, part in zip("qkv", t.chunk(3, 0)):
                T[f"{p}self_attn.{c}_proj.weight"] = (part, "BF16")
        elif rest == "self_attn.conv_weight":
            for c, part in zip("qkv", t.chunk(3, 0)):
                T[f"{p}self_attn.{c}_conv1d.weight"] = (part.unsqueeze(1).float(), "F32")
        elif rest == "self_attn.A_log":
            pad = torch.zeros(128, dtype=torch.float32, device=t.device)
            pad[: t.numel()] = t.float()
            T[p + rest] = (pad, "F32")
        elif rest.endswith("gate_up_proj.weight"):
            base = rest[: -len("gate_up_proj.weight")]
            g, u = t.chunk(2, 0)
            T[p + base + "gate_proj.weight"] = (g, "BF16")
            T[p + base + "up_proj.weight"] = (u, "BF16")
        elif rest.startswith("block_sparse_moe.experts."):
            leaf = rest.rsplit(".", 1)[1]
            if leaf != "w13":
                continue
            w13, s13 = S[name], S[name + "_sf"]
            w2, s2 = S[name.replace("w13", "w2")], S[name.replace("w13", "w2") + "_sf"]
            inter = w13.shape[1] // 2
            for e in range(w13.shape[0]):
                q = f"{p}block_sparse_moe.experts.{e}."
                T[q + "w1.weight_packed"] = (w13[e, :inter], "U8")
                T[q + "w3.weight_packed"] = (w13[e, inter:], "U8")
                T[q + "w1.weight_scale"] = (s13[e, :inter], "U8")
                T[q + "w3.weight_scale"] = (s13[e, inter:], "U8")
                T[q + "w2.weight_packed"] = (w2[e], "U8")
                T[q + "w2.weight_scale"] = (s2[e], "U8")
        elif rest == "block_sparse_moe.gate.weight":
            S[name] = bf16_round(t)
            T[p + rest] = (t.to(torch.bfloat16), "BF16")
        elif rest == "self_attn.o_norm.weight":
            T[p + rest] = (t.float(), "F32")
        else:
            T[p + rest] = (t, _DT[t.dtype])
    return S, T


def _group(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    return f"L{int(m.group(1)):03d}" if m else "misc"


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------


def meta_model(path: str, device, *, stage: bool = False):
    from minisgl.distributed import set_tp_info
    from minisgl.layers import set_rope_device
    from minisgl.models import create_model
    from minisgl.models.afd_hybrid import attention_stage_class
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config, torch_dtype

    set_tp_info(0, 1)
    set_rope_device(device)
    mc = ModelConfig.from_hf(cached_load_hf_config(path))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = attention_stage_class(mc)(mc) if stage else create_model(mc)
    return model, mc


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def compare_state(tag: str, got: dict, want: dict) -> None:
    bad = [n for n in want if n in got and not same(got[n].cpu(), want[n].cpu())]
    missing = [n for n in want if n not in got]
    check(not bad and not missing, f"{tag}: {len(want)} tensors bit-exact"
          + (f" (mismatch {bad[:4]}, missing {missing[:4]})" if bad or missing else ""))


# ---------------------------------------------------------------------------
# roundtrip
# ---------------------------------------------------------------------------


def roundtrip(model_name: str, device) -> None:
    from minisgl.models.afd_hybrid import hybrid_afd_spec
    from minisgl.models.ckpt_loaders import expert_layer_weights, load_hybrid_checkpoint
    from minisgl.models.dummy_init import init_dummy_state_dict
    from minisgl.models.fp8_utils import dequant_fp8_block
    from minisgl.models.moe_common import quant_mxfp4

    if model_name.startswith("dsv4"):
        src = os.path.join(META, "DeepSeek-V4-Flash" if model_name == "dsv4_flash" else "DeepSeek-V4-Pro")
        cfg = json.load(open(os.path.join(src, "config.json")))
        r = cfg["compress_ratios"]
        over = {"num_hidden_layers": 4, "compress_ratios": [r[0], r[2], r[3], r[4], 0], "num_hash_layers": 1,
                "n_routed_experts": 16}
        path = make_reduced_model_dir(src, os.path.join(WORK, f"ckrt_{model_name}"), over)
    else:
        over = {"num_hidden_layers": 4, "num_experts": 16,
                "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}}
        path = make_reduced_model_dir(os.path.join(META, "Kimi-K3"), os.path.join(WORK, "ckrt_k3"), over, text_config=True)
    for f in os.listdir(path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(path, f))
    model, mc = meta_model(path, device)
    S = {k: v.cpu() for k, v in init_dummy_state_dict(model, device).items()}
    if mc.is_deepseek_v4:
        S, T = canon_and_export_dsv4(S)
    else:
        S, T = canon_and_export_k3(S, mc.hf_text_config)
    write_checkpoint(path, T, _group)
    print(f"exported {len(T)} checkpoint tensors to {path}", flush=True)

    got = load_hybrid_checkpoint(model, path, device)
    compare_state(f"{model_name} colocated model", got, S)
    del got
    stage, _ = meta_model(path, device, stage=True)
    got = load_hybrid_checkpoint(stage, path, device, role="attention")
    want = {k: S[k] for k in stage.state_dict()}
    compare_state(f"{model_name} AFD attention stage", got, want)
    del got

    spec = hybrid_afd_spec(mc)
    eg_size = 2
    for eg_rank in range(eg_size):
        load = expert_layer_weights(path, spec, eg_rank, eg_size, device)
        for layer in spec.moe_layer_ids:
            w13, s13, w2, s2 = (t.cpu() for t in load(layer))
            pre = "model.layers.%d.%s." % (layer, "ffn.experts" if mc.is_deepseek_v4 else "block_sparse_moe.experts")
            n = spec.num_experts // eg_size
            sl = slice(eg_rank * n, (eg_rank + 1) * n)
            ok = (same(w13[:n], S[pre + "w13"][sl]) and same(s13[:n], S[pre + "w13_sf"][sl])
                  and same(w2[:n], S[pre + "w2"][sl]) and same(s2[:n], S[pre + "w2_sf"][sl]))
            if spec.shared_experts_per_rank:
                sp = "model.layers.%d.ffn.shared_experts." % layer
                gu = dequant_fp8_block(S[sp + "gate_up.weight"].to(device), S[sp + "gate_up.weight_scale"].to(device))
                dn = dequant_fp8_block(S[sp + "down.weight"].to(device), S[sp + "down.weight_scale"].to(device))
                p13, q13 = quant_mxfp4(gu)
                p2, q2 = quant_mxfp4(dn)
                ok = ok and w13.shape[0] == n + 1 and same(w13[n], p13.cpu()) and same(s13[n], q13.cpu()) \
                    and same(w2[n], p2.cpu()) and same(s2[n], q2.cpu())
            check(ok, f"{model_name} expert rank {eg_rank}/{eg_size} layer {layer}: routed slice"
                  + (" + MXFP4 shared slot" if spec.shared_experts_per_rank else "") + " exact")


# ---------------------------------------------------------------------------
# real shards
# ---------------------------------------------------------------------------


def real_dir(model_name: str, shards: str) -> tuple[str, dict]:
    if model_name.startswith("dsv4"):
        cfg = json.load(open(os.path.join(shards, "config.json")))
        r = cfg["compress_ratios"]
        lmap = {0: 0, 1: 2, 2: 3}  # SWA+hash, C4+indexer+hash, C128 + scored routing
        over = {"num_hidden_layers": 3, "compress_ratios": [r[0], r[2], r[3], 0], "num_hash_layers": 2}
        path = make_reduced_model_dir(shards, os.path.join(WORK, f"real_{model_name}"), over)
    else:
        lmap = {0: 0, 1: 3}  # KDA + dense MLP, MLA + LatentMoE
        over = {"num_hidden_layers": 2, "linear_attn_config": {"kda_layers": [1], "full_attn_layers": [2]}}
        path = make_reduced_model_dir(shards, os.path.join(WORK, f"real_{model_name}"), over, text_config=True)
    import shutil

    shutil.copy(os.path.join(shards, "model.safetensors.index.json"), path)
    for f in os.listdir(shards):
        if f.endswith(".safetensors") and not os.path.exists(os.path.join(path, f)):
            os.symlink(os.path.join(shards, f), os.path.join(path, f))
    os.environ["MINISGL_CKPT_LAYER_MAP"] = ",".join(f"{d}:{s}" for d, s in lmap.items())
    return path, lmap


def real(model_name: str, shards: str, device, args) -> None:
    from minisgl.models.afd_hybrid import hybrid_afd_spec
    from minisgl.models.ckpt_loaders import expert_layer_weights, load_hybrid_checkpoint
    from minisgl.models.moe_common import dequant_mxfp4

    path, lmap = real_dir(model_name, shards)
    model, mc = meta_model(path, device)
    S = load_hybrid_checkpoint(model, path, device)
    print(f"loaded {len(S)} tensors, {sum(t.numel() * t.element_size() for t in S.values()) / 2**30:.2f} GiB", flush=True)
    for name in sorted(S):
        t = S[name]
        if t.dtype in (torch.uint8, torch.int32, torch.int64):
            continue
        tf = t.float()
        if not torch.isfinite(tf).all():
            check(False, f"non-finite values in {name}")
    keys = [k for k in S if re.search(r"(attn\.wq_a\.weight_scale|A_log|gate\.weight|experts\.w13_sf|hc_attn_scale|e_score_correction_bias)$", k)]
    for k in keys[:12]:
        t = S[k].float()
        print(f"  {k}: shape={tuple(t.shape)} mean={t.mean().item():.4g} std={t.std().item():.4g} min={t.min().item():.4g} max={t.max().item():.4g}")
    # MXFP4 experts: dequantised RMS per expert and gate/up structure
    spec = hybrid_afd_spec(mc)
    for layer in spec.moe_layer_ids:
        pre = "model.layers.%d.%s." % (layer, "ffn.experts" if mc.is_deepseek_v4 else "block_sparse_moe.experts")
        w = dequant_mxfp4(S[pre + "w13"][:4], S[pre + "w13_sf"][:4]).float()
        print(f"  layer {layer} experts 0-3 dequant w13 rms={w.pow(2).mean().sqrt().item():.4g} "
              f"absmax={w.abs().max().item():.4g}", flush=True)
        eg_size = 8
        load = expert_layer_weights(path, spec, 1, eg_size, device)
        w13, s13, w2, s2 = load(layer)
        n = spec.num_experts // eg_size
        ok = same(w13[:n].cpu(), S[pre + "w13"][n : 2 * n].cpu()) and same(s2[:n].cpu(), S[pre + "w2_sf"][n : 2 * n].cpu())
        check(ok, f"{model_name} real layer {layer} (ckpt {lmap.get(layer, layer)}): expert rank 1/{eg_size} slice matches colocated load")
        if spec.shared_experts_per_rank:
            check(w13.shape[0] == n + 1, f"{model_name} real layer {layer}: MXFP4 shared slot appended")
    del S, model
    torch.cuda.empty_cache()
    if args.no_forward:
        return
    # engine forward on real weights vs the torch reference on the same weights
    os.environ["MINISGL_MOE_EXEC"] = "torch"
    import test_numerics as tn
    from harness import EngineDriver

    drv = EngineDriver(path, max_running_req=4, max_seq_len=1024, cuda_graph_bs=[1], max_extend_tokens=1024,
                       use_dummy_weight=False, num_pages=4096 if not mc.is_deepseek_v4 else None)
    m = drv.engine.model
    g = torch.Generator().manual_seed(7)
    L = args.len
    prompt = torch.randint(0, mc.vocab_size, (L,), generator=g).tolist()
    nxt = torch.randint(0, mc.vocab_size, (args.steps,), generator=g).tolist()
    mine, taps = tn.run_mine(drv, 0, prompt, args.steps, nxt)
    if mc.is_deepseek_v4:
        from ref_dsv4 import RefDSV4
        import ref_dsv4

        P, experts, shared = tn.dsv4_ref_params(m)
        rargs = tn.dsv4_ref_args(m)
        for forced in (False, True):
            ref = RefDSV4(P, rargs, experts, shared, max_seq_len=1024, dev=drv.device)
            for s in range(args.steps + 1):
                ref_dsv4.FORCE_IDS = {l: tn.tap_get(taps[s], f"L{l}.topk_ids") for l in range(rargs["n_layers"])} if forced else None
                ids = torch.tensor([prompt] if s == 0 else [[nxt[s - 1]]], device=drv.device)
                r = ref.forward(ids, 0 if s == 0 else L + s - 1)[0]
                tn.cmp(f"real {model_name} full-forward logits{' (forced routing)' if forced else ''} call{s}", mine[s], r, logits=True)
        ref_dsv4.FORCE_IDS = None
    else:
        from ref_kimi import RefKimiK3
        import ref_kimi

        P, experts = tn.k3_ref_params(m)
        cfg = m._config.hf_text_config
        for forced in (False, True):
            ref = RefKimiK3(P, cfg, experts, drv.device)
            for s in range(args.steps + 1):
                ref_kimi.FORCE_IDS = {l: tn.tap_get(taps[s], f"L{l}.topk_ids") for l in experts} if forced else None
                ids = torch.tensor(prompt if s == 0 else [nxt[s - 1]], device=drv.device)
                tn.cmp(f"real {model_name} full-forward logits{' (forced routing)' if forced else ''} call{s}",
                       mine[s], ref.forward(ids)[0], logits=True)
        ref_kimi.FORCE_IDS = None
    top = [int(x.argmax()) for x in mine]
    print(f"  argmax tokens: {top}", flush=True)
    for x in mine:
        check(bool(torch.isfinite(x).all()), f"real {model_name} logits finite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["roundtrip", "real"])
    ap.add_argument("model", choices=["dsv4_flash", "dsv4_pro", "k3"])
    ap.add_argument("--shards", default="")
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-forward", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.mode == "roundtrip":
        roundtrip(args.model, device)
    else:
        real(args.model, args.shards, device, args)
    print(f"CKPT LOADER {'FAILED (' + str(len(FAILS)) + ')' if FAILS else 'OK'}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
