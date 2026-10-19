"""Real-checkpoint loading for DeepSeek-V4 and Kimi-K3.

Both loaders are driven by the target model's own (meta) state dict: every
parameter name is resolved to checkpoint tensor(s), transformed into the model
layout, TP-sharded when the model holds a shard, and checked for shape. Only the
byte ranges of the requested tensors are read (mmap), so an attention worker
never touches routed-expert bytes and an expert worker reads only its experts.

Checkpoint formats
------------------
DeepSeek-V4 (``deepseek-ai/DeepSeek-V4-{Flash,Pro}``)
  * names as in the reference inference code (``layers.{i}.attn.wq_a.weight``,
    ``embed.weight``, ``head.weight``, ``hc_head_*``; ``mtp.*`` is not loaded);
  * linear weights F8_E4M3 with UE8M0 128x128 block scales ``<name>.scale``
    (F8_E8M0) -> ``weight`` + fp32 ``weight_scale``;
  * ``attn.wo_a`` is dequantised to BF16 (as the reference converter does);
  * routed experts MXFP4: ``experts.{e}.w{1,2,3}.weight`` I8 [N, K/2] (E2M1
    pairs, low nibble = even element) + ``.scale`` F8_E8M0 [N, K/32]
    -> stacked ``w13`` (w1 rows then w3 rows) / ``w2`` + ``*_sf`` uint8;
  * shared expert FP8 (w1/w3 merged into ``gate_up``); for AFD it runs on the
    expert workers, so it is dequantised and re-quantised to MXFP4 there;
  * gate / compressor weights BF16 (promoted to FP32), ``tid2eid`` I64.

Kimi-K3 (``moonshotai/Kimi-K3``, text part only)
  * ``language_model.`` prefix, BF16 except the routed experts;
  * routed experts compressed-tensors ``mxfp4-pack-quantized``:
    ``weight_packed`` U8 [N, K/2] (low nibble first) + ``weight_scale`` U8
    UE8M0 [N, K/32];
  * KDA ``q/k/v_proj`` -> merged ``qkv_proj``, ``{q,k,v}_conv1d.weight``
    [P, 1, 4] -> ``conv_weight`` [3P, 4]; ``A_log`` is stored padded to 128
    entries (the first ``num_heads`` are real); MLP ``gate/up_proj`` -> merged.

``MINISGL_CKPT_LAYER_MAP="0:0,1:2,2:3"`` maps model layer -> checkpoint layer;
it lets truncated-depth configs load selected real layers (loader tests).
"""

from __future__ import annotations

import glob
import json
import mmap
import os
import re
import struct
import warnings
from typing import Callable, Dict, List, Sequence

import torch

_ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.uint8,  # raw UE8M0 exponent bytes
}


class CheckpointReader:
    """Name -> tensor access over a local (sharded) safetensors checkpoint."""

    def __init__(self, model_path: str):
        if not os.path.isdir(model_path):
            raise ValueError(f"real-weight loading needs a local checkpoint directory, got {model_path!r}")
        self.folder = model_path
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, "r", encoding="utf-8") as f:
                self.where: Dict[str, str] = dict(json.load(f)["weight_map"])
        else:
            self.where = {}
            for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
                for name in self._header(os.path.basename(path))[1]:
                    if name != "__metadata__":
                        self.where[name] = os.path.basename(path)
        self._files: Dict[str, tuple] = {}

    def _header(self, fn: str):
        with open(os.path.join(self.folder, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return 8 + n, json.loads(f.read(n))

    def _open(self, fn: str):
        if fn not in self._files:
            base, header = self._header(fn)
            fobj = open(os.path.join(self.folder, fn), "rb")
            mm = mmap.mmap(fobj.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[fn] = (fobj, mm, base, header)
        return self._files[fn]

    def has(self, name: str) -> bool:
        fn = self.where.get(name)
        return fn is not None and os.path.exists(os.path.join(self.folder, fn))

    def get(self, name: str) -> torch.Tensor:
        fn = self.where.get(name)
        if fn is None:
            raise KeyError(f"{name} is not in the checkpoint index of {self.folder}")
        if not os.path.exists(os.path.join(self.folder, fn)):
            raise FileNotFoundError(f"{name} lives in {fn}, which is not present in {self.folder}")
        _, mm, base, header = self._open(fn)
        meta = header[name]
        start, end = meta["data_offsets"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer
            raw = torch.frombuffer(mm, dtype=torch.uint8, count=end - start, offset=base + start)
        t = raw.clone().view(_ST_DTYPES[meta["dtype"]])
        return t.reshape(meta["shape"])

    def close(self) -> None:
        for fobj, mm, _, _ in self._files.values():
            mm.close()
            fobj.close()
        self._files.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ckpt_layer_map() -> Dict[int, int]:
    out: Dict[int, int] = {}
    for part in os.environ.get("MINISGL_CKPT_LAYER_MAP", "").split(","):
        if part.strip():
            dst, src = part.split(":")
            out[int(dst)] = int(src)
    return out


def e8m0_to_float(t: torch.Tensor) -> torch.Tensor:
    return torch.exp2(t.view(torch.uint8).to(torch.float32) - 127.0)


def _tp_rank_size() -> tuple[int, int]:
    from minisgl.distributed import try_get_tp_info

    info = try_get_tp_info()
    return (0, 1) if info is None else (int(info.rank), int(info.size))


def _cat0(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    dt = parts[0].dtype
    if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.cat([p.view(torch.uint8) for p in parts], 0).view(dt)
    return torch.cat(list(parts), 0)


def _merge_rows(name: str, parts: Sequence[torch.Tensor], like_rows: int, rank: int, size: int) -> torch.Tensor:
    """Concatenate column-parallel components, sharding each first if the
    model holds a TP shard."""
    total = sum(int(p.shape[0]) for p in parts)
    if total == like_rows:
        return _cat0(parts)
    if total == like_rows * size:
        shards = []
        for p in parts:
            n = int(p.shape[0]) // size
            shards.append(p.narrow(0, rank * n, n))
        return _cat0(shards)
    raise RuntimeError(f"{name}: merged rows {total} incompatible with model rows {like_rows} (tp={size})")


def _fit(name: str, t: torch.Tensor, like: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    want = tuple(like.shape)
    if tuple(t.shape) == want:
        return t
    if t.dim() == len(want):
        diff = [d for d in range(t.dim()) if t.shape[d] != want[d]]
        if len(diff) == 1 and t.shape[diff[0]] == want[diff[0]] * size:
            d = diff[0]
            return t.narrow(d, rank * want[d], want[d])
    if t.numel() == like.numel():
        return t.reshape(want)
    raise RuntimeError(f"{name}: checkpoint shape {tuple(t.shape)} does not fit model shape {want} (tp={size})")


def _finish(name: str, t: torch.Tensor, like: torch.Tensor, device) -> torch.Tensor:
    if t.dtype != like.dtype:
        if like.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(f"{name}: dtype {t.dtype} cannot be loaded into {like.dtype}")
        t = t.to(like.dtype)
    return t.contiguous().to(device)


def _stack_experts(get: Callable[[int, str], torch.Tensor], lo: int, hi: int) -> tuple:
    """get(e, part) with part in w1, w1_sf, w2, w2_sf, w3, w3_sf (uint8)."""
    w13, s13, w2, s2 = [], [], [], []
    for e in range(lo, hi):
        w13.append(torch.cat([get(e, "w1"), get(e, "w3")], 0))
        s13.append(torch.cat([get(e, "w1_sf"), get(e, "w3_sf")], 0))
        w2.append(get(e, "w2"))
        s2.append(get(e, "w2_sf"))
    return torch.stack(w13), torch.stack(s13), torch.stack(w2), torch.stack(s2)


# ---------------------------------------------------------------------------
# DeepSeek-V4
# ---------------------------------------------------------------------------


class _DSV4Source:
    def __init__(self, reader: CheckpointReader):
        self.r = reader
        self.lmap = ckpt_layer_map()
        self._expert_cache: Dict[tuple, tuple] = {}

    def _src(self, layer: int) -> str:
        return f"layers.{self.lmap.get(layer, layer)}."

    def expert_part(self, src: str, e: int, part: str) -> torch.Tensor:
        w = part[:2]
        if part.endswith("_sf"):
            return self.r.get(f"{src}ffn.experts.{e}.{w}.scale").view(torch.uint8)
        return self.r.get(f"{src}ffn.experts.{e}.{w}.weight").view(torch.uint8)

    def experts(self, layer: int, lo: int, hi: int) -> tuple:
        key = (layer, lo, hi)
        if key not in self._expert_cache:
            self._expert_cache.clear()
            src = self._src(layer)
            self._expert_cache[key] = _stack_experts(lambda e, p: self.expert_part(src, e, p), lo, hi)
        return self._expert_cache[key]

    def shared_fp8(self, layer: int) -> Dict[str, torch.Tensor]:
        src = self._src(layer) + "ffn.shared_experts."
        return {w: (self.r.get(f"{src}{w}.weight"), self.r.get(f"{src}{w}.scale")) for w in ("w1", "w2", "w3")}

    def tensor(self, name: str, like: torch.Tensor, rank: int, size: int) -> torch.Tensor:
        r = self.r
        if name == "model.embed.weight":
            return r.get("embed.weight")
        if name == "lm_head.weight":
            return r.get("head.weight")
        if name == "model.norm.weight":
            return r.get("norm.weight")
        if name.startswith("model.hc_head_"):
            return r.get(name[len("model.") :])
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        if m is None:
            raise KeyError(f"no DeepSeek-V4 checkpoint mapping for {name}")
        layer, rest = int(m.group(1)), m.group(2)
        src = self._src(layer)
        if rest == "attn.wo_a":
            from .fp8_utils import dequant_fp8_block

            w = r.get(src + "attn.wo_a.weight")
            s = e8m0_to_float(r.get(src + "attn.wo_a.scale"))
            return dequant_fp8_block(w, s)
        if rest.startswith("ffn.experts."):
            n_local = int(like.shape[0])
            leaf = rest.rsplit(".", 1)[1]
            lo = rank * n_local if size > 1 else 0
            w13, s13, w2, s2 = self.experts(layer, lo, lo + n_local)
            return {"w13": w13, "w13_sf": s13, "w2": w2, "w2_sf": s2}[leaf]
        if rest.startswith("ffn.shared_experts.gate_up."):
            parts = [src + "ffn.shared_experts.w1", src + "ffn.shared_experts.w3"]
            if rest.endswith("weight_scale"):
                return _merge_rows(name, [e8m0_to_float(r.get(p + ".scale")) for p in parts], int(like.shape[0]), rank, size)
            return _merge_rows(name, [r.get(p + ".weight") for p in parts], int(like.shape[0]), rank, size)
        if rest.startswith("ffn.shared_experts.down."):
            rest = rest.replace("ffn.shared_experts.down.", "ffn.shared_experts.w2.")
        if rest.endswith(".weight_scale"):
            return e8m0_to_float(r.get(src + rest[: -len("weight_scale")] + "scale"))
        return r.get(src + rest)


# ---------------------------------------------------------------------------
# Kimi-K3
# ---------------------------------------------------------------------------


class _K3Source:
    PREFIX = "language_model."

    def __init__(self, reader: CheckpointReader, text_config: dict):
        self.r = reader
        self.cfg = text_config
        self.lmap = ckpt_layer_map()
        self._expert_cache: Dict[tuple, tuple] = {}

    def _src(self, layer: int) -> str:
        return f"{self.PREFIX}model.layers.{self.lmap.get(layer, layer)}."

    def expert_part(self, src: str, e: int, part: str) -> torch.Tensor:
        w = part[:2]
        leaf = "weight_scale" if part.endswith("_sf") else "weight_packed"
        return self.r.get(f"{src}block_sparse_moe.experts.{e}.{w}.{leaf}").view(torch.uint8)

    def experts(self, layer: int, lo: int, hi: int) -> tuple:
        key = (layer, lo, hi)
        if key not in self._expert_cache:
            self._expert_cache.clear()
            src = self._src(layer)
            self._expert_cache[key] = _stack_experts(lambda e, p: self.expert_part(src, e, p), lo, hi)
        return self._expert_cache[key]

    def tensor(self, name: str, like: torch.Tensor, rank: int, size: int) -> torch.Tensor:
        r, P = self.r, self.PREFIX
        if name == "lm_head.weight":
            return r.get(P + "lm_head.weight")
        if name == "model.output_attn_res_proj":
            return r.get(P + "model.output_attn_res_proj.weight")
        if not name.startswith("model.layers."):
            return r.get(P + name)
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        layer, rest = int(m.group(1)), m.group(2)
        src = self._src(layer)
        rows = int(like.shape[0]) if like.dim() else 1
        if rest in ("self_attention_res_proj", "mlp_res_proj"):
            return r.get(src + rest + ".weight")
        if rest == "self_attn.qkv_proj.weight":
            return _merge_rows(name, [r.get(f"{src}self_attn.{c}_proj.weight") for c in "qkv"], rows, rank, size)
        if rest == "self_attn.conv_weight":
            parts = [r.get(f"{src}self_attn.{c}_conv1d.weight") for c in "qkv"]
            parts = [p.reshape(p.shape[0], -1) for p in parts]
            return _merge_rows(name, parts, rows, rank, size)
        if rest == "self_attn.A_log":
            heads = int(self.cfg["linear_attn_config"]["num_heads"])
            return r.get(src + rest).flatten()[:heads]  # stored padded to 128
        if rest.endswith("gate_up_proj.weight"):
            base = src + rest[: -len("gate_up_proj.weight")]
            return _merge_rows(name, [r.get(base + "gate_proj.weight"), r.get(base + "up_proj.weight")], rows, rank, size)
        if rest.startswith("block_sparse_moe.experts."):
            n_local = int(like.shape[0])
            leaf = rest.rsplit(".", 1)[1]
            lo = rank * n_local if size > 1 else 0
            w13, s13, w2, s2 = self.experts(layer, lo, lo + n_local)
            return {"w13": w13, "w13_sf": s13, "w2": w2, "w2_sf": s2}[leaf]
        return r.get(src + rest)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def _source_for(model_config, reader: CheckpointReader):
    if getattr(model_config, "is_deepseek_v4", False):
        return _DSV4Source(reader)
    if getattr(model_config, "is_kimi_k3", False):
        return _K3Source(reader, model_config.hf_text_config)
    raise RuntimeError(f"no hybrid checkpoint loader for {getattr(model_config, 'architectures', None)}")


def load_hybrid_checkpoint(model, model_path: str, device, *, role: str = "full") -> Dict[str, torch.Tensor]:
    """State dict for ``model`` (a colocated DSV4/K3 model or an AFD attention
    stage; ``role`` is informational) built from a real checkpoint."""
    from minisgl.utils import init_logger

    logger = init_logger(__name__)
    mc = model._config
    reader = CheckpointReader(model_path)
    src = _source_for(mc, reader)
    rank, size = _tp_rank_size()
    out: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    try:
        for name, like in model.state_dict().items():
            try:
                t = src.tensor(name, like, rank, size)
            except (KeyError, FileNotFoundError) as exc:
                missing.append(f"{name}: {exc}")
                continue
            t = _fit(name, t, like, rank, size)
            out[name] = _finish(name, t, like, device)
    finally:
        reader.close()
    if missing:
        raise RuntimeError(f"{len(missing)} tensors could not be loaded from {model_path}:\n  " + "\n  ".join(missing[:20]))
    total = sum(t.numel() * t.element_size() for t in out.values())
    logger.info(f"Loaded {len(out)} tensors ({total / 2**30:.2f} GiB) for role={role} from {model_path}")
    return out


def expert_layer_weights(model_path: str, spec, eg_rank: int, eg_size: int, device) -> Callable[[int], tuple]:
    """Per-MoE-layer packed MXFP4 weights of one expert rank for
    ``MegaMoEM2NAfdAdapter.register_eg_layer_fp4_packed_weights``: routed
    experts [eg_rank * E/F, (eg_rank + 1) * E/F) plus, when the profile puts
    the shared expert on the F side (DSV4), a replicated MXFP4 copy of it as the
    last slot."""
    from minisgl.utils import cached_load_hf_config

    from .config import ModelConfig
    from .moe_common import quant_mxfp4

    mc = ModelConfig.from_hf(cached_load_hf_config(model_path))
    reader = CheckpointReader(model_path)
    src = _source_for(mc, reader)
    n_real = int(spec.num_experts) // int(eg_size)
    lo, hi = eg_rank * n_real, (eg_rank + 1) * n_real

    def load(layer_id: int) -> tuple:
        w13, s13, w2, s2 = (t.to(device) for t in src.experts(layer_id, lo, hi))
        src._expert_cache.clear()
        if int(spec.shared_experts_per_rank):
            from .fp8_utils import dequant_fp8_block

            sh = src.shared_fp8(layer_id)
            deq = {w: dequant_fp8_block(t.to(device), e8m0_to_float(s).to(device)) for w, (t, s) in sh.items()}
            p13, q13 = quant_mxfp4(torch.cat([deq["w1"], deq["w3"]], 0))
            p2, q2 = quant_mxfp4(deq["w2"])
            w13 = torch.cat([w13, p13.unsqueeze(0)])
            s13 = torch.cat([s13, q13.unsqueeze(0)])
            w2 = torch.cat([w2, p2.unsqueeze(0)])
            s2 = torch.cat([s2, q2.unsqueeze(0)])
        return w13.contiguous(), s13.contiguous(), w2.contiguous(), s2.contiguous()

    return load


__all__ = ["CheckpointReader", "ckpt_layer_map", "e8m0_to_float", "load_hybrid_checkpoint", "expert_layer_weights"]
