"""Config-only model directories for dummy-weight DeepSeek-V4 / Kimi-K3 runs.

Copies config.json + tokenizer files of an HF checkpoint (weights are not
needed with --dummy-weight) and optionally reduces depth / expert count while
keeping every layer kind present:

* DeepSeek-V4: layer 0 = SWA only, then C4 / C128 compressed layers; one hash
  routing layer.
* Kimi-K3: layer 0 = dense FFN; every 4th layer (1-indexed) is MLA, the rest
  KDA, so 4 layers = 3 KDA + 1 MLA with 3 LatentMoE layers.

  python make_model_dir.py --src HF_DIR --dst OUT --layers 4
  python make_model_dir.py --src HF_DIR --dst OUT            # full depth
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def make_model_dir(src: str, dst: str, *, layers: int = 0, experts: int = 0, attn_res_block: int = 0) -> str:
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    for f in s.iterdir():
        if f.is_file() and f.suffix in (".json", ".model", ".txt", ".tiktoken") and "index" not in f.name:
            if f.name != "config.json":
                shutil.copy(f, d / f.name)
    cfg = json.loads((s / "config.json").read_text())
    if cfg.get("model_type") == "deepseek_v4":
        if layers:
            ratios = cfg["compress_ratios"]
            pick = ([0] + list(range(2, len(ratios))))[:layers]
            cfg["compress_ratios"] = [ratios[i] for i in pick] + [0]
            cfg["num_hidden_layers"] = layers
            cfg["num_hash_layers"] = min(int(cfg.get("num_hash_layers", 0)), 1)
        if experts:
            cfg["n_routed_experts"] = experts
    elif "text_config" in cfg:  # Kimi-K3
        t = cfg["text_config"]
        if layers:
            t["num_hidden_layers"] = layers
            t["linear_attn_config"]["kda_layers"] = [i for i in range(1, layers + 1) if i % 4 != 0]
            t["linear_attn_config"]["full_attn_layers"] = [i for i in range(1, layers + 1) if i % 4 == 0]
        if experts:
            t["num_experts"] = experts
        if attn_res_block:
            t["attn_res_block_size"] = attn_res_block
    else:
        raise SystemExit(f"unsupported model_type {cfg.get('model_type')!r}")
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", type=int, default=0, help="0 keeps full depth")
    ap.add_argument("--experts", type=int, default=0, help="0 keeps the expert count")
    ap.add_argument("--attn-res-block", type=int, default=0)
    a = ap.parse_args()
    print(make_model_dir(a.src, a.dst, layers=a.layers, experts=a.experts, attn_res_block=a.attn_res_block))


if __name__ == "__main__":
    main()
