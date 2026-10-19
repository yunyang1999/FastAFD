"""Minimal driver for the minisgl Engine (no tokenizer / scheduler).

Allocates request slots and pages, builds prefill / decode Batches and runs the
model (eager or through the CUDA-graph runner), returning logits. Used by the
DSV4 / K3 bring-up and numerics tests.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch


def make_reduced_model_dir(src_dir: str, dst_dir: str, overrides: dict, text_config: bool = False) -> str:
    """Copy config/tokenizer metadata of ``src_dir`` with config overrides."""
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".json", ".py", ".model", ".txt") and "index" not in f.name and f.name != "config.json":
            shutil.copy(f, dst / f.name)
    cfg = json.loads((src / "config.json").read_text())
    target = cfg["text_config"] if text_config else cfg
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k].update(v)
        else:
            target[k] = v
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return str(dst)


@dataclass
class _ReqState:
    table_idx: int
    tokens: list[int]
    pages: list[int] = field(default_factory=list)
    device_len: int = 0  # tokens with KV written


class EngineDriver:
    def __init__(self, model_path: str, *, max_running_req: int = 8, max_seq_len: int = 1024,
                 cuda_graph_bs=None, num_pages: int | None = None, max_extend_tokens: int = 2048,
                 memory_ratio: float = 0.85, use_dummy_weight: bool = True):
        from minisgl.distributed import DistributedInfo
        from minisgl.engine import Engine
        from minisgl.scheduler import SchedulerConfig

        cfg = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            max_running_req=max_running_req,
            cuda_graph_bs=cuda_graph_bs if cuda_graph_bs is not None else [],
            cuda_graph_max_bs=max(cuda_graph_bs) if cuda_graph_bs else 0,
            use_dummy_weight=use_dummy_weight,
            max_seq_len_override=max_seq_len,
            num_page_override=num_pages,
            max_extend_tokens=max_extend_tokens,
            memory_ratio=memory_ratio,
            cache_type="naive",
            offline_mode=True,
        )
        self.engine = Engine(cfg)
        self.cfg = cfg
        self.page_size = self.engine.ctx.page_size
        self.free_pages = list(range(self.engine.num_pages - 1, -1, -1))  # pop() = lowest page
        self.free_tables = list(range(max_running_req))
        self.reqs: dict[int, _ReqState] = {}
        self.device = self.engine.device

    # ---------------- allocation ----------------
    def add_request(self, uid: int, tokens: list[int]) -> None:
        t = self.free_tables.pop(0)
        self.reqs[uid] = _ReqState(table_idx=t, tokens=list(tokens))

    def _ensure_pages(self, st: _ReqState, upto: int) -> None:
        P = self.page_size
        need = (upto + P - 1) // P
        while len(st.pages) < need:
            pg = self.free_pages.pop()
            st.pages.append(pg)
            i = len(st.pages) - 1
            width = self.engine.page_table.shape[1]
            hi = min((i + 1) * P, width)
            vals = torch.arange(pg * P, pg * P + (hi - i * P), dtype=torch.int32, device=self.device)
            self.engine.page_table[st.table_idx, i * P : hi] = vals

    # ---------------- batches ----------------
    def _make_batch(self, items: list[tuple[int, int]], phase: str):
        """items: (uid, n_new_tokens) -> Batch with reqs / tensors filled."""
        from minisgl.core import Batch, Req, SamplingParams

        reqs, ids, pos, loc = [], [], [], []
        for uid, n in items:
            st = self.reqs[uid]
            L0 = st.device_len
            L1 = L0 + n
            self._ensure_pages(st, L1)
            r = Req(
                input_ids=torch.tensor(st.tokens[:L1], dtype=torch.int32),
                table_idx=st.table_idx, cached_len=L0, output_len=1, uid=uid,
                sampling_params=SamplingParams(), cache_handle=None,  # type: ignore[arg-type]
            )
            reqs.append(r)
            ids.extend(st.tokens[L0:L1])
            pos.extend(range(L0, L1))
        batch = Batch(reqs=reqs, phase=phase)
        batch.padded_reqs = list(reqs)
        dev = self.device
        batch.input_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        batch.positions = torch.tensor(pos, dtype=torch.int32, device=dev)
        rows = torch.tensor([self.reqs[u].table_idx for u, n in items for _ in range(n)], dtype=torch.long, device=dev)
        batch.out_loc = self.engine.page_table[rows, batch.positions.long()]
        return batch

    def forward(self, items: list[tuple[int, int]], phase: str, use_graph: bool = False) -> torch.Tensor:
        eng = self.engine
        batch = self._make_batch(items, phase)
        if use_graph and phase == "decode":
            eng.graph_runner.pad_batch(batch)
            if batch.padded_size > batch.size:
                # pad tensors for the graph bucket with the dummy request
                n = batch.padded_size - batch.size
                batch.input_ids = torch.cat([batch.input_ids, batch.input_ids.new_zeros(n)])
                batch.positions = torch.cat([batch.positions, batch.positions.new_zeros(n)])
                dummy_slot = int(eng.page_table[eng.dummy_req.table_idx, 0])
                batch.out_loc = torch.cat([batch.out_loc, batch.out_loc.new_full((n,), dummy_slot)])
        eng.attn_backend.prepare_metadata(batch)
        with eng.ctx.forward_batch(batch):
            if use_graph and phase == "decode" and eng.graph_runner.can_use_cuda_graph(batch):
                logits = eng.graph_runner.replay(batch)
            else:
                logits = eng.model.forward()
        for uid, n in items:
            self.reqs[uid].device_len += n
        return logits[: len(items)].float().clone()

    def prefill(self, uids: list[int], chunk: int | None = None) -> torch.Tensor:
        """Prefill all remaining prompt tokens; returns last-token logits per uid."""
        out = {}
        while True:
            items = []
            for u in uids:
                st = self.reqs[u]
                rem = len(st.tokens) - st.device_len
                if rem > 0:
                    items.append((u, min(rem, chunk) if chunk else rem))
            if not items:
                break
            logits = self.forward(items, "prefill")
            for i, (u, n) in enumerate(items):
                if self.reqs[u].device_len == len(self.reqs[u].tokens):
                    out[u] = logits[i]
        return torch.stack([out[u] for u in uids])

    def decode(self, uids: list[int], next_tokens: list[int], use_graph: bool = False) -> torch.Tensor:
        for u, t in zip(uids, next_tokens):
            self.reqs[u].tokens.append(int(t))
        return self.forward([(u, 1) for u in uids], "decode", use_graph=use_graph)


__all__ = ["EngineDriver", "make_reduced_model_dir"]
