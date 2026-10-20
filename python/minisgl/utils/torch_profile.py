"""Env-gated torch.profiler window over engine steps (colocated scheduler).

Spec ``<phase>:<first>:<count>[:<bs>,<bs>,...]``: starting with the ``first``-th step
of ``phase`` (counted per listed batch size when sizes are given, so one window is
recorded for every size of a sweep), profile ``count`` consecutive steps of that size.
Every profiled step is wrapped in ``record_function("<tag>_step[<ordinal>]")`` and
ends with a device synchronize, so all of a step's kernels (CUDA-graph replays
included) fall inside its annotation and can be attributed by timestamp.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import torch


class StepProfiler:
    def __init__(
        self,
        spec: str,
        out_dir: str,
        *,
        tag: str,
        device: torch.device,
        log: Callable[[str], None] | None = None,
    ):
        parts = spec.split(":")
        self.phase = parts[0] or "decode"
        self.first = int(parts[1]) if len(parts) > 1 and parts[1] else 20
        self.count = int(parts[2]) if len(parts) > 2 and parts[2] else 6
        sizes = parts[3] if len(parts) > 3 else ""
        self.sizes = [int(v) for v in sizes.split(",") if v.strip()]
        self.pending = set(self.sizes)
        self.out_dir = out_dir
        self.tag = tag
        self.device = device
        self.log = log or (lambda msg: None)
        self.seen: dict[int | None, int] = {}
        self.prof = None
        self.rf = None
        self.key: int | None = None
        self.steps: list[dict] = []
        self.done = False

    def begin(self, phase: object, size: int) -> None:
        if self.done or str(phase) != self.phase:
            return
        size = int(size)
        if self.prof is None:
            key = size if self.sizes else None
            if self.sizes and key not in self.pending:
                return
            n = self.seen.get(key, 0) + 1
            self.seen[key] = n
            if n != self.first:
                return
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize(self.device)
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.prof.__enter__()
            self.key, self.steps = key, []
            self.log(f"torch_profile:start {self.phase} size={size} ordinal={n} count={self.count}")
        elif self.key is not None and size != self.key:
            self._finish()  # the sweep moved on: close this window without the new step
            self.begin(phase, size)
            return
        from torch.profiler import record_function

        self.rf = record_function(f"{self.tag}_step[{len(self.steps)}]")
        self.rf.__enter__()
        self.steps.append({"ordinal": len(self.steps), "size": size})

    def end(self) -> None:
        if self.prof is None or self.rf is None:
            return
        torch.cuda.synchronize(self.device)
        self.rf.__exit__(None, None, None)
        self.rf = None
        if len(self.steps) >= self.count:
            self._finish()

    def _finish(self) -> None:
        if self.prof is None:
            return
        if self.rf is not None:
            self.rf.__exit__(None, None, None)
            self.rf = None
        torch.cuda.synchronize(self.device)
        self.prof.__exit__(None, None, None)
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, f"torch_profile_{self.tag}")
        if self.key is not None:
            base += f"_bs{self.key}"
        self.prof.export_chrome_trace(base + ".json")
        with open(base + "_meta.json", "w") as f:
            json.dump(
                {
                    "tag": self.tag,
                    "phase": self.phase,
                    "batch_size": self.key,
                    "steps": self.steps,
                    "key_averages": self.prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=60, max_name_column_width=100
                    ),
                },
                f,
            )
        self.log(f"torch_profile:done size={self.key} steps={len(self.steps)} trace={base}.json")
        self.prof = None
        if self.sizes:
            self.pending.discard(self.key)
            self.done = not self.pending
        else:
            self.done = True
        self.key = None
