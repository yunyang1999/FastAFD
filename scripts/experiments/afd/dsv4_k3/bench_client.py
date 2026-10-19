"""Batch-sweep client for AFD step-time measurements.

For each per-attention-GPU batch size ``b`` it submits ``b * attn_dp`` requests
of ``--prompt-len`` tokens in one batch call (ignore_eos, fixed ``--max-tokens``),
waits for completion and writes a JSON log of the phases (wall-clock marks are
used to attribute coordinator steps to batch sizes).

  python bench_client.py --url http://127.0.0.1:PORT --tokenizer DIR \
      --attn-dp 4 --batch-sizes 8,16,32 --prompt-len 1024 --max-tokens 64 --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def build_prompt(tokenizer_dir: str, target: int) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    unit = " hello"
    lo, hi = 1, max(target * 2, 16)
    while len(tok.encode((unit * hi).strip())) <= target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.encode((unit * mid).strip())) <= target:
            lo = mid
        else:
            hi = mid - 1
    return (unit * lo).strip()


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--attn-dp", type=int, required=True)
    ap.add_argument("--batch-sizes", required=True)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompt = build_prompt(args.tokenizer, args.prompt_len)
    log = {"prompt_len": args.prompt_len, "max_tokens": args.max_tokens, "attn_dp": args.attn_dp, "runs": []}
    for bs in [int(x) for x in args.batch_sizes.split(",") if x]:
        n = bs * args.attn_dp
        t0 = time.time()
        res = post(
            f"{args.url}/v1/chat/completions/batch",
            {"model": "m", "prompts": [prompt] * n, "max_tokens": args.max_tokens, "ignore_eos": True,
             "temperature": 0.0, "top_k": 1, "return_token_ids": True},
            args.timeout,
        )
        t1 = time.time()
        ntok = sum(len(c.get("token_ids") or []) for c in res.get("choices", []))
        run = {"per_attn_gpu_bs": bs, "num_requests": n, "start": t0, "end": t1, "generated_tokens": ntok}
        log["runs"].append(run)
        print(json.dumps(run), flush=True)
        time.sleep(2.0)
    with open(args.out, "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
