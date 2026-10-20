"""Batch-sweep client for AFD / colocated step-time measurements.

For each per-GPU batch size ``b`` it submits ``b * --attn-gpus`` requests of
``--prompt-len`` tokens in one batch call (ignore_eos, fixed ``--max-tokens``),
waits for completion and writes a JSON log of the runs (wall-clock marks are
used to attribute server-side steps to batch sizes).

Prompts are synthetic by default; with ``--prompt-file`` (one text per line)
every request gets a distinct real-text prompt of exactly ``--prompt-len``
tokens (consecutive lines are concatenated when one line is too short).

  python bench_client.py --url http://127.0.0.1:PORT --tokenizer DIR \
      --attn-gpus 4 --batch-sizes 8,16,32 --prompt-len 1024 --max-tokens 64 --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def build_prompt(tokenizer_dir: str, target: int) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)  # no checkpoint remote code
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


def real_prompt_ids(tokenizer_dir: str, prompt_file: str, n: int, length: int) -> list[list[int]]:
    """``n`` distinct token-id prompts of exactly ``length`` tokens from the file:
    prompt i starts at line i (mod #lines) and continues into the following lines."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    lines = [ln.strip() for ln in open(prompt_file, encoding="utf-8") if ln.strip()]
    if not lines:
        raise SystemExit(f"{prompt_file}: no prompts")
    ids = [tok.encode(ln, add_special_tokens=False) for ln in lines]
    out = []
    for i in range(n):
        cur: list[int] = []
        j = i
        while len(cur) < length:
            cur.extend(ids[j % len(ids)])
            j += 1
        out.append(cur[:length])
    return out


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--attn-gpus", "--attn-dp", dest="attn_gpus", type=int, required=True,
                    help="GPUs the batch is spread over (requests per run = b * attn_gpus)")
    ap.add_argument("--batch-sizes", required=True)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-file", default="", help="real text prompts, one per line (distinct per request)")
    ap.add_argument("--timeout", type=float, default=7200)
    # sampled decoding keeps dummy-weight models from collapsing onto one token
    # (which would also collapse MoE routing, unlike real traffic)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bss = [int(x) for x in args.batch_sizes.split(",") if x]
    log = {"prompt_len": args.prompt_len, "max_tokens": args.max_tokens, "attn_dp": args.attn_gpus,
           "attn_gpus": args.attn_gpus, "prompt_file": args.prompt_file, "temperature": args.temperature, "runs": []}
    if args.prompt_file:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        n_max = max(bss) * args.attn_gpus
        texts = [tok.decode(p) for p in real_prompt_ids(args.tokenizer, args.prompt_file, n_max, args.prompt_len)]
    else:
        texts = None
        prompt = build_prompt(args.tokenizer, args.prompt_len)
    for bs in bss:
        n = bs * args.attn_gpus
        prompts = texts[:n] if texts is not None else [prompt] * n
        t0 = time.time()
        res = post(
            f"{args.url}/v1/chat/completions/batch",
            {"model": "m", "prompts": prompts, "max_tokens": args.max_tokens, "ignore_eos": True,
             "temperature": args.temperature, "top_k": 1 if args.temperature == 0 else -1,
             "return_token_ids": True},
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
