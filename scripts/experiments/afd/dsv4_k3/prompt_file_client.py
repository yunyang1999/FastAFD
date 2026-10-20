"""Serve real text prompts (one per line of a file) through the batch endpoint.

  python prompt_file_client.py --url http://127.0.0.1:PORT --prompt-file prompts.txt \
      --tokenizer MODEL_DIR --num-prompts 1152 --prompt-len 2048 --max-tokens 16 --out out.json

Each prompt is cut to its first ``--prompt-len`` tokens of the served model's
tokenizer (tokenizer files only, no checkpoint code). Used for routing captures
(MINISGL_DUMP_EXPERT_LOAD) and real-weight sanity runs; sampling defaults to the
model's generation config (temperature 1, top_p 1).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def load_prompts(path: str, tokenizer_dir: str, n: int, length: int) -> tuple[list[str], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompts, lens = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line, add_special_tokens=False)[:length]
            prompts.append(tok.decode(ids))
            lens.append(len(ids))
            if len(prompts) == n:
                break
    return prompts, lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num-prompts", type=int, default=1152)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts, lens = load_prompts(args.prompt_file, args.tokenizer, args.num_prompts, args.prompt_len)
    payload = {"model": "m", "prompts": prompts, "max_tokens": args.max_tokens, "ignore_eos": True,
               "temperature": args.temperature, "top_p": args.top_p, "return_token_ids": True}
    t0 = time.time()
    req = urllib.request.Request(f"{args.url}/v1/chat/completions/batch", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        res = json.loads(resp.read())
    t1 = time.time()
    choices = res["choices"]
    out = {
        "num_prompts": len(prompts),
        "prompt_tokens": lens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "start": t0,
        "end": t1,
        "generated_tokens": sum(len(c.get("token_ids") or []) for c in choices),
        "samples": [{"prompt_tail": p[-200:], "completion": c["message"]["content"]} for p, c in zip(prompts[:8], choices[:8])],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("num_prompts", "generated_tokens", "start", "end")}))


if __name__ == "__main__":
    main()
