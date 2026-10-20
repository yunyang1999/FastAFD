"""Greedy completions of a fixed prompt set (AFD vs colocated comparisons).

  python greedy_tokens.py run --url http://127.0.0.1:PORT --max-tokens 16 --out afd.json
  python greedy_tokens.py compare colocated.json afd.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1905, Albert Einstein published",
    "The quick brown fox jumps over",
    "Water boils at a temperature of",
    "Translate to German: good morning",
    "1, 1, 2, 3, 5, 8,",
    "The largest planet in the solar system is",
]


def run(args) -> None:
    payload = {"model": "m", "prompts": PROMPTS, "max_tokens": args.max_tokens, "ignore_eos": True,
               "temperature": 0.0, "top_k": 1, "return_token_ids": True}
    req = urllib.request.Request(f"{args.url}/v1/chat/completions/batch", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        res = json.loads(resp.read())
    ids = [c.get("token_ids") or [] for c in res["choices"]]
    texts = [(c.get("message") or {}).get("content", "") for c in res["choices"]]
    json.dump({"prompts": PROMPTS, "token_ids": ids, "texts": texts}, open(args.out, "w"), indent=1)
    print(json.dumps({"out": args.out, "lens": [len(x) for x in ids]}))


def compare(args) -> None:
    a = json.load(open(args.a))["token_ids"]
    b = json.load(open(args.b))["token_ids"]
    prefix, same, total = [], 0, 0
    for x, y in zip(a, b):
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        prefix.append(n)
        same += sum(int(p == q) for p, q in zip(x, y))
        total += max(len(x), len(y))
    print(json.dumps({"matching_prefix_per_prompt": prefix, "position_agreement": same / max(total, 1),
                      "tokens_per_prompt": [len(x) for x in a]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
