import json
import sys
import time
from pathlib import Path

import httpx

def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    n = len(values)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

def main(path: str, url: str = "http://localhost:8000/ask"):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    client = httpx.Client(timeout=30.0)
    hits = 0
    latencies = []
    tokens_total = 0
    for line in lines:
        obj = json.loads(line)
        q = obj["q"]
        expected = obj["expected_doc"]
        t0 = time.perf_counter()
        resp = client.post(url, json={"question": q})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        resp.raise_for_status()
        data = resp.json()
        top = data["sources"][0]["id"] if data["sources"] else None
        if top == expected:
            hits += 1
        tokens_total += data["usage"]["total_tokens"]
    n = len(lines)
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    print(json.dumps({
        "count": n,
        "hit_rate": hits/n if n else 0.0,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "tokens_total": tokens_total
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_client.py path/to/queries.jsonl")
        sys.exit(1)
    main(sys.argv[1])
