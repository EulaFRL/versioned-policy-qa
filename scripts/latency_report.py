"""Tail latency of the answer stage across runs of the same questions (scripts/eval_answers.py traces).

    python scripts/latency_report.py runs/latency/base runs/latency/hedge runs/latency/sort runs/latency/both

Each argument is a run directory (or a `*_s0` prefix); its trace gives every node's time per question.
For each run: answer-node p50 / p90 / p95 / max with a bootstrap 95% interval on p95, answer timeouts, the
end-to-end time (all nodes), the final outcomes, hedged requests and cost per question. Then, against the
first run, the paired difference in answer time per question (median and bootstrap interval), since a
p95 over 100 questions moves a lot from one sample to the next.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def files(arg: str) -> tuple[Path, Path]:
    p = Path(arg)
    trace = next(p.glob("*.trace.jsonl")) if p.is_dir() else Path(f"{arg}.trace.jsonl")
    return trace, trace.with_name(trace.name.replace(".trace.jsonl", ".summary.json"))


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]


def boot(stat, xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def load(arg: str) -> tuple[dict, dict, dict]:
    trace, summary = files(arg)
    answer, total, status = {}, defaultdict(float), {}
    for r in map(json.loads, trace.read_text().splitlines()):
        total[r["run_id"]] += r.get("ms") or 0
        if r["node"] == "answer":
            answer[r["run_id"]], status[r["run_id"]] = r["ms"], r["status"]
    return answer, dict(total), {"status": status, "summary": json.loads(summary.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    runs = {a.rstrip("/").split("/")[-1]: load(a) for a in args.runs}
    print(f"{'run':10} {'n':>4} {'p50':>6} {'p90':>6} {'p95':>6} {'p95 95% CI':>15} {'max':>6} {'t/o':>4} "
          f"{'e2e p50':>8} {'e2e p95':>8} {'answered':>8} {'degraded':>8} {'fallbk':>6} {'hedged':>6} {'$/q':>7}")
    for name, (answer, total, info) in runs.items():
        a, t, s = [x / 1000 for x in answer.values()], [x / 1000 for x in total.values()], info["summary"]
        lo, hi = boot(lambda v: quantile(v, 0.95), a)
        timeouts = sum(v == "timeout" for v in info["status"].values())
        print(f"{name:10} {len(a):4d} {quantile(a, .5):6.1f} {quantile(a, .9):6.1f} {quantile(a, .95):6.1f} "
              f"{f'[{lo:.1f}, {hi:.1f}]':>15} {max(a):6.1f} {timeouts:4d} {quantile(t, .5):8.1f} {quantile(t, .95):8.1f} "
              f"{s['final']['answered']:8.2f} {s['final']['degraded']:8.2f} {s['fallbacks']:6.2f} "
              f"{s['llm'].get('hedged', 0):6d} {s['cost_per_item']:7.4f}")
    first, (base, _, _) = next(iter(runs.items()))
    print(f"\npaired answer-time difference against {first} (seconds, negative = faster)")
    for name, (answer, _, _) in list(runs.items())[1:]:
        ids = sorted(base.keys() & answer.keys())
        d = [(answer[i] - base[i]) / 1000 for i in ids]
        lo, hi = boot(lambda v: quantile(v, 0.5), d)
        slow = [i for i in ids if base[i] > quantile(list(base.values()), 0.9)]  # the base run's slowest tenth
        print(f"{name:10} median {quantile(d, .5):+6.2f} [{lo:+.2f}, {hi:+.2f}]   "
              f"on the base run's slowest 10%: {sum(answer[i] for i in slow) / max(len(slow), 1) / 1000:5.1f} s "
              f"vs {sum(base[i] for i in slow) / max(len(slow), 1) / 1000:5.1f} s")


if __name__ == "__main__":
    main()
