"""Refusal precision / recall / F1 on a paired run: the same turns answerable and with their gold documents
withheld (scripts/eval_answers.py --withhold gold-doc).

    python scripts/refusal_report.py runs/answers/md2d_validation_n300_s0.jsonl \
        runs/answers/md2d_validation_n300_s0_withheld.jsonl \
        [--covered runs/answers/md2d_validation_n300_s0_withheld.covered.json]

Two readings of "the system declined": strict, the final reply is NOT_FOUND; abstain, the final reply is
NOT_FOUND or the passage text shown because no answer passed the verifier. Each is scored on the first
answer (no verifier) and on the final output (after the verifier and revisions). With --covered
(scripts/judge_withheld.py), pairs whose withheld turn is still answerable from other documents are
dropped, so every withheld turn left is truly unanswerable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vpqa.eval.metrics import refusal_prf


def load(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answerable")
    ap.add_argument("withheld")
    ap.add_argument("--covered", type=Path)
    args = ap.parse_args()
    answerable, withheld = load(args.answerable), load(args.withheld)
    ids = sorted(answerable.keys() & withheld.keys())
    if args.covered:
        covered = json.loads(args.covered.read_text())
        ids = [i for i in ids if not covered[i]]
    rows = [(answerable[i], False) for i in ids] + [(withheld[i], True) for i in ids]
    unanswerable = [u for _, u in rows]
    readings = {
        "first answer, strict": [r["refused"] for r, _ in rows],
        "final, strict": [r["outcome"] == "refused" for r, _ in rows],
        "final, abstain": [r["outcome"] != "answered" for r, _ in rows],
    }
    print(f"{len(ids)} pairs")
    print(f"{'':24} {'P':>6} {'R':>6} {'F1':>6}")
    for name, pred in readings.items():
        p, r, f = refusal_prf(pred, unanswerable)
        print(f"{name:24} {p:6.3f} {r:6.3f} {f:6.3f}")


if __name__ == "__main__":
    main()
