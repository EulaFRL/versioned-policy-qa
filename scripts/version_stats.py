"""How often the eCFR parts in the corpus actually change (data/processed/ecfr_versions.jsonl).

    python scripts/version_stats.py      # writes runs/version_stats.md

Per domain: paragraphs, the share with more than one version, the dates on which some paragraph changed,
and what a system that indexed the text once would get wrong today: the share of paragraphs whose text in
force today differs from the text in force N years ago (N = 1, 3, 5), and for a random as-of date the share
of paragraphs whose text in force then is no longer the text in force today.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from vpqa.data.ecfr import load_versions


def in_force(v, on: date) -> bool:
    return v.effective_from <= on and (v.effective_to is None or on < v.effective_to)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=Path, default=Path("data/processed/ecfr_versions.jsonl"))
    ap.add_argument("--today", type=date.fromisoformat, default=None, help="default: the latest effective date")
    ap.add_argument("--out", type=Path, default=Path("runs/version_stats.md"))
    args = ap.parse_args()
    versions = load_versions(args.versions)
    today = args.today or max(v.effective_from for v in versions)
    start = min(v.effective_from for v in versions)
    by_key = defaultdict(list)
    for v in versions:
        by_key[(v.domain, v.pid.split("@")[0])].append(v)

    def text_on(vs, on):
        return next((v.text for v in vs if in_force(v, on)), None)

    rng = random.Random(0)
    lines = [f"eCFR versions {start} to {today} ({args.versions}).", "",
             "| Domain | Paragraphs | Versions | >1 version | Change dates | Changed vs 1 y ago | vs 3 y | vs 5 y "
             "| Stale at random as-of date |", "|---|---|---|---|---|---|---|---|---|"]
    domains = sorted({d for d, _ in by_key}) + ["all"]
    for dom in domains:
        keys = [k for k in by_key if dom in ("all", k[0])]
        groups = [sorted(by_key[k], key=lambda v: v.effective_from) for k in keys]
        multi = sum(len(g) > 1 for g in groups)
        change_dates = {v.effective_from for g in groups for v in g[1:]}
        cells = []
        for years in (1, 3, 5):
            then = today - timedelta(days=365 * years)
            exist = [g for g in groups if text_on(g, then) is not None and text_on(g, today) is not None]
            cells.append(f"{sum(text_on(g, then) != text_on(g, today) for g in exist) / max(1, len(exist)):.3f}")
        stale = []
        for _ in range(2000):
            g = rng.choice(groups)
            on = start + timedelta(days=rng.randrange((today - start).days))
            old, now = text_on(g, on), text_on(g, today)
            if old is not None:
                stale.append(old != now)
        lines.append(f"| {dom} | {len(groups)} | {sum(map(len, groups))} | {multi / len(groups):.3f} | {len(change_dates)} | "
                     + " | ".join(cells) + f" | {sum(stale) / len(stale):.3f} |")
    report = "\n".join(lines)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
