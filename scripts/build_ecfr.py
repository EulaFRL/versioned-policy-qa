"""Fetch every dated snapshot of the eCFR parts in `vpqa.data.ecfr.PARTS` and merge them into versioned passages.

Snapshots are cached under data/raw/ecfr/; the result is data/processed/ecfr_versions.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vpqa.data.ecfr import PARTS, build_versions, fetch_part, list_versions, parse_part, save_versions, snapshot_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(PARTS))
    ap.add_argument("--out", type=Path, default=Path("data/processed/ecfr_versions.jsonl"))
    args = ap.parse_args()

    passages = []
    for domain in args.domains:
        for title, part in PARTS[domain]:
            dates = snapshot_dates(list_versions(title, part))
            snapshots = []
            for on in dates:
                xml = fetch_part(title, part, on)
                if xml is None:
                    continue
                snapshots.append((on, parse_part(xml)))
                print(f"{title} CFR {part} {on}: {len(snapshots[-1][1])} passages", flush=True)
            versions = build_versions(snapshots, title, part, domain)
            per_key = Counter(p.pid.split("@")[0] for p in versions)
            print(f"== {title} CFR {part}: {len(snapshots)}/{len(dates)} snapshots ({snapshots[0][0]} .. {snapshots[-1][0]}), "
                  f"{len(per_key)} passage keys, {len(versions)} versions, "
                  f"{sum(n > 1 for n in per_key.values())} keys with >1 version", flush=True)
            passages += versions
    save_versions(passages, args.out)
    print(f"wrote {len(passages)} versions to {args.out}")


if __name__ == "__main__":
    main()
