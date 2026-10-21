"""Replay one run from a trace file and compare it node by node with what was recorded.

    python scripts/replay_trace.py runs/answers/md2d_validation_n300_s0_withheld.trace.jsonl --list
    python scripts/replay_trace.py runs/answers/md2d_validation_n300_s0_withheld.trace.jsonl <run_id>

The `<given>` record of a trace holds the run's inputs in full; the run goes through the same pipeline
again (without stage timeouts, so a node that timed out in the recorded run shows what it would have
returned) and every node's outputs are compared with the recorded ones. LLM replies come from the response
cache when the prompt is unchanged, so a replay costs nothing and reproduces the answer exactly; a node
whose outputs differ marks where the code, the index, or a prompt changed since the run.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from vpqa.data.ecfr import load_versions
from vpqa.data.multidoc2dial import load_passages
from vpqa.pipeline.dag import Pipeline
from vpqa.schema import Turn


def records(path: Path, run_id: str | None = None) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if run_id is None or r["run_id"] == run_id:
            runs.setdefault(r["run_id"], []).append(r)
    return runs


def turn_from(given: dict) -> Turn:
    t = dict(given["turn"])
    t["as_of"] = date.fromisoformat(t["as_of"]) if t.get("as_of") else None
    return Turn(**t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    args = ap.parse_args()
    if args.list or not args.run_id:
        for rid, recs in records(args.trace).items():
            status = {r["node"]: r["status"] for r in recs if r["status"] not in ("ok", "skipped")}
            print(rid, status or "")
        return

    [recorded] = records(args.trace, args.run_id).values()
    given = recorded[0]
    assert given["node"] == "<given>" and given.get("full"), "trace written before full inputs were recorded"
    turn = turn_from(given["outputs"])
    cfg = yaml.safe_load(open(args.config))
    generate = any(r["node"] == "answer" for r in recorded)
    if turn.as_of is None:
        pipe = Pipeline(load_passages(Path("data/raw/multidoc2dial/multidoc2dial_doc.json")), cfg,
                        Path("data/index/BAAI__bge-m3.npy"), args.router, timeouts=False, generate=generate)
    else:
        pipe = Pipeline(load_versions(), cfg, Path("data/index/BAAI__bge-m3__ecfr.npy"), timeouts=False,
                        generate=generate)
    _, trace = pipe.run(turn)
    replayed = {r["node"]: r for r in trace.records}
    norm = lambda v: json.loads(json.dumps(v, default=str))
    print(f"{args.run_id}: {turn.query}\n")
    print(f"{'node':16} {'recorded':10} {'replayed':10} {'ms then':>9} {'ms now':>9}  outputs")
    for r in recorded[1:]:
        now = replayed.get(r["node"], {})
        same = norm(now.get("outputs")) == norm(r.get("outputs")) if now else False
        print(f"{r['node']:16} {r['status']:10} {now.get('status', '-'):10} {r['ms']:9.1f} {now.get('ms', 0):9.1f}  "
              f"{'same' if same else 'DIFFERENT'}")
    final = [r for r in recorded if r["node"] == "finalize"]
    if final:
        print("\nfinal:", final[0]["outputs"]["final"])


if __name__ == "__main__":
    main()
