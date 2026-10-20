"""Pick the `gated` word threshold on a split by RRF Recall@5.

Each turn is retrieved twice (current utterance alone, and with recent history); a threshold only
decides which of the two rankings a turn uses, so the sweep needs no extra retrieval.

    python scripts/select_gate.py --split train --sample 4000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import recall_at_k
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query

RAW = Path("data/raw/multidoc2dial")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--sample", type=int, default=4000, help="random answerable turns to use (0 = all)")
    ap.add_argument("--thresholds", type=int, nargs="+", default=list(range(3, 11)))
    ap.add_argument("--out", default="runs/select_gate")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rcfg, qcfg = cfg["retrieval"], cfg["query"]

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc_path) if t.answerable]
    if args.sample and args.sample < len(turns):
        turns = random.Random(0).sample(turns, args.sample)
    print(f"{len(turns)} answerable turns ({args.split})")

    bm25 = BM25Index(passages)
    name = rcfg["dense_model"]
    dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}.npy")

    def retrieve(mode: str) -> list[list[str]]:
        queries = [build_query(t, mode, qcfg["history_turns"], qcfg["max_words"]) for t in turns]
        dense_runs = dense.search_many(queries, rcfg["dense_k"])
        return [rrf([bm25.search(q, rcfg["bm25_k"]), d], k=rcfg["rrf_k"]) for q, d in zip(queries, dense_runs)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{args.split}_rankings_n{len(turns)}_h{qcfg['history_turns']}_w{qcfg['max_words']}.json"
    if cache.exists():
        alone, with_hist = json.loads(cache.read_text())
    else:
        alone, with_hist = retrieve("last"), retrieve("recent")
        cache.write_text(json.dumps([alone, with_hist]))
    rows = {}
    for w in args.thresholds:
        r5 = [recall_at_k(h if t.history and len(t.query.split()) <= w else a, t.gold_pids, 5)
              for t, a, h in zip(turns, alone, with_hist)]
        rows[w] = sum(r5) / len(r5)
    rows["none"] = sum(recall_at_k(a, t.gold_pids, 5) for t, a in zip(turns, alone)) / len(turns)

    (out / f"{args.split}.json").write_text(json.dumps({str(k): v for k, v in rows.items()}, indent=2))
    print("\n| short_words | RRF R@5 |\n|---|---|")
    for k, v in rows.items():
        print(f"| {k} | {v:.3f} |")
    best = max(args.thresholds, key=rows.get)
    print(f"\nbest short_words on {args.split}: {best}")


if __name__ == "__main__":
    main()
