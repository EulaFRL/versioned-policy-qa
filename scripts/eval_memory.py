"""Session memory and LLM query rewriting on MultiDoc2Dial retrieval.

    python scripts/eval_memory.py --split train --dialogues 200 --seed 0      # pick the settings
    python scripts/eval_memory.py --split validation                        # all validation dialogues

Each dialogue is replayed turn by turn through the retrieval half of the pipeline (query, route, corpus
filter, retrieve, focus, rerank). After each turn the session records the document of the top-ranked
passage, standing in for the passages the reply would cite, and the next turn's focus is the documents
of the last `replies` replies. With `--source oracle` the focus is the previous turn's gold document
instead: an upper bound, a session that always remembers the right document.

Query: `off` is the length-gated query; `short` has the LLM (VPQA_MODEL) rewrite the follow-ups the gate
gives history, `all` every follow-up. Session settings: `k` focus passages added to the rerank candidates
and a `bonus` added to the rerank score of every passage of a focus document.

Every rerank score is computed once and kept in runs/memory/, so the grid replays the dialogues without
the cross-encoder. Recall is over the answerable turns, by group: first turns, follow-ups whose gold
document is the previous turn's, and follow-ups that switch document.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from vpqa import llm
from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import mrr, recall_at_k
from vpqa.memory import Session
from vpqa.pipeline.dag import Pipeline
from vpqa.retrieval.query import needs_context

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"


class ScoreMemo:
    """Reranker stand-in: scores pairs it has not seen with the cross-encoder and remembers every score."""

    def __init__(self, reranker, path: Path):
        self.reranker, self.path = reranker, path
        self.cache: dict[str, dict[str, float]] = json.loads(path.read_text()) if path.exists() else {}

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        missing = list(dict.fromkeys((q, p) for q, p in pairs if p not in self.cache.get(q, {})))
        if missing:
            for (q, p), s in zip(missing, self.reranker.score_many(missing)):
                self.cache.setdefault(q, {})[p] = s
        return [self.cache[q][p] for q, p in pairs]

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--dialogues", type=int, default=0, help="sample this many dialogues (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rewrite", nargs="+", default=["off", "short", "all"])
    ap.add_argument("--source", nargs="+", default=["system", "oracle"])
    ap.add_argument("--replies", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--k", nargs="+", type=int, default=[0, 10])
    ap.add_argument("--bonus", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--workers", type=int, default=8, help="rewrite calls at once")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--out", type=Path, default=Path("runs/memory"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    turns = load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", DOC)
    dialogues: dict[str, list] = defaultdict(list)
    for t in turns:
        dialogues[t.dialog_id].append(t)
    ids = sorted(dialogues)
    if args.dialogues:
        ids = sorted(random.Random(args.seed).sample(ids, args.dialogues))
    dialogues = {d: sorted(dialogues[d], key=lambda t: t.turn_id) for d in ids}
    turns = [t for d in ids for t in dialogues[d]]
    index = {(t.dialog_id, t.turn_id): i for i, t in enumerate(turns)}
    print(f"{len(ids)} dialogues, {len(turns)} turns, {sum(t.answerable for t in turns)} answerable", flush=True)

    cfg = yaml.safe_load(open(args.config))
    pipe = Pipeline(load_passages(DOC), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, timeouts=False)
    memo = ScoreMemo(pipe.reranker, args.out / f"rerank_{args.split}.json")
    pipe.reranker = memo
    doc = lambda pid: pipe.by_pid[pid].doc_id

    # queries per rewrite mode; LLM calls run in threads and are timed (only meaningful when not yet cached)
    queries, rewrite_ms = {}, []
    for mode in args.rewrite:
        def one(t, mode=mode):
            t0 = time.perf_counter()
            q = pipe.query(t, mode)["query"]
            if mode != "off" and t.history and (mode == "all" or needs_context(t.query, cfg["query"]["short_words"])):
                rewrite_ms.append((time.perf_counter() - t0) * 1000)
            return q
        with ThreadPoolExecutor(args.workers if mode != "off" else 1) as pool:
            queries[mode] = list(pool.map(one, turns))
        print(f"queries {mode}: cost so far {llm.TOTALS['cost']:.3f}", flush=True)

    # route, filter and candidates do not depend on the session
    base = {}
    for mode in args.rewrite:
        rows = []
        for t, q in zip(turns, queries[mode]):
            domain, valid = pipe.route(t)["domain"], pipe.corpus_filter(t)["valid"]
            rows.append((valid, pipe.retrieve(q, domain, valid)["candidates"]))
        base[mode] = rows
        memo.score_many([(q, p) for q, (_, c) in zip(queries[mode], rows) for p in c])
        memo.save()
        print(f"candidates scored for {mode}", flush=True)

    def replay(mode: str, source: str, replies: int, k: int, bonus: float) -> list[list[str]]:
        pipe.focus_cfg = {"k": k, "bonus": bonus} if source != "none" else None
        ranked = [None] * len(turns)
        for d in ids:
            session, previous_gold = Session(), []
            for t in dialogues[d]:
                i = index[(t.dialog_id, t.turn_id)]
                focus = session.focus(replies) if source == "system" else previous_gold if source == "oracle" else []
                turn = replace(t, focus_docs=focus)
                valid, candidates = base[mode][i]
                focused = pipe.focus(queries[mode][i], turn, valid)
                ranked[i] = pipe.rerank(queries[mode][i], candidates, **focused)["ranked"]
                if ranked[i]:
                    session.record([doc(ranked[i][0])])
                if t.gold_pids:
                    previous_gold = list(dict.fromkeys(doc(g) for g in t.gold_pids))
        return ranked

    groups = {}
    for d in ids:
        prev = None
        for t in dialogues[d]:
            if not t.gold_pids:
                continue
            gold_docs = {doc(g) for g in t.gold_pids}
            if t.answerable:
                groups[index[(t.dialog_id, t.turn_id)]] = ("first" if not t.history else "same document"
                                                           if prev and gold_docs & prev else "switch")
            prev = gold_docs

    def score(ranked: list[list[str]]) -> dict:
        out = {}
        for g in ("all", "first", "same document", "switch"):
            sel = [i for i, x in groups.items() if g == "all" or x == g]
            out[g] = {"n": len(sel),
                      "R@1": np.mean([recall_at_k(ranked[i], turns[i].gold_pids, 1) for i in sel]),
                      "R@5": np.mean([recall_at_k(ranked[i], turns[i].gold_pids, 5) for i in sel]),
                      "MRR@20": np.mean([mrr(ranked[i][:20], turns[i].gold_pids) for i in sel])}
        return out

    results = []
    for mode in args.rewrite:
        results.append({"rewrite": mode, "source": "none", "replies": 0, "k": 0, "bonus": 0.0,
                        **score(replay(mode, "none", 0, 0, 0.0))})
        for source, replies, k, bonus in itertools.product(args.source, args.replies, args.k, args.bonus):
            if k == 0 and bonus == 0:
                continue
            results.append({"rewrite": mode, "source": source, "replies": replies, "k": k, "bonus": bonus,
                            **score(replay(mode, source, replies, k, bonus))})
        memo.save()
        print(f"replayed {mode}", flush=True)

    name = f"{args.split}" + (f"_d{args.dialogues}_s{args.seed}" if args.dialogues else "")
    summary = {"dialogues": len(ids), "turns": len(turns), "answerable": len(groups),
               "groups": {g: sum(x == g for x in groups.values()) for g in ("first", "same document", "switch")},
               "rewrite_ms_p50": float(np.percentile(rewrite_ms, 50)) if rewrite_ms else None,
               "rewrite_ms_p95": float(np.percentile(rewrite_ms, 95)) if rewrite_ms else None,
               "rewrite_calls": len(rewrite_ms), "llm": llm.TOTALS, "results": results}
    (args.out / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))
    rows = sorted(results, key=lambda r: -r["all"]["R@5"])
    print(f"\n| rewrite | source | replies | k | bonus | R@1 | R@5 | MRR@20 | first R@5 | same doc R@5 | switch R@5 |")
    print("|---" * 11 + "|")
    for r in [x for x in results if x["source"] == "none"] + rows[:25]:
        print(f"| {r['rewrite']} | {r['source']} | {r['replies']} | {r['k']} | {r['bonus']} | {r['all']['R@1']:.3f} | "
              f"{r['all']['R@5']:.3f} | {r['all']['MRR@20']:.3f} | {r['first']['R@5']:.3f} | "
              f"{r['same document']['R@5']:.3f} | {r['switch']['R@5']:.3f} |")
    print(f"groups {summary['groups']}; rewrite p50/p95 ms {summary['rewrite_ms_p50']} / {summary['rewrite_ms_p95']}; "
          f"cost {llm.TOTALS['cost']:.3f}")


if __name__ == "__main__":
    main()
