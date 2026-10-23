"""Fixed pipeline: query -> route -> corpus filter -> retrieve -> focus -> rerank, then (vpqa.pipeline.generate)
answer -> verify -> at most two revisions -> finalize.

Every stage is a node of vpqa.pipeline.engine.Graph with declared input and output types, so the order
is fixed in code and each run leaves a per-node trace. Stages with a time budget (configs `harness`) degrade
instead of failing: query rewrite -> the length-gated query, route -> search all domains, retrieve -> BM25 only, rerank -> keep the RRF order,
answer or verify -> show the passage text instead of an unchecked answer. The
corpus filter has no fallback: searching every version would silently return stale text.

Session memory (configs `memory.focus`, vpqa.memory.Session) adds the best passages of the documents the
previous reply cited to the rerank candidates and gives that document a fixed bonus on the rerank score.

Model calls hold one lock: a timed-out stage keeps running in its thread, and two threads using the GPU at
once crash Metal (MPS). A stage that waits for the lock spends its own budget and falls back.
"""
from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vpqa import llm
from vpqa.pipeline.generate import Generator
from vpqa.pipeline.engine import Graph, Node, Trace
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query, needs_context, rewrite_query
from vpqa.retrieval.rerank import Reranker
from vpqa.retrieval.route import routing_text
from vpqa.schema import Passage, Turn


@dataclass
class Result:
    answer: str | None
    cited_pids: list[str]
    refused: bool
    trace: dict


class Pipeline:
    def __init__(self, passages: list[Passage], cfg: dict, dense_cache: Path | None = None,
                 router: Path | None = None, rerank: bool = True, timeouts: bool = True, generate: bool = False,
                 version_filter: bool = True):
        qcfg, self.rcfg = cfg["query"], cfg["retrieval"]
        self.qcfg = qcfg
        self.by_pid = {p.pid: p for p in passages}
        self.bm25 = BM25Index(passages)
        self.dense = DenseIndex(passages, self.rcfg["dense_model"], cache=dense_cache)
        self.reranker = Reranker(passages, cfg["rerank"]["model"]) if rerank else None
        self.reranker_pid = passages[0].pid
        self.model_lock = threading.Lock()
        self.router = None
        route_cfg = cfg.get("route", {})
        if router is not None and route_cfg.get("enabled", True):
            with open(router, "rb") as f:
                self.router = pickle.load(f)
            self.router["threshold"] = route_cfg.get("threshold", self.router["threshold"])
        # validity windows as day ordinals; no end date means still in force
        self.start = np.array([p.effective_from.toordinal() if p.effective_from else 0 for p in passages])
        self.end = np.array([p.effective_to.toordinal() if p.effective_to else np.iinfo(np.int64).max
                             for p in passages])
        self.versioned = version_filter and any(p.effective_from for p in passages)  # False: ablation, all versions
        self.row = {p.pid: i for i, p in enumerate(passages)}
        self.departments = sorted({p.domain for p in passages})
        self.doc_of = np.array([p.doc_id for p in passages])
        focus = cfg.get("memory", {}).get("focus", {})
        self.focus_cfg = focus if focus.get("enabled") else None
        fan_out = cfg.get("route", {}).get("fan_out", {})
        self.fan_out = fan_out if fan_out.get("enabled") else None  # per-department search when no route
        self.pool = ThreadPoolExecutor(len(self.departments)) if self.fan_out else None
        ms = cfg.get("harness", {}).get("timeouts_ms", {}) if timeouts else {}
        limit = lambda name: ms[name] / 1000 if name in ms else None
        nodes = [
            Node("query", self.query, {"turn": Turn}, {"query": str}, limit("rewrite"), fallback=self.gated_query),
            Node("route", self.route, {"turn": Turn}, {"domain": str | None},
                 limit("route"), fallback=lambda turn: {"domain": None}),
            Node("corpus_filter", self.corpus_filter, {"turn": Turn}, {"valid": np.ndarray | None}),
            Node("retrieve", self.retrieve, {"query": str, "domain": str | None, "valid": np.ndarray | None},
                 {"bm25": list[str], "dense": list[str], "candidates": list[str], "skipped": list[str]},
                 limit("retrieve"), fallback=self.retrieve_bm25),
            Node("focus", self.focus, {"query": str, "turn": Turn, "valid": np.ndarray | None},
                 {"focus_docs": list[str], "focus_pids": list[str]}),
            Node("rerank", self.rerank, {"query": str, "candidates": list[str], "focus_docs": list[str],
                                         "focus_pids": list[str]},
                 {"ranked": list[str], "top_score": float | None},
                 limit("rerank"), fallback=lambda query, candidates, **_: {"ranked": candidates, "top_score": None}),
        ]
        if generate:
            a = cfg.get("answer", {})
            self.generator = Generator(self.by_pid, llm.chat, llm.judge_json, self.rcfg["top_n"],
                                       a.get("max_revisions", 2), a.get("history_turns", 4), a.get("passage_words", 400),
                                       a.get("min_score"), a.get("check_question", False), a.get("rules"))
            nodes += self.generator.nodes(limit)
        self.graph = Graph(nodes, given={"turn": Turn})
        self.warm_up()

    def warm_up(self) -> None:
        """The first call to each model is slow (weights moved to the device); keep it out of the budgets."""
        self.dense.encode(["warm up"])
        if self.reranker is not None:
            self.reranker.score_many([("warm up", self.reranker_pid)])

    def query(self, turn: Turn, rewrite: str | None = None) -> dict:
        """The length-gated query, or an LLM rewrite (configs `query.rewrite`): `short` rewrites the follow-ups
        the gate would give history, `all` every follow-up. A first turn is used as it is."""
        q = self.qcfg
        rewrite = rewrite or q.get("rewrite", "off")
        if rewrite == "off" or not turn.history or (rewrite == "short" and not needs_context(turn.query, q["short_words"])):
            return self.gated_query(turn)
        return {"query": rewrite_query(turn, llm.chat, q["history_turns"]) or self.gated_query(turn)["query"]}

    def gated_query(self, turn: Turn) -> dict:
        q = self.qcfg
        return {"query": build_query(turn, "gated", q["history_turns"], q["max_words"], q["short_words"])}

    def route(self, turn: Turn) -> dict:
        """Domain to search, or None (all domains) without a router or below its confidence threshold."""
        if self.router is None:
            return {"domain": None}
        r, th = self.router, self.router["threshold"]
        threshold = th["follow_up"] if turn.history else th["first_turn"]
        with self.model_lock:
            emb = self.dense.encode([routing_text(turn, r["mode"])])
        return {"domain": r["router"].route(emb, threshold)[0]}

    def corpus_filter(self, turn: Turn) -> dict:
        """Passages that may be retrieved: in force on the turn's as-of date and not withheld; None for all.

        Withheld passages are masked, not removed, so BM25 still uses the full corpus statistics.
        """
        valid = None
        if turn.as_of is not None and self.versioned:
            day = turn.as_of.toordinal()
            valid = (self.start <= day) & (day < self.end)
        if turn.withheld:
            valid = np.ones(len(self.row), bool) if valid is None else valid
            valid[[self.row[p] for p in turn.withheld]] = False
        return {"valid": valid}

    def retrieve(self, query: str, domain: str | None, valid: np.ndarray | None) -> dict:
        if domain is None and self.fan_out:
            return self.retrieve_departments(query, valid)
        bm25 = self.bm25.search(query, self.rcfg["bm25_k"], domain, valid)
        with self.model_lock:
            dense = self.dense.search(query, self.rcfg["dense_k"], domain, valid)
        return {"bm25": bm25, "dense": dense, "candidates": rrf([bm25, dense], k=self.rcfg["rrf_k"]), "skipped": []}

    def retrieve_departments(self, query: str, valid: np.ndarray | None) -> dict:
        """No confident route: search every department on its own, in parallel, each within its own budget.

        A department that runs out of time is left out and listed in `skipped`. Each department's BM25 and
        dense lists are fused, then the departments' lists are fused, so every department that answered
        reaches the reranker even when one department dominates the global scores.
        """
        k, budget = self.fan_out["k"], self.fan_out["timeout_ms"] / 1000

        def one(dept: str) -> tuple[list[str], list[str]]:
            bm25 = self.bm25.search(query, k, dept, valid)
            with self.model_lock:
                return bm25, self.dense.search(query, k, dept, valid)

        futures = {d: self.pool.submit(one, d) for d in self.departments}
        deadline = time.perf_counter() + budget
        lists, skipped = {}, []
        for d, f in futures.items():
            try:
                lists[d] = f.result(timeout=max(deadline - time.perf_counter(), 0.0))
            except TimeoutError:
                skipped.append(d)
        fused = [rrf([b, dn], k=self.rcfg["rrf_k"]) for b, dn in lists.values()]
        return {"bm25": [p for b, _ in lists.values() for p in b], "dense": [p for _, dn in lists.values() for p in dn],
                "candidates": rrf(fused, k=self.rcfg["rrf_k"]), "skipped": skipped}

    def retrieve_bm25(self, query: str, domain: str | None, valid: np.ndarray | None) -> dict:
        bm25 = self.bm25.search(query, self.rcfg["bm25_k"], domain, valid)
        return {"bm25": bm25, "dense": [], "candidates": bm25, "skipped": []}

    def focus(self, query: str, turn: Turn, valid: np.ndarray | None) -> dict:
        """The documents in the session's focus, and their best `k` passages (RRF of BM25 and dense)."""
        if self.focus_cfg is None or not turn.focus_docs:
            return {"focus_docs": [], "focus_pids": []}
        mask = np.isin(self.doc_of, turn.focus_docs)
        if valid is not None:
            mask &= valid
        k = self.focus_cfg["k"]
        if not k:
            return {"focus_docs": turn.focus_docs, "focus_pids": []}
        bm25 = self.bm25.search(query, k, None, mask)
        with self.model_lock:
            dense = self.dense.search(query, k, None, mask)
        return {"focus_docs": turn.focus_docs, "focus_pids": rrf([bm25, dense], k=self.rcfg["rrf_k"])[:k]}

    def rerank(self, query: str, candidates: list[str], focus_docs: list[str], focus_pids: list[str]) -> dict:
        """Cross-encoder order of the candidates and the focus passages, with the session bonus added to
        every passage of a focus document; top_score is the best raw score, None without a rerank."""
        seen = set(candidates)
        pool = candidates + [p for p in focus_pids if p not in seen]
        if self.reranker is None or not pool:
            return {"ranked": pool, "top_score": None}
        with self.model_lock:
            scores = self.reranker.score_many([(query, p) for p in pool])
        bonus = self.focus_cfg["bonus"] if self.focus_cfg else 0.0
        adjusted = [s + bonus * (self.by_pid[p].doc_id in focus_docs) for p, s in zip(pool, scores)]
        order = sorted(range(len(pool)), key=lambda i: -adjusted[i])
        return {"ranked": [pool[i] for i in order], "top_score": max(scores)}

    def run(self, turn: Turn) -> tuple[dict, Trace]:
        return self.graph.run(turn=turn)
