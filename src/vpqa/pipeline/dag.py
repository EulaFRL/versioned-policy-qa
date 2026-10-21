"""Fixed pipeline: query -> route -> version filter -> retrieve -> rerank (-> answer -> verify, to come).

Every stage is a node of vpqa.pipeline.engine.Graph with declared input and output types, so the order
is fixed in code and each run leaves a per-node trace. Stages with a time budget (configs `harness`) degrade
instead of failing: route -> search all domains, retrieve -> BM25 only, rerank -> keep the RRF order. The
version filter has no fallback: searching every version would silently return stale text.

Model calls hold one lock: a timed-out stage keeps running in its thread, and two threads using the GPU at
once crash Metal (MPS). A stage that waits for the lock spends its own budget and falls back.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vpqa.pipeline.engine import Graph, Node, Trace
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query
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
                 router: Path | None = None, rerank: bool = True, timeouts: bool = True):
        qcfg, self.rcfg = cfg["query"], cfg["retrieval"]
        self.qcfg = qcfg
        self.bm25 = BM25Index(passages)
        self.dense = DenseIndex(passages, self.rcfg["dense_model"], cache=dense_cache)
        self.reranker = Reranker(passages, cfg["rerank"]["model"]) if rerank else None
        self.reranker_pid = passages[0].pid
        self.model_lock = threading.Lock()
        self.router = None
        if router is not None:
            with open(router, "rb") as f:
                self.router = pickle.load(f)
        # validity windows as day ordinals; no end date means still in force
        self.start = np.array([p.effective_from.toordinal() if p.effective_from else 0 for p in passages])
        self.end = np.array([p.effective_to.toordinal() if p.effective_to else np.iinfo(np.int64).max
                             for p in passages])
        self.versioned = any(p.effective_from for p in passages)
        ms = cfg.get("harness", {}).get("timeouts_ms", {}) if timeouts else {}
        limit = lambda name: ms[name] / 1000 if name in ms else None
        self.graph = Graph([
            Node("query", self.query, {"turn": Turn}, {"query": str}),
            Node("route", self.route, {"turn": Turn}, {"domain": str | None},
                 limit("route"), fallback=lambda turn: {"domain": None}),
            Node("version_filter", self.version_filter, {"turn": Turn}, {"valid": np.ndarray | None}),
            Node("retrieve", self.retrieve, {"query": str, "domain": str | None, "valid": np.ndarray | None},
                 {"bm25": list[str], "dense": list[str], "candidates": list[str]},
                 limit("retrieve"), fallback=self.retrieve_bm25),
            Node("rerank", self.rerank, {"query": str, "candidates": list[str]}, {"ranked": list[str]},
                 limit("rerank"), fallback=lambda query, candidates: {"ranked": candidates}),
        ], given={"turn": Turn})
        self.warm_up()

    def warm_up(self) -> None:
        """The first call to each model is slow (weights moved to the device); keep it out of the budgets."""
        self.dense.encode(["warm up"])
        if self.reranker is not None:
            self.reranker.score_many([("warm up", self.reranker_pid)])

    def query(self, turn: Turn) -> dict:
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

    def version_filter(self, turn: Turn) -> dict:
        """Passages in force on the turn's as-of date; None when the turn or corpus carries no dates."""
        if turn.as_of is None or not self.versioned:
            return {"valid": None}
        day = turn.as_of.toordinal()
        return {"valid": (self.start <= day) & (day < self.end)}

    def retrieve(self, query: str, domain: str | None, valid: np.ndarray | None) -> dict:
        bm25 = self.bm25.search(query, self.rcfg["bm25_k"], domain, valid)
        with self.model_lock:
            dense = self.dense.search(query, self.rcfg["dense_k"], domain, valid)
        return {"bm25": bm25, "dense": dense, "candidates": rrf([bm25, dense], k=self.rcfg["rrf_k"])}

    def retrieve_bm25(self, query: str, domain: str | None, valid: np.ndarray | None) -> dict:
        bm25 = self.bm25.search(query, self.rcfg["bm25_k"], domain, valid)
        return {"bm25": bm25, "dense": [], "candidates": bm25}

    def rerank(self, query: str, candidates: list[str]) -> dict:
        if self.reranker is None:
            return {"ranked": candidates}
        with self.model_lock:
            return {"ranked": self.reranker.rerank(query, candidates)}

    def run(self, turn: Turn) -> tuple[dict, Trace]:
        return self.graph.run(turn=turn)
