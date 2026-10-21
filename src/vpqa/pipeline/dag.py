"""Fixed pipeline: query -> route -> version filter -> retrieve -> rerank -> answer -> verify.

Every stage is a node of vpqa.pipeline.engine.Graph with declared input and output types, so the order
is fixed in code and each run leaves a per-node trace. Stages with a time budget (configs `harness`) degrade
instead of failing: route -> search all domains, retrieve -> BM25 only, rerank -> keep the RRF order,
answer -> quote the top passage, verify -> fail the answer. The
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

from vpqa import llm
from vpqa.pipeline import answer, verifier
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
                 router: Path | None = None, rerank: bool = True, timeouts: bool = True, generate: bool = False):
        qcfg, self.rcfg, self.acfg = cfg["query"], cfg["retrieval"], cfg.get("answer", {})
        self.qcfg = qcfg
        self.by_pid = {p.pid: p for p in passages}
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
        nodes = [
            Node("query", self.query, {"turn": Turn}, {"query": str}),
            Node("route", self.route, {"turn": Turn}, {"domain": str | None},
                 limit("route"), fallback=lambda turn: {"domain": None}),
            Node("version_filter", self.version_filter, {"turn": Turn}, {"valid": np.ndarray | None}),
            Node("retrieve", self.retrieve, {"query": str, "domain": str | None, "valid": np.ndarray | None},
                 {"bm25": list[str], "dense": list[str], "candidates": list[str]},
                 limit("retrieve"), fallback=self.retrieve_bm25),
            Node("rerank", self.rerank, {"query": str, "candidates": list[str]}, {"ranked": list[str]},
                 limit("rerank"), fallback=lambda query, candidates: {"ranked": candidates}),
        ]
        if generate:
            nodes.append(Node("answer", self.respond, {"turn": Turn, "ranked": list[str]},
                              {"answer": str, "cited_pids": list[str], "refused": bool, "issues": list[str],
                               "claims": list[dict]},
                              limit("answer"), fallback=self.quote))
            nodes.append(Node("verify", self.verify, {"turn": Turn, "refused": bool, "claims": list[dict],
                                                      "issues": list[str]},
                              {"verified": bool, "problems": list[str]}, limit("verify"),
                              fallback=lambda **_: {"verified": False, "problems": ["verifier did not finish"]}))
        self.graph = Graph(nodes, given={"turn": Turn})
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

    def respond(self, turn: Turn, ranked: list[str]) -> dict:
        if not ranked:
            return {"answer": answer.NOT_FOUND, "cited_pids": [], "refused": True, "issues": ["no passages"],
                    "claims": []}
        shown = [self.by_pid[p] for p in ranked[:self.rcfg["top_n"]]]
        parsed = answer.answer(turn, shown, llm.chat, **self.acfg)
        return {"answer": parsed.answer, "cited_pids": parsed.cited_pids, "refused": parsed.refused,
                "issues": parsed.issues, "claims": [{"text": t, "pids": ps} for t, ps in parsed.claims]}

    def quote(self, turn: Turn, ranked: list[str]) -> dict:
        """Fallback when no answer could be generated: show the top passage as is."""
        if not ranked:
            return {"answer": answer.NOT_FOUND, "cited_pids": [], "refused": True, "issues": ["no passages"],
                    "claims": []}
        top = self.by_pid[ranked[0]]
        text = " ".join(top.text.split()[:80])
        return {"answer": f"I could not compose a checked answer. The most relevant passage says: \"{text}\" [1]",
                "cited_pids": [top.pid], "refused": False, "issues": ["fallback: quoted the top passage"],
                "claims": []}

    def verify(self, turn: Turn, refused: bool, claims: list[dict], issues: list[str]) -> dict:
        if refused:
            return {"verified": True, "problems": []}
        v = verifier.verify([(c["text"], c["pids"]) for c in claims], issues, self.by_pid, turn.as_of,
                            llm.chat_json, self.acfg.get("passage_words", 400))
        return {"verified": v.passed, "problems": v.problems}

    def run(self, turn: Turn) -> tuple[dict, Trace]:
        return self.graph.run(turn=turn)
