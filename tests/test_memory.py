from types import SimpleNamespace

import numpy as np

from vpqa.memory import Session
from vpqa.pipeline.dag import Pipeline
from vpqa.retrieval.query import rewrite_query
from vpqa.schema import Passage, Turn

P = [Passage("a1", "A", "x", "t"), Passage("a2", "A", "x", "t"), Passage("b1", "B", "x", "t")]


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def bare(scores, bonus=0.5):
    """A Pipeline with a fake reranker and searchers that return every passage they may."""
    pipe = Pipeline.__new__(Pipeline)
    pipe.by_pid = {p.pid: p for p in P}
    pipe.doc_of = np.array([p.doc_id for p in P])
    pipe.focus_cfg = {"enabled": True, "k": 10, "bonus": bonus}
    pipe.rcfg = {"rrf_k": 60}
    pipe.model_lock = NoLock()
    search = lambda q, k, d, mask: [p.pid for p, m in zip(P, mask) if m][:k]
    pipe.bm25 = SimpleNamespace(search=search)
    pipe.dense = SimpleNamespace(search=search)
    pipe.reranker = SimpleNamespace(score_many=lambda pairs: [scores[p] for _, p in pairs])
    return pipe


def test_session_focus_is_most_recent_first_without_repeats():
    s = Session()
    s.record(["A", "A"])
    s.record(["B"])
    assert s.focus(1) == ["B"]
    assert s.focus(2) == ["B", "A"]


def test_focus_searches_only_the_focus_documents():
    pipe = bare({})
    out = pipe.focus("q", Turn("d", 1, "x", ["user: hi"], "q", focus_docs=["A"]), None)
    assert out["focus_docs"] == ["A"] and sorted(out["focus_pids"]) == ["a1", "a2"]
    assert pipe.focus("q", Turn("d", 1, "x", [], "q"), None) == {"focus_docs": [], "focus_pids": []}


def test_focus_bonus_lifts_the_focus_document_and_adds_its_passages():
    pipe = bare({"a1": 0.2, "a2": 0.1, "b1": 0.6})
    out = pipe.rerank("q", ["b1", "a2"], ["A"], ["a1"])
    assert out["ranked"] == ["a1", "b1", "a2"]  # 0.7, 0.6, 0.6: ties keep candidate order
    assert out["top_score"] == 0.6
    pipe.focus_cfg = None
    assert pipe.rerank("q", ["b1", "a2"], [], [])["ranked"] == ["b1", "a2"]


def test_rewrite_takes_the_first_line_and_none_when_empty():
    turn = Turn("d", 1, "x", ["user: I lost my card", "agent: Which card?"], "the green one")
    seen = []
    chat = lambda m: seen.append(m) or '"How do I replace a lost green card?"\nextra'
    assert rewrite_query(turn, chat) == "How do I replace a lost green card?"
    assert "agent: Which card?" in seen[0][0]["content"]
    assert rewrite_query(turn, lambda m: "  ") is None


def test_answer_memory_reuses_only_versions_in_force():
    from datetime import date

    from vpqa.memory import AnswerMemory
    from vpqa.schema import Passage

    passages = {"a@1": Passage("a@1", "A", "x", "old", date(2020, 1, 1), date(2022, 1, 1)),
                "a@2": Passage("a@2", "A", "x", "new", date(2022, 1, 1), None),
                "b@1": Passage("b@1", "B", "x", "other", date(2020, 1, 1), None)}
    vecs = {"q": np.array([1.0, 0.0]), "q2": np.array([0.96, 0.28]), "far": np.array([0.0, 1.0])}
    encode = lambda texts: np.stack([vecs[t] for t in texts])
    bound = AnswerMemory(encode, passages, threshold=0.9)
    bound.store("q", "old answer", ["a@1", "b@1"])
    assert bound.lookup("q2", date(2021, 6, 1))[0].answer == "old answer"  # paraphrase, same version window
    assert bound.lookup("q", date(2023, 1, 1)) == (None, "stale")  # a@1 expired
    assert bound.lookup("far", date(2021, 6, 1)) == (None, "miss")
    bound.store("q", "new answer", ["a@2"])
    assert bound.lookup("q", date(2023, 1, 1))[0].answer == "new answer"
    assert bound.lookup("q", date(2021, 1, 1))[0].answer == "old answer"

    assert not bound.store("q", "Which state do you live in?", [])  # a clarifying question binds to nothing

    unbound = AnswerMemory(encode, passages, threshold=0.9, bind=False)
    unbound.store("q", "old answer", ["a@1"])
    assert unbound.lookup("q", date(2023, 1, 1))[0].answer == "old answer"
