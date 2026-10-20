from vpqa.eval.metrics import mrr, recall_at_k, refusal_prf
from vpqa.retrieval.fusion import rrf


def test_recall_and_mrr():
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert mrr(["a", "b", "c"], ["b"]) == 0.5


def test_rrf_prefers_consensus():
    assert rrf([["a", "b"], ["b", "a"], ["b", "c"]])[0] == "b"


def test_refusal_prf():
    p, r, f = refusal_prf([True, False, True], [True, True, False])
    assert (p, r) == (0.5, 0.5)


def test_gated_query_uses_history_only_for_replies():
    from vpqa.retrieval.query import build_query
    from vpqa.schema import Turn

    hist = ["user: my insurance ended", "agent: did you get a letter from the DMV?"]
    reply = Turn("d", 3, "dmv", hist, "yes")
    question = Turn("d", 3, "dmv", hist, "how do I renew my license?")
    assert build_query(reply, "gated", history_turns=1) == "yes agent: did you get a letter from the DMV?"
    assert build_query(question, "gated", history_turns=1) == "how do I renew my license?"
    long = Turn("d", 3, "dmv", ["a " * 200], "yes")
    assert build_query(long, "recent", max_words=10).split()[0] == "yes"
    assert len(build_query(long, "recent", max_words=10).split()) == 10


def test_rerank_scores_map_back_to_pairs():
    from vpqa.retrieval.rerank import Reranker

    class Stub:  # scores a pair by passage length, so the length-sorted batching must be undone
        def predict(self, pairs, **_):
            return [float(len(p)) for _, p in pairs]

    r = object.__new__(Reranker)
    r.model, r.batch_size = Stub(), 2
    r.text = {"long": "x" * 30, "short": "x", "mid": "x" * 10}
    assert r.score_many([("q", "long"), ("q", "short"), ("q", "mid")]) == [30.0, 1.0, 10.0]
    assert r.rerank("q", ["short", "long", "mid"], top_n=2) == ["long", "mid"]
