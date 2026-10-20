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


def test_router_text_and_threshold():
    import numpy as np

    from vpqa.retrieval.route import DomainRouter, routing_text
    from vpqa.schema import Turn

    t = Turn("d", 3, "dmv", ["user: a b", "agent: c d"], "yes")
    assert routing_text(t, "current") == "yes"
    assert routing_text(t, "dialogue") == "yes agent: c d user: a b"
    assert routing_text(t, "dialogue", max_words=3) == "yes agent: c"

    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    router = DomainRouter().fit(emb, ["dmv", "dmv", "va", "va"])
    assert router.route(np.array([[1.0, 0.0], [0.5, 0.5]]), threshold=0.55) == ["dmv", None]


def test_units_hold_their_spans():
    from vpqa.data.multidoc2dial import _doc_units

    def span(sp, sec, title, start, text):
        return {"id_sp": sp, "id_sec": sec, "title": title, "parent_titles": [], "start_sp": start,
                "end_sp": start + len(text), "text_sp": text, "text_sec": text}

    text = "Intro " + " ".join(f"w{i}" for i in range(150))
    doc = {"title": "Doc#1", "doc_text": text, "spans": {
        "1": span("1", "t_0", "Intro", 0, "Intro "),
        "2": span("2", "1", "Intro", 6, text[6:500]),
        "3": span("3", "2", "Other", 500, text[500:])}}
    assert [held for _, _, held in _doc_units(doc, "section")] == [["1"], ["2"], ["3"]]
    assert [held for _, _, held in _doc_units(doc, "heading")] == [["1", "2"], ["3"]]  # same title merges
    token = _doc_units(doc, "token")
    assert len(token) == 2 and token[0][1].startswith("Doc\nIntro w0")
    assert "2" in token[0][2] and "2" in token[1][2]  # a span cut by the window belongs to both


def test_span_recall_within_budget():
    from vpqa.eval.metrics import span_recall_within

    words = {"a": 50, "b": 300, "c": 50}
    span_pids = {"s1": ["a"], "s2": ["c"]}
    assert span_recall_within(["a", "c", "b"], ["s1", "s2"], span_pids, words, 200) == 1.0
    assert span_recall_within(["a", "b", "c"], ["s1", "s2"], span_pids, words, 200) == 0.5  # stops at b
