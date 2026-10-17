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
