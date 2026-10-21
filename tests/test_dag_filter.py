from datetime import date

import numpy as np

from vpqa.pipeline.dag import Pipeline
from vpqa.schema import Passage, Turn


def bare(passages):
    """A Pipeline with only what corpus_filter reads, without loading models."""
    pipe = Pipeline.__new__(Pipeline)
    pipe.row = {p.pid: i for i, p in enumerate(passages)}
    pipe.start = np.array([p.effective_from.toordinal() if p.effective_from else 0 for p in passages])
    pipe.end = np.array([p.effective_to.toordinal() if p.effective_to else np.iinfo(np.int64).max for p in passages])
    pipe.versioned = any(p.effective_from for p in passages)
    return pipe


def test_no_dates_no_withheld_searches_everything():
    pipe = bare([Passage("a", "d", "x", "t"), Passage("b", "d", "x", "t")])
    assert pipe.corpus_filter(Turn("q", 0, "x", [], "q"))["valid"] is None


def test_withheld_passages_are_masked():
    pipe = bare([Passage("a", "d1", "x", "t"), Passage("b", "d2", "x", "t")])
    valid = pipe.corpus_filter(Turn("q", 0, "x", [], "q", withheld=["a"]))["valid"]
    assert valid.tolist() == [False, True]


def test_withheld_combines_with_version_window():
    old = Passage("a1", "d1", "x", "t", date(2020, 1, 1), date(2022, 1, 1))
    new = Passage("a2", "d1", "x", "t", date(2022, 1, 1))
    other = Passage("b", "d2", "x", "t", date(2020, 1, 1))
    pipe = bare([old, new, other])
    turn = Turn("q", 0, "x", [], "q", as_of=date(2023, 1, 1), withheld=["a2"])
    assert pipe.corpus_filter(turn)["valid"].tolist() == [False, False, True]
