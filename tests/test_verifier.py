from datetime import date

from vpqa.pipeline.verifier import verify
from vpqa.schema import Passage

P = {"old": Passage("old", "20 CFR 1", "ssa", "Fee is $10.", date(2020, 1, 1), date(2022, 1, 1)),
     "new": Passage("new", "20 CFR 1", "ssa", "Fee is $20.", date(2022, 1, 1)),
     "plain": Passage("plain", "doc", "dmv", "Renew online.")}


def judge_all(supported):
    return lambda messages: {"verdicts": [{"claim": i + 1, "supported": s, "problem": "not stated"}
                                          for i, s in enumerate(supported)]}


def never_called(messages):
    raise AssertionError("the judge should not be asked")


def test_stale_version_fails_without_asking_the_judge():
    v = verify([("Fee is $10", ["old"])], [], P, date(2023, 5, 1), never_called)
    assert not v.passed and "not in force on 2023-05-01" in v.problems[0]


def test_format_problems_fail_first():
    v = verify([("Fee is $20", [])], ["uncited: Bring cash."], P, None, never_called)
    assert v.problems == ["uncited: Bring cash.", "claim 1 cites no passage: Fee is $20"]


def test_support_verdicts():
    claims = [("Fee is $20", ["new"]), ("Renew online", ["plain"])]
    assert verify(claims, [], P, date(2023, 5, 1), judge_all([True, True])).passed
    v = verify(claims, [], P, date(2023, 5, 1), judge_all([True, False]))
    assert not v.passed and v.problems[0].startswith("claim 2 is not supported by its passages (not stated)")
    v = verify(claims, [], P, date(2023, 5, 1), lambda m: None)
    assert v.problems == ["claim 1 was not judged", "claim 2 was not judged"]


def test_question_only_passes():
    assert verify([], [], P, None, never_called).passed
