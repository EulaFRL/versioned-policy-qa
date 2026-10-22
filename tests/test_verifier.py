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


def test_question_check_rejects_an_answer_to_another_question():
    claims = [("Fee is $20", ["new"])]
    seen = []

    def judge(messages):
        seen.append(messages[0]["content"])
        if "addresses_question" in seen[-1]:
            return {"addresses_question": False, "gap": "the fee for seniors"}
        return {"verdicts": [{"claim": 1, "supported": True}]}

    v = verify(claims, [], P, None, judge, question="Latest user message: What do seniors pay?")
    assert not v.passed and "the fee for seniors" in v.problems[0]
    assert len(seen) == 2 and "What do seniors pay?" in seen[1] and "seniors" not in seen[0]  # support prompt unchanged
    seen.clear()
    verify(claims, [], P, None, lambda m: seen.append(1) or {"verdicts": [{"claim": 1, "supported": False}]},
           question="Latest user message: What do seniors pay?")
    assert len(seen) == 1  # an unsupported answer is not checked against the question
    assert verify(claims, [], P, None, judge_all([True])).passed  # without the question, support only
