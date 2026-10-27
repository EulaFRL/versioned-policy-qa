from vpqa.pipeline.engine import Graph
from vpqa.pipeline.generate import DEGRADED, Generator
from vpqa.schema import Passage, Turn

P = {"a": Passage("a", "doc", "dmv", "You can renew online."), "b": Passage("b", "doc", "dmv", "Fees are $20.")}
TURN = Turn("d", 1, "dmv", [], "how do I renew?")


def run(replies, verdicts, top_score=1.0, min_score=None):
    """Stub answer model returning `replies` in order; stub judge returning one verdict per call."""
    replies, verdicts, asked = list(replies), list(verdicts), []

    def llm(messages):
        asked.append(messages)
        return replies.pop(0)

    def judge(messages):
        ok = verdicts.pop(0)
        return {"verdicts": [{"claim": 1, "supported": ok, "problem": "" if ok else "not stated"}]}

    g = Generator(P, llm, judge, min_score=min_score)
    given = {"turn": Turn, "ranked": list[str], "top_score": float | None}
    state, trace = Graph(g.nodes(), given=given).run(turn=TURN, ranked=["a", "b"], top_score=top_score)
    return state, {r["node"]: r["status"] for r in trace.records}, asked


def test_passes_first_time():
    state, status, _ = run(["Renew online [1]."], [True])
    assert state["outcome"] == "answered" and state["final_pids"] == ["a"]
    assert status["revise_1"] == status["verify_2"] == "skipped"


def test_revision_fixes_the_answer_and_sees_the_problems():
    state, status, asked = run(["Renew online for free [1].", "Renew online [1]."], [False, True])
    assert state["outcome"] == "answered" and state["final"] == "Renew online [1]."
    assert status["revise_1"] == "ok" and status["revise_2"] == "skipped"
    assert asked[1][1] == {"role": "assistant", "content": "Renew online for free [1]."}
    assert "not supported by its passages (not stated)" in asked[1][2]["content"]


def test_degrades_to_passage_text_after_two_revisions():
    state, _, asked = run(["Free [1].", "Still free [1].", "Free again [2]."], [False, False, False])
    assert len(asked) == 3 and state["outcome"] == "degraded"
    assert state["final"] == f'{DEGRADED} "Fees are $20." [1]' and state["final_pids"] == ["b"]


def test_refusal_is_kept_and_format_problems_skip_the_judge():
    state, _, _ = run(["NOT_FOUND"], [])
    assert state["outcome"] == "refused"
    state, _, asked = run(["Renew online.", "Renew online [1]."], [True])  # first reply has no citation
    assert state["outcome"] == "answered" and "uncited: Renew online." in asked[1][2]["content"]


def test_low_rerank_score_refuses_without_asking():
    state, _, asked = run([], [], top_score=-3.0, min_score=-1.0)
    assert state["outcome"] == "refused" and asked == []
    state, _, asked = run(["Renew online [1]."], [True], top_score=0.5, min_score=-1.0)
    assert state["outcome"] == "answered" and len(asked) == 1


def run_checked(mode, conditions, replies=("Renew online [1].",)):
    replies, asked = list(replies), []

    def llm(messages):
        asked.append(messages)
        return replies.pop(0)

    g = Generator(P, llm, lambda m: {"verdicts": [{"claim": 1, "supported": True}]}, condition_check=mode,
                  llm_json=lambda m: {"conditions": conditions})
    given = {"turn": Turn, "ranked": list[str], "top_score": float | None}
    state, _ = Graph(g.nodes(), given=given).run(turn=TURN, ranked=["a", "b"], top_score=1.0)
    return state, asked


def test_condition_check_refuses_or_notes_uncovered_conditions():
    minor = {"condition": "under 18", "covered": False, "essential": True}
    aside = {"condition": "in Ohio", "covered": False, "essential": False}
    covered = {"condition": "a car", "covered": True, "essential": True}
    state, asked = run_checked("any", [aside])
    assert state["outcome"] == "refused" and not asked
    state, asked = run_checked("essential", [aside, covered])
    assert state["outcome"] == "answered" and len(asked) == 1
    state, _ = run_checked("essential", [minor])
    assert state["outcome"] == "refused"
    state, asked = run_checked("note", [minor, covered])
    assert state["outcome"] == "answered"
    assert "do not mention these conditions from the user's message: under 18." in asked[0][0]["content"]
    state, asked = run_checked("any", "malformed")  # an unreadable check changes nothing
    assert state["outcome"] == "answered" and "do not mention" not in asked[0][0]["content"]
