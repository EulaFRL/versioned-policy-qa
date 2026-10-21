from vpqa.pipeline.engine import Graph
from vpqa.pipeline.generate import DEGRADED, Generator
from vpqa.schema import Passage, Turn

P = {"a": Passage("a", "doc", "dmv", "You can renew online."), "b": Passage("b", "doc", "dmv", "Fees are $20.")}
TURN = Turn("d", 1, "dmv", [], "how do I renew?")


def run(replies, verdicts):
    """Stub answer model returning `replies` in order; stub judge returning one verdict per call."""
    replies, verdicts, asked = list(replies), list(verdicts), []

    def llm(messages):
        asked.append(messages)
        return replies.pop(0)

    def judge(messages):
        ok = verdicts.pop(0)
        return {"verdicts": [{"claim": 1, "supported": ok, "problem": "" if ok else "not stated"}]}

    g = Generator(P, llm, judge)
    state, trace = Graph(g.nodes(), given={"turn": Turn, "ranked": list[str]}).run(turn=TURN, ranked=["a", "b"])
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
