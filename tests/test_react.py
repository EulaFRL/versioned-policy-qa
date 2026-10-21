import json
from types import SimpleNamespace

import numpy as np

from vpqa.pipeline.react import ReactAgent
from vpqa.schema import Passage, Turn

P = {"a": Passage("a", "A", "dmv", "You can renew online."), "b": Passage("b", "B", "ssa", "Benefits start at 62.")}


def stub_pipe(seen):
    def corpus_filter(turn):
        return {"valid": np.array([p not in turn.withheld for p in P]) if turn.withheld else None}

    def retrieve(query, domain, valid):
        seen.append((query, domain, None if valid is None else valid.tolist()))
        return {"candidates": [p for i, p in enumerate(P) if valid is None or valid[i]]}

    return SimpleNamespace(by_pid=P, versioned=False, corpus_filter=corpus_filter, retrieve=retrieve,
                           rerank=lambda q, c: {"ranked": c})


def scripted(replies):
    replies, asked = list(replies), []

    def chat_tools(messages, tools, tool_choice="auto"):
        asked.append((list(messages), tool_choice))
        return replies.pop(0)

    return chat_tools, asked


def call(i, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": "search", "arguments": json.dumps(args)}}


TURN = Turn("d", 1, "dmv", [], "how do I renew?")


def test_search_then_answer_maps_numbers_to_pids():
    seen = []
    chat, asked = scripted([{"role": "assistant", "tool_calls": [call(1, query="renew", department="dmv")]},
                            {"role": "assistant", "content": "Renew online [1]."}])
    r = ReactAgent(stub_pipe(seen), chat).run(TURN)
    assert r.cited_pids == ["a"] and not r.refused and r.llm_calls == 2
    assert seen == [("renew", "dmv", None)] and r.searches[0]["pids"] == ["a", "b"]
    assert asked[1][0][-1]["role"] == "tool" and asked[1][0][-1]["content"].startswith("[1] You can renew")


def test_answer_without_search_is_recorded():
    chat, _ = scripted([{"role": "assistant", "content": "Renew at the DMV."}])
    r = ReactAgent(stub_pipe([]), chat).run(TURN)
    assert r.searches == [] and r.issues == ["uncited: Renew at the DMV."]


def test_search_budget_forces_an_answer_and_withheld_stays_hidden():
    seen = []
    replies = [{"role": "assistant", "tool_calls": [call(i, query=f"q{i}")]} for i in range(2)]
    chat, asked = scripted(replies + [{"role": "assistant", "content": "NOT_FOUND"}])
    r = ReactAgent(stub_pipe(seen), chat, max_searches=2).run(Turn("d", 1, "dmv", [], "q", withheld=["a"]))
    assert r.refused and r.forced and len(r.searches) == 2 and asked[-1][1] == "none"
    assert all(v == [False, True] for _, _, v in seen)
