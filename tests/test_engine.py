import json
import time

import pytest

from vpqa.pipeline.engine import ContractError, Graph, Node, NodeFailed, conforms


def double(xs):
    return {"ys": [2 * x for x in xs]}


def total(ys):
    return {"total": sum(ys)}


def test_runs_in_order_and_traces(tmp_path):
    g = Graph([Node("double", double, {"xs": list[int]}, {"ys": list[int]}),
               Node("total", total, {"ys": list[int]}, {"total": int})],
              given={"xs": list[int]})
    state, trace = g.run(xs=[1, 2, 3])
    assert state["total"] == 12
    assert [r["node"] for r in trace.records] == ["<given>", "double", "total"]
    trace.write(tmp_path / "t.jsonl", "r1")
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert lines[2] == {**lines[2], "run_id": "r1", "node": "total", "outputs": {"total": 12}}


def test_missing_upstream_input_is_rejected_at_build():
    with pytest.raises(ContractError, match="not produced"):
        Graph([Node("total", total, {"ys": list[int]}, {"total": int})], given={"xs": list[int]})


def test_type_mismatch_between_nodes_is_rejected_at_build():
    with pytest.raises(ContractError, match="upstream declares"):
        Graph([Node("double", double, {"xs": list[int]}, {"ys": list[int]}),
               Node("total", total, {"ys": list[str]}, {"total": int})], given={"xs": list[int]})


def test_wrong_output_is_rejected_at_run():
    g = Graph([Node("bad", lambda xs: {"ys": ["a"]}, {"xs": list[int]}, {"ys": list[int]})], given={"xs": list[int]})
    with pytest.raises(ContractError, match="not list"):
        g.run(xs=[1])
    g = Graph([Node("bad", lambda xs: {}, {"xs": list[int]}, {"ys": list[int]})], given={"xs": list[int]})
    with pytest.raises(ContractError, match="declared"):
        g.run(xs=[1])


def test_conforms():
    assert conforms(None, str | None)
    assert conforms({"a": [1.0]}, dict[str, list[float]])
    assert not conforms({"a": ["x"]}, dict[str, list[float]])


def slow_double(xs):
    time.sleep(0.5)
    return double(xs)


def keep(xs):
    return {"ys": xs}


def test_timeout_takes_the_fallback():
    g = Graph([Node("double", slow_double, {"xs": list[int]}, {"ys": list[int]}, timeout_s=0.05, fallback=keep),
               Node("total", total, {"ys": list[int]}, {"total": int})], given={"xs": list[int]})
    start = time.perf_counter()
    state, trace = g.run(xs=[1, 2])
    assert time.perf_counter() - start < 0.3
    assert state["total"] == 3
    assert [r["status"] for r in trace.records] == ["ok", "timeout", "ok"]


def test_error_takes_the_fallback_and_without_one_fails():
    def boom(xs):
        raise RuntimeError("down")

    g = Graph([Node("double", boom, {"xs": list[int]}, {"ys": list[int]}, fallback=keep)], given={"xs": list[int]})
    state, trace = g.run(xs=[1])
    assert state["ys"] == [1] and trace.records[-1]["status"] == "error" and "down" in trace.records[-1]["error"]
    g = Graph([Node("double", boom, {"xs": list[int]}, {"ys": list[int]})], given={"xs": list[int]})
    with pytest.raises(NodeFailed, match="down"):
        g.run(xs=[1])


def test_total_budget_sends_later_nodes_to_fallback():
    g = Graph([Node("double", slow_double, {"xs": list[int]}, {"ys": list[int]}, fallback=keep),
               Node("total", total, {"ys": list[int]}, {"total": int}, fallback=lambda ys: {"total": -1})],
              given={"xs": list[int]})
    state, trace = g.run(budget_s=0.1, xs=[1])
    assert [r["status"] for r in trace.records] == ["ok", "timeout", "timeout"]
    assert state["total"] == -1 and "budget spent" in trace.records[-1]["error"]


def test_skip_if_leaves_outputs_and_needs_them_upstream():
    flag = Node("flag", lambda xs: {"done": True, "ys": xs}, {"xs": list[int]}, {"done": bool, "ys": list[int]})
    again = Node("again", double, {"xs": list[int]}, {"ys": list[int]}, skip_if="done")
    state, trace = Graph([flag, again], given={"xs": list[int]}).run(xs=[1])
    assert state["ys"] == [1] and trace.records[-1]["status"] == "skipped"
    with pytest.raises(ContractError, match="skippable"):
        Graph([Node("flag", lambda xs: {"done": True}, {"xs": list[int]}, {"done": bool}), again],
              given={"xs": list[int]})
