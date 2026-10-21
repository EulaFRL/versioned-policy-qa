"""A small fixed-DAG engine: nodes declare typed inputs and outputs, run in a fixed order, and leave a trace.

Each node has a time budget. A node that runs out of time or raises is replaced by its fallback (a cheaper
way to produce the same outputs, e.g. skip the rerank and keep the fused order); a node without a fallback
fails the run. A run can also have a total budget: once it is spent, the remaining nodes go straight to
their fallbacks. Contract violations are bugs and are never degraded.

Python cannot stop a thread, so a timed-out node keeps running in the background and its result is dropped.
"""
from __future__ import annotations

import json
import threading
import time
import types
import typing
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable


class ContractError(Exception):
    """A node read or wrote something its declaration does not allow."""


class StageTimeout(Exception):
    pass


class NodeFailed(Exception):
    """A node raised or timed out and has no fallback."""


@dataclass
class Node:
    name: str
    fn: Callable[..., dict[str, Any]]  # called with the declared inputs as keyword arguments
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    timeout_s: float | None = None  # budget for this stage
    fallback: Callable[..., dict[str, Any]] | None = None  # same inputs and outputs as fn
    skip_if: str | None = None  # name of a bool in the state; all outputs must already exist upstream


def call_with_timeout(fn: Callable[..., dict], args: dict, timeout_s: float | None) -> dict:
    if timeout_s is None:
        return fn(**args)
    box: dict[str, Any] = {}

    def target():
        try:
            box["out"] = fn(**args)
        except BaseException as e:  # re-raised in the caller's thread
            box["err"] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(max(timeout_s, 0.0))
    if worker.is_alive():
        raise StageTimeout(f"over {timeout_s:.3f}s")
    if "err" in box:
        raise box["err"]
    return box["out"]


def conforms(value: Any, tp: Any) -> bool:
    """isinstance for the annotations nodes use: plain classes, X | None, list[X], dict[K, V]."""
    if tp is Any:
        return True
    origin = typing.get_origin(tp)
    if origin is None:
        return isinstance(value, tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        return any(conforms(value, a) for a in args)
    if not isinstance(value, origin):
        return False
    if origin in (list, tuple, set) and args:
        return all(conforms(v, args[0]) for v in value)
    if origin is dict and args:
        return all(conforms(k, args[0]) and conforms(v, args[1]) for k, v in value.items())
    return True


def summarize(value: Any, limit: int = 200) -> Any:
    """Short JSON-safe view of a value for the trace."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:limit] if isinstance(value, str) else value
    if isinstance(value, (list, tuple)):
        head = [summarize(v, limit) for v in list(value)[:5]]
        return head + [f"... {len(value)} items"] if len(value) > 5 else head
    if isinstance(value, dict):
        return {str(k): summarize(v, limit) for k, v in list(value.items())[:10]}
    if is_dataclass(value):
        return summarize(asdict(value), limit)
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy array
        return f"<array {tuple(value.shape)} {value.dtype}" + (f", {int(value.sum())} true>" if value.dtype == bool else ">")
    return str(value)[:limit]


@dataclass
class Trace:
    records: list[dict] = field(default_factory=list)

    def add(self, **record) -> None:
        self.records.append(record)

    def write(self, path: Path, run_id: str) -> None:
        with Path(path).open("a") as f:
            for r in self.records:
                f.write(json.dumps({"run_id": run_id, **r}, default=str) + "\n")


class Graph:
    def __init__(self, nodes: list[Node], given: dict[str, Any]):
        """`given` declares the inputs supplied by the caller. Every node input must be given or produced upstream."""
        self.nodes = nodes
        self.given = given
        available = dict(given)
        for node in nodes:
            for key, tp in node.inputs.items():
                if key not in available:
                    raise ContractError(f"{node.name}: input {key!r} is not produced by any earlier node")
                if available[key] != tp and tp is not Any:
                    raise ContractError(f"{node.name}: input {key!r} is {tp}, upstream declares {available[key]}")
            if node.skip_if is not None:
                if available.get(node.skip_if) is not bool:
                    raise ContractError(f"{node.name}: skip_if {node.skip_if!r} is not an upstream bool")
                missing = [k for k, tp in node.outputs.items() if available.get(k) != tp]
                if missing:
                    raise ContractError(f"{node.name}: skippable, but {missing} not produced upstream")
            available.update(node.outputs)

    def run(self, budget_s: float | None = None, **values) -> tuple[dict[str, Any], Trace]:
        for key, tp in self.given.items():
            if key not in values or not conforms(values[key], tp):
                raise ContractError(f"given input {key!r} missing or not {tp}")
        state = dict(values)
        trace = Trace()
        trace.add(node="<given>", status="ok", ms=0.0, outputs=summarize(values))
        deadline = None if budget_s is None else time.perf_counter() + budget_s
        for node in self.nodes:
            if node.skip_if is not None and state[node.skip_if]:
                trace.add(node=node.name, status="skipped", ms=0.0)
                continue
            args = {k: state[k] for k in node.inputs}
            timeout = node.timeout_s
            if deadline is not None:
                timeout = min(timeout if timeout is not None else float("inf"), deadline - time.perf_counter())
            start = time.perf_counter()
            record = {"node": node.name, "inputs": summarize({k: v for k, v in args.items() if k not in self.given})}
            try:
                if timeout is not None and timeout <= 0 and node.fallback is not None:
                    raise StageTimeout("total budget spent")
                out, status = call_with_timeout(node.fn, args, timeout), "ok"
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {e}"[:300]
                if node.fallback is None:
                    trace.add(**record, status="failed", ms=round((time.perf_counter() - start) * 1000, 1))
                    raise NodeFailed(f"{node.name}: {record['error']}") from e
                out, status = node.fallback(**args), "timeout" if isinstance(e, StageTimeout) else "error"
            self.check_outputs(node, out)
            state.update(out)
            trace.add(**record, status=status, ms=round((time.perf_counter() - start) * 1000, 1),
                      outputs=summarize(out))
        return state, trace

    @staticmethod
    def check_outputs(node: Node, out: dict[str, Any]) -> None:
        if set(out) != set(node.outputs):
            raise ContractError(f"{node.name}: returned {sorted(out)}, declared {sorted(node.outputs)}")
        for key, tp in node.outputs.items():
            if not conforms(out[key], tp):
                raise ContractError(f"{node.name}: output {key!r} is not {tp}")
