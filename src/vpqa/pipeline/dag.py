"""Fixed pipeline: rewrite -> route -> retrieve -> rerank -> version filter -> answer -> verify."""
from __future__ import annotations

from dataclasses import dataclass

from vpqa.schema import Turn


@dataclass
class Result:
    answer: str | None
    cited_pids: list[str]
    refused: bool
    trace: dict


def run(turn: Turn) -> Result:
    raise NotImplementedError
