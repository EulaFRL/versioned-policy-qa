from datetime import date

from vpqa.pipeline.answer import build_prompt, parse
from vpqa.schema import Passage, Turn


def test_parse_maps_citations_and_flags_issues():
    p = parse("You need U.S. citizenship [2]. It takes ten days [1] [2]. See also [7]. Bring your card.", ["a", "b"])
    assert p.cited_pids == ["b", "a"] and not p.refused
    assert p.claims == [("You need U.S. citizenship", ["b"]), ("It takes ten days", ["a", "b"]), ("See also", [])]
    assert p.issues == ["unknown citation [7]", "uncited: Bring your card."]


def test_parse_question_and_refusal():
    assert parse("Good morning. Is this for a new license?", ["a"]).issues == []
    assert parse("NOT_FOUND", ["a"]).refused
    assert parse("", ["a"]).issues == ["empty reply"]


def test_prompt_numbers_passages_and_shows_validity():
    p = [Passage("x@2020-01-01", "38 CFR 3.1", "va", "text one", date(2020, 1, 1), date(2021, 1, 1)),
         Passage("y", "doc", "dmv", "text two")]
    prompt = build_prompt(Turn("d", 1, "", ["user: hi"], "what?"), p)
    assert "[1] (38 CFR 3.1, in force from 2020-01-01 until 2021-01-01) text one" in prompt
    assert "[2] text two" in prompt and "x@2020" not in prompt
