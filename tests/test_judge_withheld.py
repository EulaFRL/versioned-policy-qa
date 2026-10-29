import importlib.util
import json
from pathlib import Path

import pytest

from vpqa import llm


spec = importlib.util.spec_from_file_location(
    "judge_withheld", Path(__file__).parents[1] / "scripts/judge_withheld.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


@pytest.mark.parametrize("reply", ["", "not JSON", '{}', '{"covered": "false"}', '[]'])
def test_missing_or_invalid_verdict_is_unknown(reply):
    assert judge.parse(reply)[0] is None


def test_truncated_reason_keeps_explicit_boolean():
    assert judge.parse('{"covered": false, "reason": "not in the text') == (
        False, "not in the text [cut off]")


def test_cache_only_never_calls_api(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cache-only mode called the network")
    monkeypatch.setattr(llm, "_fetch", forbidden)
    with pytest.raises(FileNotFoundError):
        llm.chat([{"role": "user", "content": "missing"}], model_name="test",
                 cache=tmp_path, cache_only=True)


def test_interrupted_judge_resumes_without_losing_or_repeating_verdicts(tmp_path, monkeypatch):
    out = tmp_path / "run.covered.json"
    prompts = {"a": "first", "b": "second"}
    calls = []

    def interrupted(messages, **kwargs):
        calls.append(messages[0]["content"])
        if len(calls) == 2:
            raise ConnectionError("disconnected")
        return '{"covered": false, "reason": "missing"}'

    monkeypatch.setattr(llm, "chat", interrupted)
    with pytest.raises(ConnectionError):
        judge.judge_prompts(prompts, out, "test")
    assert not out.exists()
    monkeypatch.setattr(llm, "chat", lambda messages, **kwargs:
                        calls.append(messages[0]["content"]) or '{"covered": true}')
    assert judge.judge_prompts(prompts, out, "test") == {"a": False, "b": True}
    assert calls == ["first", "second", "second"]
    assert json.loads(out.read_text()) == {"a": False, "b": True}


def test_unknown_does_not_publish_complete_labels(tmp_path, monkeypatch):
    out = tmp_path / "run.covered.json"
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: "")
    assert judge.judge_prompts({"a": "first"}, out, "test") is None
    assert not out.exists()
