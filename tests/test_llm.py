import threading
import time

from vpqa import llm


def test_hedged_keeps_the_faster_copy(monkeypatch):
    calls, lock = [], threading.Lock()

    def fetch(base, headers, payload, body):
        with lock:
            calls.append(len(calls))
            n = calls[-1]
        time.sleep(0.5 if n == 0 else 0.05)  # the first copy is slow
        return {"role": "assistant", "content": f"copy {n}"}, {"cost": 1.0}

    monkeypatch.setattr(llm, "_fetch", fetch)
    before = dict(llm.TOTALS)
    message, _ = llm._hedged(0.1, "b", {}, {}, {"model": "m"})
    assert message["content"] == "copy 1"
    time.sleep(0.6)  # the slower copy finishes and is counted
    assert llm.TOTALS["hedged"] == before["hedged"] + 1
    assert llm.TOTALS["calls"] == before["calls"] + 1


def test_hedged_no_second_copy_when_fast(monkeypatch):
    monkeypatch.setattr(llm, "_fetch", lambda *a: ({"role": "assistant", "content": "ok"}, None))
    before = llm.TOTALS["hedged"]
    assert llm._hedged(1.0, "b", {}, {}, {"model": "m"})[0]["content"] == "ok"
    assert llm.TOTALS["hedged"] == before
