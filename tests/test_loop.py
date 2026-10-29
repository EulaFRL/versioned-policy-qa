from vpqa.loop import attribute, bootstrap_diff, mcnemar, merge, passes, sign_flip


def test_merge_nested_and_lists():
    cfg = {"answer": {"rules": ["a"], "passage_words": 400}, "route": {"threshold": {"first_turn": 0.9, "follow_up": 0.7}}}
    out = merge(cfg, {"answer": {"rules": ["b", "a"]}, "route": {"threshold": {"follow_up": 0.9}}})
    assert out["answer"] == {"rules": ["a", "b"], "passage_words": 400}
    assert out["route"]["threshold"] == {"first_turn": 0.9, "follow_up": 0.9}
    assert cfg["answer"]["rules"] == ["a"]  # not modified


def test_attribute_order():
    trace = lambda routed, ranked: {"<given>": {"turn": {"domain": "dmv"}}, "route": {"domain": routed},
                                    "rerank": {"ranked": ranked}}
    row = lambda outcome, final: {"outcome": outcome, "final_pids": final, "gold": ["g"]}
    assert attribute(row("answered", ["g"]), trace("dmv", ["g"]), True, 5) is None
    assert attribute(row("answered", ["x"]), trace("va", ["x"]), True, 5) == "intent"
    assert attribute(row("answered", ["x"]), trace(None, ["x"] * 5 + ["g"]), True, 5) == "retrieval"
    assert attribute(row("refused", []), trace("dmv", ["x", "g"]), True, 5) == "generation"
    assert attribute(row("answered", ["x"]), trace("dmv", ["x"]), False, 5) == "knowledge gap"
    assert attribute(row("refused", []), trace("dmv", ["x"]), False, 5) is None


def test_mcnemar_and_bootstrap():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 9 + [False]
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (0, 9) and p < 0.01
    lo, hi = bootstrap_diff(a, b, samples=2000)
    assert 0 < lo <= 0.45 <= hi
    assert mcnemar(a, a) == (0, 0, 1.0)


def test_sign_flip_matches_mcnemar_on_single_samples():
    a = [1.0] * 14 + [0.0] * 4 + [1.0] * 50 + [0.0] * 132
    b = [0.0] * 14 + [1.0] * 4 + [1.0] * 50 + [0.0] * 132
    only_a, only_b, p = sign_flip(a, b)
    assert (only_a, only_b) == (14, 4) and abs(p - mcnemar([x == 1 for x in a], [x == 1 for x in b])[2]) < 0.005
    assert sign_flip(a, a) == (0, 0, 1.0)
    # means over samples: a turn that rose from 1/3 to 2/3 counts once
    assert sign_flip([1 / 3, 1.0], [2 / 3, 1.0])[:2] == (0, 1)


def test_passes_rejects_only_a_significant_and_material_guard_drop():
    gain = {"a": 0.866, "b": 0.917, "only_a": 0, "only_b": 14, "p": 0.0001}
    guard = lambda a, b, p: {"a": a, "b": b, "only_a": 6, "only_b": 0, "p": p}
    assert passes(gain, guard(0.430, 0.413, 0.033), 0.05, 0.02)       # significant, 1.7 points
    assert not passes(gain, guard(0.435, 0.390, 0.049), 0.05, 0.02)   # significant, 4.5 points
    assert passes(gain, guard(0.435, 0.390, 0.2), 0.05, 0.02)         # 4.5 points, not significant
    assert not passes(gain, guard(0.430, 0.413, 0.033), 0.05, 0.0)    # margin 0: any significant drop
    assert not passes({**gain, "p": 0.07}, guard(0.43, 0.43, 1.0), 0.05, 0.02)
