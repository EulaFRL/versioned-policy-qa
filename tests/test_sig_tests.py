import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "sig_tests", Path(__file__).parents[1] / "scripts/sig_tests.py")
sig_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sig_tests)


def test_model_label_reads_run_summary(tmp_path):
    run = tmp_path / "ecfr_test.jsonl"
    run.with_suffix(".summary.json").write_text(json.dumps({"answer_model": "deepseek/deepseek-v4-flash"}))
    assert sig_tests.model_label(run) == "DeepSeek V4 Flash"


def test_model_label_marks_missing_summary_as_legacy(tmp_path):
    assert sig_tests.model_label(tmp_path / "old_run.jsonl") == "legacy (answer_model unavailable)"


def test_model_label_marks_summary_without_model_as_legacy(tmp_path):
    run = tmp_path / "old_run.jsonl"
    run.with_suffix(".summary.json").write_text("{}")
    assert sig_tests.model_label(run) == "legacy (answer_model unavailable)"
