"""End-to-end gate behaviour, including the regression scenario the tool exists for."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from rageval.dataset import DatasetError, load_golden_set  # noqa: E402
from rageval.runner import gate, run  # noqa: E402
import toy_rag  # noqa: E402

GOLDEN = ROOT / "examples" / "golden.jsonl"


@pytest.fixture()
def cases():
    return load_golden_set(GOLDEN)


def test_golden_set_loads_and_validates(cases):
    assert len(cases) == 4
    assert cases[0].id == "refund-timeframe"


def test_healthy_system_passes_every_case(cases):
    report = run(cases, toy_rag.answer)
    assert not report.failures
    assert report.overall > 0.9


def test_first_run_records_rather_than_blocks(cases):
    report = run(cases, toy_rag.answer)
    decision = gate(report, baseline=None)
    assert decision.ok


def test_regression_is_caught_against_baseline(cases):
    healthy = run(cases, toy_rag.answer)
    baseline = healthy.to_dict()

    broken = run(cases, toy_rag.broken_answer)
    decision = gate(broken, baseline)

    assert not decision.ok
    joined = " ".join(decision.reasons)
    assert "refund-timeframe" in joined, "the newly failing case must be named"


def test_dropped_fact_fails_the_case_even_with_high_overlap(cases):
    broken = run(cases, toy_rag.broken_answer)
    failing = {r.id for r in broken.failures}
    assert "refund-timeframe" in failing
    result = next(r for r in broken.results if r.id == "refund-timeframe")
    assert result.scores["must_include"] < 1.0, "missing '5 business days' is the cause"


def test_tolerance_absorbs_noise_but_not_regression(cases):
    healthy = run(cases, toy_rag.answer)
    baseline = healthy.to_dict()
    decision = gate(run(cases, toy_rag.answer), baseline, tolerance=0.02)
    assert decision.ok, "identical run must never trip the gate"


class TestDatasetValidation:
    def test_duplicate_ids_are_rejected(self, tmp_path):
        p = tmp_path / "dup.jsonl"
        p.write_text(
            '{"id": "a", "question": "q1"}\n{"id": "a", "question": "q2"}\n'
        )
        with pytest.raises(DatasetError, match="duplicate case id"):
            load_golden_set(p)

    def test_missing_question_names_the_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"id": "a", "question": "ok"}\n{"id": "b"}\n')
        with pytest.raises(DatasetError, match="line 2"):
            load_golden_set(p)

    def test_string_where_list_expected_gets_a_helpful_error(self, tmp_path):
        p = tmp_path / "typo.jsonl"
        p.write_text('{"id": "a", "question": "q", "must_include": "5 days"}\n')
        with pytest.raises(DatasetError, match="Did you mean"):
            load_golden_set(p)

    def test_empty_set_is_an_error(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("# only a comment\n")
        with pytest.raises(DatasetError, match="no cases"):
            load_golden_set(p)
