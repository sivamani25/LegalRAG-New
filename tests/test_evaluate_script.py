"""Lightweight tests for scripts/evaluate_retrieval.py (no model / no indexes)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_retrieval as ev  # noqa: E402


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, query):
        return self.table.get(query, [])


def test_evaluate_category_structure_and_values():
    cases = [
        {"category": "acts", "query": "q1", "relevant_ids": ["A"]},
        {"category": "acts", "query": "q2", "relevant_ids": ["Z"]},
    ]
    retriever = FakeRetriever({
        "q1": [("A", 1.0, {}), ("B", 0.9, {})],
        "q2": [("X", 1.0, {}), ("Y", 0.9, {})],
    })

    out = ev.evaluate_category(retriever, cases, [1, 3])

    assert out["num_cases"] == 2
    assert set(out["metrics_by_k"]) == {"1", "3"}
    # p@1: q1 -> 1.0, q2 -> 0.0  => macro 0.5
    assert out["metrics_by_k"]["1"]["precision"] == pytest.approx(0.5)
    # p@3: q1 -> 1/3, q2 -> 0    => macro 1/6
    assert out["metrics_by_k"]["3"]["precision"] == pytest.approx(1 / 6)
    assert out["zero_hit_queries"] == ["q2"]
    assert out["per_query"][0]["retrieved_ids"] == ["A", "B"]
    assert out["per_query"][0]["relevant_ids"] == ["A"]


def test_macro_overall_averages_categories():
    cats = {
        "acts": {"metrics_by_k": {"1": {"precision": 1.0, "recall": 1.0, "mrr": 1.0, "map": 1.0}}},
        "judgments": {"metrics_by_k": {"1": {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "map": 0.0}}},
    }
    overall = ev.macro_overall(cats, [1])
    assert overall["1"]["precision"] == pytest.approx(0.5)
    assert overall["1"]["map"] == pytest.approx(0.5)


def test_build_report_skips_missing_category_and_reports_errors():
    cases = [{"category": "acts", "query": "q1", "relevant_ids": ["A"]}]

    def factory(cfg, category):
        raise FileNotFoundError("index missing for test")

    report = ev.build_report(
        cases, [1],
        config_map={"acts": ("config_acts_eval.yaml", "acts")},
        retriever_factory=factory,
    )
    assert report["categories"]["acts"]["error"].startswith("FileNotFoundError")
    assert "constitution" in report["categories"]
    assert report["categories"]["constitution"]["num_cases"] == 0
