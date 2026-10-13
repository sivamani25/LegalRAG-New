import pytest

from legalrag.eval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision,
)


def test_precision_at_k():
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, 2) == pytest.approx(0.5)


def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], {"A", "C"}, 2) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert reciprocal_rank(["A", "B", "C"], {"C"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_result():
    assert reciprocal_rank(["A", "B"], {"C"}) == 0.0


def test_average_precision():
    assert average_precision(["A", "B", "C"], {"A", "C"}) == pytest.approx(5 / 6)


def test_empty_relevant_set():
    assert recall_at_k(["A"], set(), 1) == 0.0
    assert average_precision(["A"], set()) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, -1)


def test_precision_at_k_fewer_results_than_k():
    # Standard precision@k divides by k, not by the number of returned items.
    assert precision_at_k(["A"], {"A"}, 5) == pytest.approx(0.2)
    assert precision_at_k(["A", "B"], {"A", "B"}, 4) == pytest.approx(0.5)
    assert precision_at_k(["A"], {"Z"}, 5) == 0.0
    assert precision_at_k([], {"A"}, 5) == 0.0
