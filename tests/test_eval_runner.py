import pytest

from legalrag.eval.runner import evaluate_retriever


class FakeRetriever:
    def search(self, query):
        rankings = {
            "query one": [("A", 0.9, {}), ("B", 0.8, {}), ("C", 0.7, {})],
            "query two": [("X", 0.9, {}), ("Y", 0.8, {})],
        }
        return rankings[query]


def test_evaluate_retriever_returns_metrics():
    cases = [
        {"query": "query one", "relevant_ids": {"A", "C"}},
        {"query": "query two", "relevant_ids": {"Y"}},
    ]

    result = evaluate_retriever(FakeRetriever(), cases, k=2)

    assert result["query_count"] == 2
    assert result["k"] == 2
    assert result["metrics"]["precision@2"] == pytest.approx(0.5)
    assert result["metrics"]["recall@2"] == pytest.approx(0.75)
    assert result["metrics"]["mrr"] == pytest.approx(0.75)
    assert result["metrics"]["map"] == pytest.approx(0.5)
    assert len(result["results"]) == 2


def test_rejects_empty_cases():
    with pytest.raises(ValueError, match="cases must not be empty"):
        evaluate_retriever(FakeRetriever(), [], k=5)


def test_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be greater than zero"):
        evaluate_retriever(
            FakeRetriever(),
            [{"query": "query one", "relevant_ids": {"A"}}],
            k=0,
        )


def test_rejects_case_without_relevant_ids():
    with pytest.raises(ValueError, match="missing relevant_ids"):
        evaluate_retriever(
            FakeRetriever(),
            [{"query": "query one"}],
            k=5,
        )


def test_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="no relevant IDs"):
        evaluate_retriever(
            FakeRetriever(),
            [{"query": "query one", "relevant_ids": set()}],
            k=5,
        )

