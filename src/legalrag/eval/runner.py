"""Run retrieval evaluation against queries with known relevant chunk IDs."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .metrics import (
    average_precision,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def evaluate_retriever(
    retriever: Any,
    cases: Sequence[Mapping[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate a retriever on cases containing query and relevant_ids.

    Each case must contain:
      - query: a non-empty query string
      - relevant_ids: a collection of relevant chunk IDs

    The retriever must expose search(query), returning tuples whose first
    element is the retrieved chunk ID.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not cases:
        raise ValueError("cases must not be empty")

    results = []

    for index, case in enumerate(cases):
        query = case.get("query")
        relevant_ids = case.get("relevant_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {index} has a missing or empty query")
        if relevant_ids is None:
            raise ValueError(f"Case {index} is missing relevant_ids")

        relevant = set(relevant_ids)
        if not relevant:
            raise ValueError(f"Case {index} has no relevant IDs")

        retrieved_rows = retriever.search(query)
        retrieved_ids = [str(row[0]) for row in retrieved_rows[:k]]

        results.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": sorted(str(item) for item in relevant),
            f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
            f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
            "mrr": reciprocal_rank(retrieved_ids, relevant),
            "map": average_precision(retrieved_ids, relevant),
        })

    metric_names = [f"precision@{k}", f"recall@{k}", "mrr", "map"]
    aggregate = {
        name: sum(row[name] for row in results) / len(results)
        for name in metric_names
    }

    return {
        "query_count": len(results),
        "k": k,
        "metrics": aggregate,
        "results": results,
    }
