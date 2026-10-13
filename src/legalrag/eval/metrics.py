"""Ranking metrics for LegalRAG retrieval evaluation."""

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved IDs that are relevant.

    Standard precision@k divides by *k* (the requested cutoff), so a query that
    returns fewer than *k* results is not rewarded for the short list.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return sum(item in relevant for item in top_k) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all known relevant IDs found in the top-k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Average precision over the retrieved ranking."""
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()

    for rank, item in enumerate(retrieved, start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)
