"""Retriever construction and federated (cross-category) retrieval for M7.

Each category has its own index namespace and index directory; queries are
fanned out to all three retrievers and merged by reciprocal rank (RRF). The
judgments retriever keeps its configured ``rrf_k`` (10) unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from legalrag.retrieval import HybridRetriever
from legalrag.retrieval.pipeline import _rrf_score

# category -> (config file, HybridRetriever category)
CATEGORY_CONFIG: dict[str, tuple[str, str]] = {
    "acts": ("config_acts_eval.yaml", "acts"),
    "constitution": ("config.yaml", "constitution"),
    "judgments": ("config_judgments.yaml", "judgments"),
}
CATEGORIES: tuple[str, ...] = ("acts", "constitution", "judgments")
CROSS_CATEGORY_RRF_K = 60


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_retrievers(root: str | Path, categories: tuple[str, ...] = CATEGORIES) -> dict[str, HybridRetriever]:
    """Build one ``HybridRetriever`` per category from its own config file."""
    root = Path(root)
    retrievers: dict[str, HybridRetriever] = {}
    for category in categories:
        config_file, name = CATEGORY_CONFIG[category]
        cfg = load_config(root / config_file)
        retrievers[category] = HybridRetriever(cfg, category=name)
    return retrievers


class MultiCategoryRetriever:
    """Fan out to per-category retrievers and merge results by rank."""

    def __init__(self, retrievers: dict[str, Any], rrf_k: int = CROSS_CATEGORY_RRF_K) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, question: str, top_k: int = 8) -> list[tuple[str, float, dict]]:
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        # Best native (within-category) score per category, used only to break
        # rank-RRF ties: categories use different rrf_k/boost scales, so native
        # scores are not merged directly, but a stronger match should win a tie.
        category_strength: dict[str, float] = {}
        for category, retriever in self.retrievers.items():
            results = retriever.search(question)
            if results:
                category_strength[category] = float(results[0][1])
            for rank, (chunk_id, _score, item_meta) in enumerate(results):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + _rrf_score(rank, self.rrf_k)
                merged = dict(item_meta or {})
                merged.setdefault("category", category)
                meta[chunk_id] = merged

        def _sort_key(chunk_id: str) -> tuple[float, float]:
            category = meta[chunk_id].get("category")
            return (scores[chunk_id], category_strength.get(category, 0.0))

        ordered = sorted(scores, key=_sort_key, reverse=True)[:top_k]
        return [(cid, scores[cid], meta[cid]) for cid in ordered]

    def related_cases(self, question: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        """Top judgment results, treated as related cases."""
        retriever = self.retrievers.get("judgments")
        if retriever is None:
            return []
        out: list[tuple[str, float, dict]] = []
        for chunk_id, score, item_meta in retriever.search(question)[:top_k]:
            merged = dict(item_meta or {})
            merged["category"] = "judgments"
            out.append((chunk_id, score, merged))
        return out
