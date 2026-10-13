"""Full-text lookup across the three category corpora (M7).

Dispatches a chunk_id to the correct corpus by its ``category/`` prefix and
delegates to the existing ``legalrag.generation.text_store.CorpusTextStore``
(byte-offset sidecars are written to a cache dir, never inside the corpus or
index directories).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from legalrag.generation.text_store import CorpusTextStore

# category -> corpus path relative to the project root
CATEGORY_CORPORA: dict[str, str] = {
    "acts": "data/pilot/processed/acts/chunks.jsonl",
    "constitution": "data/processed/chunks.jsonl",
    "judgments": "data/pilot/processed/judgments-full/chunks.jsonl",
}


def category_of(chunk_id: str) -> str | None:
    """Return the leading ``category`` segment of a chunk id, if any."""
    if "/" in chunk_id:
        return chunk_id.split("/", 1)[0]
    return None


class MultiCorpusTextStore:
    """Read full chunk records from the corpus matching each chunk's category."""

    def __init__(
        self,
        root: str | Path,
        corpora: dict[str, str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        root = Path(root)
        corpora = corpora or CATEGORY_CORPORA
        cache_dir = Path(cache_dir) if cache_dir else root / "data/cache/text_offsets"
        self._stores: dict[str, CorpusTextStore] = {}
        for category, rel_path in corpora.items():
            self._stores[category] = CorpusTextStore(
                root / rel_path,
                offsets_path=cache_dir / f"{category}.offsets.json",
            )

    def get_many(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        for chunk_id in chunk_ids:
            category = category_of(chunk_id)
            if category is None:
                continue
            grouped.setdefault(category, []).append(chunk_id)

        found: dict[str, dict[str, Any]] = {}
        for category, ids in grouped.items():
            store = self._stores.get(category)
            if store is None:
                continue
            found.update(store.get_many(ids))
        return found

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self.get_many([chunk_id]).get(chunk_id)
