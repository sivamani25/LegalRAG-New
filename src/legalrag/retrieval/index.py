"""Persistent vector-store and sparse-index primitives for LegalRAG (Milestone 2).

Design
------
- ``VectorStore`` / ``SparseIndex`` are abstract base classes so a future
  MongoDB Atlas or other backend can be added without touching retrieval code.
- ``FAISSDenseIndex`` uses ``IndexFlatIP`` (inner product).  Because all
  vectors are L2-normalised by the Embedder, IP equals cosine similarity.
- ``BM25SparseIndex`` uses ``rank_bm25.BM25Okapi``, serialised with pickle
  (accepted for M2; ``index_version`` allows future format migrations).
- ``ChunkStore`` persists the ``chunk_id → metadata`` mapping so retrieval
  results can be enriched without re-reading chunks.jsonl.
- Staleness is signalled by ``IndexStaleError`` (a ``ValueError`` subclass),
  never silently ignored.  The stored ``corpus_fingerprint`` and
  ``config_fingerprint`` drive the freshness check.
- All public ``save()`` / ``load()`` signatures accept a ``metadata`` dict
  so the backend is extensible without an API change.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Increment to invalidate all persisted indexes (e.g. after a schema change).
INDEX_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IndexStaleError(ValueError):
    """Raised when a persisted index is incompatible with current config/corpus.

    Callers should catch this and either rebuild or pass ``force_rebuild=True``
    to ``IndexBuilder``.
    """


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def corpus_fingerprint(chunks_path: str) -> str:
    """SHA-256 of *chunks_path* content — changes whenever the corpus changes."""
    h = hashlib.sha256()
    with open(chunks_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def config_fingerprint(
    embedding_model: str,
    sac_enabled: bool,
    document_prefix: str,
    index_version: str,
) -> str:
    """SHA-256 of the key config parameters that affect the stored vectors."""
    blob = json.dumps(
        {
            "embedding_model": embedding_model,
            "sac_enabled": sac_enabled,
            "document_prefix": document_prefix,
            "index_version": index_version,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _write_meta(meta_path: str, data: dict) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Index metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_meta(meta_path: str, expected: dict) -> None:
    """Raise ``IndexStaleError`` if any *expected* key differs from stored value."""
    stored = _read_meta(meta_path)
    for k, v in expected.items():
        if k in stored and stored[k] != v:
            raise IndexStaleError(
                f"Index metadata mismatch for '{k}': "
                f"expected {v!r}, stored {stored[k]!r}. "
                f"Delete the index directory or use --force-rebuild."
            )


# ---------------------------------------------------------------------------
# ChunkStore — chunk_id → metadata mapping
# ---------------------------------------------------------------------------

class ChunkStore:
    """Persistent mapping from ``chunk_id`` to non-text chunk metadata.

    ``text`` and ``original_text`` are excluded to keep the file small.
    Retrieval code looks up this store to enrich results; callers that need
    the full text re-read ``chunks.jsonl`` by offset.
    """

    _EXCLUDE = frozenset({"text", "original_text"})

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    # -- Build ----------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        for c in chunks:
            self._store[c["chunk_id"]] = {
                k: v for k, v in c.items() if k not in self._EXCLUDE
            }

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._store = json.load(f)

    # -- Lookup ----------------------------------------------------------------

    def get(self, chunk_id: str) -> dict | None:
        return self._store.get(chunk_id)

    def __len__(self) -> int:
        return len(self._store)

    def chunk_ids(self) -> list[str]:
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """Abstract dense vector store (FAISS for M2; MongoDB Atlas later)."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """Add *embeddings* with corresponding *doc_ids*."""

    @abstractmethod
    def save(self, path_prefix: str, metadata: dict | None = None) -> None:
        """Persist to disk under *path_prefix*.{ext}."""

    @abstractmethod
    def load(self, path_prefix: str, expected_metadata: dict | None = None) -> None:
        """Load from disk, validating *expected_metadata* if given."""

    @property
    @abstractmethod
    def doc_ids(self) -> list[str]:
        """Ordered list of document IDs matching index positions."""


class SparseIndex(ABC):
    """Abstract sparse index (BM25 for M2)."""

    @abstractmethod
    def add(self, texts: list[str], doc_ids: list[str]) -> None:
        """Build from *texts* with corresponding *doc_ids*."""

    @abstractmethod
    def save(self, path_prefix: str, metadata: dict | None = None) -> None:
        """Persist to disk."""

    @abstractmethod
    def load(self, path_prefix: str, expected_metadata: dict | None = None) -> None:
        """Load from disk, validating *expected_metadata* if given."""

    @property
    @abstractmethod
    def doc_ids(self) -> list[str]:
        """Ordered list of document IDs."""


# ---------------------------------------------------------------------------
# FAISS dense index
# ---------------------------------------------------------------------------

class FAISSDenseIndex(VectorStore):
    """Dense vector index backed by ``faiss.IndexFlatIP``.

    Inner-product over L2-normalised vectors equals cosine similarity.
    Normalisation must be applied by the ``Embedder``; this class only
    casts to ``float32`` as required by FAISS.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._doc_ids: list[str] = []

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        if len(embeddings) != len(doc_ids):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) != doc_ids length ({len(doc_ids)})"
            )
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._doc_ids.extend(doc_ids)

    def save(self, path_prefix: str, metadata: dict | None = None) -> None:
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}_ids.json", "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f)
        meta = (metadata or {}).copy()
        meta.update({"chunk_count": len(self._doc_ids), "index_version": INDEX_VERSION})
        _write_meta(f"{path_prefix}_meta.json", meta)

    def load(self, path_prefix: str, expected_metadata: dict | None = None) -> None:
        if expected_metadata:
            _check_meta(f"{path_prefix}_meta.json", expected_metadata)
        self.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}_ids.json", "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)

    def read_meta(self, path_prefix: str) -> dict:
        return _read_meta(f"{path_prefix}_meta.json")


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------

class BM25SparseIndex(SparseIndex):
    """Sparse index backed by ``rank_bm25.BM25Okapi``.

    Serialised via pickle for M2.  The ``index_version`` field in metadata
    allows future format migrations without a separate version flag.
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def add(self, texts: list[str], doc_ids: list[str]) -> None:
        if len(texts) != len(doc_ids):
            raise ValueError(
                f"Texts length ({len(texts)}) != doc_ids length ({len(doc_ids)})"
            )
        tokenised = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenised)
        self._doc_ids = list(doc_ids)

    def save(self, path_prefix: str, metadata: dict | None = None) -> None:
        with open(f"{path_prefix}_bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self._doc_ids}, f)
        meta = (metadata or {}).copy()
        meta.update({"chunk_count": len(self._doc_ids), "index_version": INDEX_VERSION})
        _write_meta(f"{path_prefix}_meta.json", meta)

    def load(self, path_prefix: str, expected_metadata: dict | None = None) -> None:
        if expected_metadata:
            _check_meta(f"{path_prefix}_meta.json", expected_metadata)
        with open(f"{path_prefix}_bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self._doc_ids = data["doc_ids"]

    def read_meta(self, path_prefix: str) -> dict:
        return _read_meta(f"{path_prefix}_meta.json")
