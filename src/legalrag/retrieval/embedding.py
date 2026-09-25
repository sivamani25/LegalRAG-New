"""Embedding interface and implementations for LegalRAG (Milestone 2).

Design principles
-----------------
- ``Embedder`` is the only interface the rest of the system touches.
- ``SentenceTransformerEmbedder`` is the concrete M2 implementation.
- Normalization is *always* the caller's explicit decision (``normalize=`` kwarg).
  It is never silently applied in multiple layers.
- Prompt prefixes (Nomic "search_document:" / "search_query:") are injected
  at encode-time so the same Embedder instance can serve both indexing and
  query paths with different prefixes.
- The model-name registry maps short config names to HuggingFace hub identifiers;
  unknown names are passed through unchanged so any HF model works.
- ``from_config`` is the canonical factory for building from a config dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Model-name registry: short config name → HuggingFace hub identifier
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, str] = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
}

# Default task-type prefixes for Nomic Embed Text v1.5 (and compatible models).
# These are written into index metadata so M3 can assert the correct prefix.
NOMIC_DOCUMENT_PREFIX: str = "search_document: "
NOMIC_QUERY_PREFIX: str = "search_query: "


def resolve_hf_name(model_name: str) -> str:
    """Return the HuggingFace hub identifier for *model_name*.

    Falls back to *model_name* itself when not in the registry, allowing any
    HF-compatible model to be used by specifying its full hub path in config.
    """
    return _MODEL_REGISTRY.get(model_name, model_name)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Abstract embedding interface.

    Implementations must be swappable without changing the indexing or
    retrieval code.  The only contract is ``encode()`` and ``dimension``.
    """

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        prompt_prefix: str = "",
    ) -> np.ndarray:
        """Encode *texts* into dense vectors.

        Args:
            texts:         Input strings.
            normalize:     If ``True``, L2-normalise each vector before
                           returning.  Required for FAISS IndexFlatIP to
                           behave as cosine similarity.  Applied *once* here
                           and nowhere else.
            prompt_prefix: Prepended to every text before encoding.
                           Use ``NOMIC_DOCUMENT_PREFIX`` at index time and
                           ``NOMIC_QUERY_PREFIX`` at query time.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), self.dimension)``,
            dtype ``float32``.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short config-level model name (e.g. ``"nomic-embed-text"``)."""

    @classmethod
    def from_config(cls, retrieval_cfg: dict) -> "Embedder":
        """Build the default Embedder from the ``retrieval`` config section.

        Args:
            retrieval_cfg: The ``retrieval:`` subsection of ``config.yaml``.

        Returns:
            A concrete ``Embedder`` instance ready to use.
        """
        model_name = retrieval_cfg["embedding_model"]
        return SentenceTransformerEmbedder(model_name)


# ---------------------------------------------------------------------------
# Concrete implementation — sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by ``sentence-transformers``.

    Supports any model in the HuggingFace registry.  Uses the short config
    name ``"nomic-embed-text"`` to load ``nomic-ai/nomic-embed-text-v1.5``
    with ``trust_remote_code=True`` as required.
    """

    def __init__(self, model_name: str) -> None:
        """
        Args:
            model_name: Short config-level name (e.g. ``"nomic-embed-text"``)
                        or a full HuggingFace hub path.
        """
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model_name = model_name
        hf_name = resolve_hf_name(model_name)
        self.model = SentenceTransformer(hf_name, trust_remote_code=True)

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        prompt_prefix: str = "",
    ) -> np.ndarray:
        """Encode *texts*, optionally prefixing and normalising.

        The prefix is applied here so that the same model instance can be
        used for both document indexing (``NOMIC_DOCUMENT_PREFIX``) and
        query encoding (``NOMIC_QUERY_PREFIX``) in M3.

        Normalisation is applied exactly once at this layer.  FAISS and BM25
        must not re-normalise.
        """
        if prompt_prefix:
            texts = [prompt_prefix + t for t in texts]
        embeddings: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
