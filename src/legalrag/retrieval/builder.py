"""Index builder orchestration for LegalRAG (Milestone 2).

Design
------
- Reads ``chunks.jsonl`` (M1 output) and builds dense/sparse indexes.
- Builds a per-category index AND a global merged index (``_merged_``).
- Skips building if the on-disk index is fresh (fingerprints match).
- Uses ``chunk["text"]`` exactly as-is (SAC is already applied by M1).
- Injects ``NOMIC_DOCUMENT_PREFIX`` into the embedder for Nomic models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .embedding import Embedder, NOMIC_DOCUMENT_PREFIX
from .index import (
    BM25SparseIndex,
    ChunkStore,
    FAISSDenseIndex,
    INDEX_VERSION,
    IndexStaleError,
    config_fingerprint,
    corpus_fingerprint,
)

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Orchestrates building dense and sparse indexes from chunks.jsonl."""

    #: Minimum number of chunks required to build a production index.
    #: A real M2 corpus will always have hundreds of chunks; this guard prevents
    #: diagnostic runs from silently overwriting production indexes.
    MIN_CORPUS_SIZE: int = 10

    def __init__(self, config: dict) -> None:
        self.config = config
        self.retrieval_cfg = config.get("retrieval", {})
        self.ingest_cfg = config.get("ingest", {})
        self.chunking_cfg = config.get("chunking", {})

        self.model_name = self.retrieval_cfg.get("embedding_model", "nomic-embed-text")
        self.normalize = self.retrieval_cfg.get("embedding_normalize", True)
        self.batch_size: int = int(self.retrieval_cfg.get("embedding_batch_size", 32))
        self.index_dir = Path(self.retrieval_cfg.get("index_dir", "data/index"))
        self.processed_dir = Path(self.ingest_cfg.get("processed_dir", "data/processed"))
        self.chunks_file = self.ingest_cfg.get("chunks_file", "chunks.jsonl")
        self.use_sac = bool(self.chunking_cfg.get("use_sac", True))

        self.embedder: Embedder | None = None  # Lazy init

        # Cache of already-computed embeddings keyed by ordered tuple of chunk_ids.
        # Lets _build_one reuse vectors for the merged index when its corpus is
        # identical to an already-built per-category corpus, avoiding a duplicate
        # forward pass through the (expensive) Nomic model.
        # Entries are released after load_or_build() finishes (FIX 3).
        self._embedding_cache: dict[tuple, np.ndarray] = {}

    def _init_embedder(self) -> None:
        if self.embedder is None:
            self.embedder = Embedder.from_config(self.retrieval_cfg)

    def load_or_build(self, force_rebuild: bool = False) -> None:
        """Build indexes for all categories and a global merged index.

        If ``force_rebuild=False``, skips if the index is fresh according to
        fingerprints.
        """
        chunks_path = self.processed_dir / self.chunks_file
        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. Run ingestion first."
            )

        # 1. Compute fingerprints
        corp_fp = corpus_fingerprint(str(chunks_path))
        doc_prefix = NOMIC_DOCUMENT_PREFIX if "nomic" in self.model_name else ""
        conf_fp = config_fingerprint(
            embedding_model=self.model_name,
            sac_enabled=self.use_sac,
            document_prefix=doc_prefix,
            index_version=INDEX_VERSION,
        )

        logger.info("Corpus fingerprint: %s", corp_fp[:8])
        logger.info("Config fingerprint: %s", conf_fp[:8])

        # 2. Read chunks and group by category
        categories: dict[str, list[dict]] = {}
        all_chunks: list[dict] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                cat = chunk.get("category", "")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(chunk)
                all_chunks.append(chunk)

        # Safety guard: refuse to build from a suspiciously small corpus.
        # This prevents diagnostic/test runs from silently overwriting production
        # indexes with truncated test data.
        if len(all_chunks) < self.MIN_CORPUS_SIZE:
            raise RuntimeError(
                f"Corpus too small to build production indexes: "
                f"{len(all_chunks)} chunks loaded from {chunks_path} "
                f"(minimum required: {self.MIN_CORPUS_SIZE}). "
                f"Ensure chunks.jsonl contains the full ingested corpus."
            )
        logger.info("Loaded %d chunks from %s", len(all_chunks), chunks_path)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 3. Write ChunkStore (metadata mapping)
        store = ChunkStore()
        store.add_chunks(all_chunks)
        store_path = self.index_dir / "chunk_store.json"
        store.save(str(store_path))
        logger.info("Saved ChunkStore with %d items to %s", len(store), store_path)

        # 4. Build per-category indexes
        for cat, chunks in categories.items():
            if not cat:
                cat = "unknown"
            self._build_one(
                name=cat,
                chunks=chunks,
                corp_fp=corp_fp,
                conf_fp=conf_fp,
                doc_prefix=doc_prefix,
                force_rebuild=force_rebuild,
            )

        # 5. Build global merged index
        if all_chunks:
            self._build_one(
                name="_merged",
                chunks=all_chunks,
                corp_fp=corp_fp,
                conf_fp=conf_fp,
                doc_prefix=doc_prefix,
                force_rebuild=force_rebuild,
            )

        # FIX 3: Release cached embedding arrays now that all indexes are built.
        # The cache served its purpose (avoiding a duplicate Nomic forward pass for
        # the merged index); holding on to it would waste memory unnecessarily.
        self._embedding_cache.clear()

    def _build_one(
        self,
        name: str,
        chunks: list[dict],
        corp_fp: str,
        conf_fp: str,
        doc_prefix: str,
        force_rebuild: bool,
    ) -> None:
        """Build or skip a single index namespace.

        Embeddings are stored in ``self._embedding_cache`` keyed by the
        ordered tuple of ``chunk_id`` values.  When a subsequent call (e.g.
        the global ``_merged`` index) presents the exact same ordered corpus
        as a previously built per-category index, the cached ``np.ndarray``
        is reused directly — no second forward pass through the model.
        """
        sac_suffix = "sac" if self.use_sac else "no_sac"
        namespace = f"{name}_{self.model_name}_{sac_suffix}"
        prefix = self.index_dir / namespace
        prefix_str = str(prefix)

        expected_meta = {
            "corpus_fingerprint": corp_fp,
            "config_fingerprint": conf_fp,
            "embedding_model": self.model_name,
            "sac_enabled": self.use_sac,
            "document_prefix": doc_prefix,
            "index_version": INDEX_VERSION,
        }

        # Check freshness — skip rebuild if on-disk index matches fingerprints.
        if not force_rebuild:
            dense_meta = prefix_str + "_dense_meta.json"
            if Path(dense_meta).exists():
                try:
                    with open(dense_meta, "r", encoding="utf-8") as f:
                        stored = json.load(f)
                    stale = any(
                        k in stored and stored[k] != v
                        for k, v in expected_meta.items()
                    )
                    if not stale and stored.get("chunk_count") == len(chunks):
                        logger.info("[%s] Index is fresh, skipping rebuild.", name)
                        return
                except Exception:
                    pass  # Any read/parse error → fall through to rebuild

        logger.info("[%s] Building index for %d chunks...", name, len(chunks))

        texts = [c["text"] for c in chunks]
        doc_ids = [c["chunk_id"] for c in chunks]
        cache_key = tuple(doc_ids)

        # FIX 2: Reuse already-computed embeddings when the corpus is identical
        # to one built earlier in this session (e.g. merged == single category).
        if cache_key in self._embedding_cache:
            logger.info(
                "[%s] Reusing cached embeddings from a previous index "
                "(corpus is identical — no second embedding pass).",
                name,
            )
            embeddings = self._embedding_cache[cache_key]
        else:
            # FIX 1: pass configured batch_size so sentence_transformers
            # processes texts in controlled-size forward passes.
            self._init_embedder()
            assert self.embedder is not None
            
            # Avoid tokenizing all chunks at once, which causes massive memory spikes
            # Process in outer blocks of 1000
            import numpy as np
            all_embeddings = []
            outer_batch = 5000
            for i in range(0, len(texts), outer_batch):
                batch_texts = texts[i:i+outer_batch]
                batch_emb = self.embedder.encode(
                    batch_texts,
                    normalize=self.normalize,
                    prompt_prefix=doc_prefix,
                    batch_size=self.batch_size,
                )
                all_embeddings.append(batch_emb)
            
            embeddings = np.vstack(all_embeddings)
            
            # Store for potential reuse by the merged index.
            self._embedding_cache[cache_key] = embeddings

        self._init_embedder()
        assert self.embedder is not None

        dense = FAISSDenseIndex(self.embedder.dimension)
        dense.add(embeddings, doc_ids)
        dense.save(prefix_str + "_dense", metadata=expected_meta)

        sparse = BM25SparseIndex()
        sparse.add(texts, doc_ids)
        sparse.save(prefix_str + "_sparse", metadata=expected_meta)

        logger.info("[%s] Saved dense and sparse indexes to %s", name, namespace)

