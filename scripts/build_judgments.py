"""Canonical production builder for the judgment index.

Builds the dense (FAISS, Nomic 768-d) and sparse (BM25) indexes for the
judgments corpus **only**.  It never reads or writes the Constitution or
``_merged_`` indexes.

Design
------
Two independent, memory-bounded phases:

Phase A (dense)
    Stream ``chunks.jsonl`` once, embedding in bounded batches and writing a
    checkpoint (FAISS dense index + ids + metadata + chunk store) whenever
    processing crosses another ``checkpoint_interval`` chunk boundary
    (bucket-based, so it does not depend on divisibility by the batch size).
    The embedding model is released at the end of the phase.

Phase B (BM25)
    Runs with the Nomic model released.  Streams the corpus again, tokenising
    as it goes, then builds and saves the BM25 artifact only after it has
    completed successfully.  An interruption here never touches the completed
    dense index.

Resume safety
-------------
Before resuming, the stored dense ids are validated against the *first N*
``chunk_id`` values of the current corpus.  If they do not match exactly (or
the config/corpus fingerprints differ), the build refuses to resume and
requires a clean rebuild — it never silently produces duplicate or misaligned
vectors.

Usage
-----
    python scripts/build_judgments.py                       # full corpus
    python scripts/build_judgments.py config.yaml 1000      # first 1000 chunks
    python scripts/build_judgments.py --phase dense         # dense only
    python scripts/build_judgments.py --phase bm25          # BM25 only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import yaml

from legalrag.retrieval.embedding import Embedder, NOMIC_DOCUMENT_PREFIX
from legalrag.retrieval.index import (
    BM25SparseIndex,
    ChunkStore,
    FAISSDenseIndex,
    INDEX_VERSION,
    config_fingerprint,
    corpus_fingerprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_judgments")

DEFAULT_CHECKPOINT_INTERVAL = 5000
DEFAULT_PROGRESS_INTERVAL = 2000


class ResumeMismatchError(RuntimeError):
    """Raised when an on-disk judgment index cannot be safely resumed."""


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _iter_chunks(chunks_path):
    """Yield chunk dicts from *chunks_path*, skipping blank lines."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _count_chunks(chunks_path) -> int:
    """Count non-blank lines in *chunks_path* without parsing JSON."""
    total = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


# ---------------------------------------------------------------------------
# Context / configuration
# ---------------------------------------------------------------------------

def _build_context(cfg: dict) -> dict:
    retrieval_cfg = cfg.get("retrieval", {})
    ingest_cfg = cfg.get("ingest", {})
    chunking_cfg = cfg.get("chunking", {})

    model_name = retrieval_cfg.get("embedding_model", "nomic-embed-text")
    normalize = retrieval_cfg.get("embedding_normalize", True)
    batch_size = int(retrieval_cfg.get("embedding_batch_size", 32))
    use_sac = bool(chunking_cfg.get("use_sac", True))

    index_dir = Path(retrieval_cfg.get("index_dir", "data/index")) / "judgments"
    index_dir.mkdir(parents=True, exist_ok=True)

    base_processed = Path(ingest_cfg.get("processed_dir", "data/processed"))
    processed_dir = base_processed
    if not processed_dir.name.endswith("judgments-full"):
        processed_dir = processed_dir / "judgments-full"
    if not processed_dir.exists():
        processed_dir = Path("data/pilot/processed/judgments-full")

    chunks_file = ingest_cfg.get("chunks_file", "chunks.jsonl")
    chunks_path = processed_dir / chunks_file
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    doc_prefix = NOMIC_DOCUMENT_PREFIX if "nomic" in model_name else ""
    corp_fp = corpus_fingerprint(str(chunks_path))
    conf_fp = config_fingerprint(model_name, use_sac, doc_prefix, INDEX_VERSION)

    expected_meta = {
        "corpus_fingerprint": corp_fp,
        "config_fingerprint": conf_fp,
        "embedding_model": model_name,
        "sac_enabled": use_sac,
        "document_prefix": doc_prefix,
        "index_version": INDEX_VERSION,
    }

    sac_suffix = "sac" if use_sac else "no_sac"
    namespace = f"judgments_{model_name}_{sac_suffix}"
    prefix = index_dir / namespace

    return {
        "retrieval_cfg": retrieval_cfg,
        "index_dir": index_dir,
        "processed_dir": processed_dir,
        "chunks_path": chunks_path,
        "model_name": model_name,
        "normalize": normalize,
        "batch_size": batch_size,
        "use_sac": use_sac,
        "doc_prefix": doc_prefix,
        "expected_meta": expected_meta,
        "prefix_str": str(prefix),
        "store_path": index_dir / "chunk_store.json",
        "total": _count_chunks(chunks_path),
    }


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------

def _validate_resume(prefix_str: str, chunks_path, expected_meta: dict) -> int:
    """Return the number of already-embedded vectors if resume is safe.

    Raises ``ResumeMismatchError`` when the on-disk index cannot be trusted
    (missing metadata, config/corpus changed, or the stored ids are not an
    exact prefix of the current corpus).  Returns ``0`` when there is no
    checkpoint to resume from.
    """
    dense_meta = prefix_str + "_dense_meta.json"
    dense_faiss = prefix_str + "_dense.faiss"
    dense_ids = prefix_str + "_dense_ids.json"

    if not (Path(dense_meta).exists() and Path(dense_faiss).exists()
            and Path(dense_ids).exists()):
        return 0

    try:
        with open(dense_meta, "r", encoding="utf-8") as f:
            stored_meta = json.load(f)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear error
        raise ResumeMismatchError(
            f"Existing dense metadata is unreadable ({dense_meta}): {exc}. "
            f"Clean rebuild required (--force-rebuild)."
        ) from exc

    for key, value in expected_meta.items():
        if key in stored_meta and stored_meta[key] != value:
            raise ResumeMismatchError(
                f"Existing judgment index does not match the current "
                f"config/corpus (field '{key}': stored={stored_meta[key]!r}, "
                f"expected={value!r}). Refusing to resume; clean rebuild "
                f"required (--force-rebuild)."
            )

    with open(dense_ids, "r", encoding="utf-8") as f:
        stored_ids = json.load(f)
    n = len(stored_ids)
    if n == 0:
        return 0

    seen = 0
    for chunk in _iter_chunks(chunks_path):
        if seen >= n:
            break
        cid = chunk.get("chunk_id")
        if cid != stored_ids[seen]:
            raise ResumeMismatchError(
                f"Resume mismatch at position {seen}: stored dense id "
                f"{stored_ids[seen]!r} != corpus chunk_id {cid!r}. Refusing "
                f"to resume to avoid duplicate/misaligned vectors; clean "
                f"rebuild required (--force-rebuild)."
            )
        seen += 1

    if seen < n:
        raise ResumeMismatchError(
            f"Corpus has only {seen} chunks but the existing dense index "
            f"holds {n} vectors. Refusing to resume; clean rebuild required "
            f"(--force-rebuild)."
        )

    logger.info("[dense] Resume validated: %d stored vectors match the first "
                "%d corpus chunks.", n, n)
    return n


# ---------------------------------------------------------------------------
# Progress + checkpoint helpers
# ---------------------------------------------------------------------------

def _log_progress(phase: str, done: int, total: int, phase_start: float,
                  extra: str, checkpoint: bool = False) -> None:
    elapsed = time.monotonic() - phase_start
    pct = (done / total * 100.0) if total else 0.0
    rate = (done / elapsed) if elapsed > 0 else 0.0
    eta = ((total - done) / rate) if rate > 0 else float("inf")
    logger.info(
        "[%s] %d/%d chunks (%.1f%%) | elapsed %.0fs | %.0f chunks/s | "
        "ETA %.0fs | %s | %s",
        phase, done, total, pct, elapsed, rate, eta, extra,
        "CHECKPOINT SAVED" if checkpoint else "in progress",
    )


def _checkpoint(dense: FAISSDenseIndex, store: ChunkStore, store_path,
                prefix_str: str, expected_meta: dict, store_dirty: bool,
                final: bool = False) -> None:
    """Persist the dense index (and chunk store) atomically enough to resume."""
    dense.save(prefix_str + "_dense", metadata=expected_meta)
    if store_dirty or final:
        store.save(str(store_path))
    logger.debug("[dense] checkpoint written (vectors=%d, store_dirty=%s, "
                 "final=%s)", len(dense.doc_ids), store_dirty, final)


# ---------------------------------------------------------------------------
# Phase A — dense
# ---------------------------------------------------------------------------

def _build_dense(ctx: dict, force_rebuild: bool, checkpoint_interval: int,
                 max_chunks=None, progress_interval: int = DEFAULT_PROGRESS_INTERVAL) -> int:
    prefix_str = ctx["prefix_str"]
    chunks_path = ctx["chunks_path"]
    expected_meta = ctx["expected_meta"]
    batch_size = ctx["batch_size"]
    normalize = ctx["normalize"]
    doc_prefix = ctx["doc_prefix"]
    total = ctx["total"]
    store_path = ctx["store_path"]

    phase_start = time.monotonic()
    embedder = Embedder.from_config(ctx["retrieval_cfg"])
    try:
        logger.info("[dense] Phase A starting | total=%d | batch_size=%d | "
                    "checkpoint_interval=%d", total, batch_size, checkpoint_interval)

        dense = FAISSDenseIndex(embedder.dimension)

        if force_rebuild:
            start_offset = 0
            logger.info("[dense] --force-rebuild: starting from zero vectors.")
        else:
            start_offset = _validate_resume(prefix_str, chunks_path, expected_meta)
            if start_offset:
                dense.load(prefix_str + "_dense", expected_metadata=expected_meta)
                logger.info("[dense] Resuming from %d existing vectors.", start_offset)
            else:
                logger.info("[dense] No valid checkpoint; starting from zero vectors.")

        store = ChunkStore()
        store_dirty = False

        batch_texts: list[str] = []
        batch_ids: list[str] = []
        embedded = start_offset
        lines_seen = 0
        last_bucket = embedded // checkpoint_interval
        last_progress = embedded

        for chunk in _iter_chunks(chunks_path):
            lines_seen += 1

            cid = chunk["chunk_id"]
            if cid not in store._store:
                store._store[cid] = {
                    k: v for k, v in chunk.items() if k not in ChunkStore._EXCLUDE
                }
                store_dirty = True

            if lines_seen > start_offset:
                batch_texts.append(chunk["text"])
                batch_ids.append(cid)

                if len(batch_texts) >= batch_size:
                    embeddings = embedder.encode(
                        batch_texts,
                        normalize=normalize,
                        prompt_prefix=doc_prefix,
                        batch_size=batch_size,
                    )
                    dense.add(embeddings, batch_ids)
                    embedded += len(batch_texts)
                    batch_texts = []
                    batch_ids = []

                    if embedded - last_progress >= progress_interval:
                        _log_progress("dense", embedded, total, phase_start,
                                      extra=f"FAISS={len(dense.doc_ids)} "
                                            f"batch={batch_size} streaming")
                        last_progress = embedded

                    bucket = embedded // checkpoint_interval
                    if bucket > last_bucket:
                        _checkpoint(dense, store, store_path, prefix_str,
                                    expected_meta, store_dirty, final=False)
                        store_dirty = False
                        last_bucket = bucket
                        _log_progress("dense", embedded, total, phase_start,
                                      extra=f"FAISS={len(dense.doc_ids)} "
                                            f"batch={batch_size} streaming",
                                      checkpoint=True)

            if max_chunks and lines_seen >= max_chunks:
                break

        if batch_texts:
            embeddings = embedder.encode(
                batch_texts,
                normalize=normalize,
                prompt_prefix=doc_prefix,
                batch_size=batch_size,
            )
            dense.add(embeddings, batch_ids)
            embedded += len(batch_texts)
            batch_texts = []
            batch_ids = []

        _checkpoint(dense, store, store_path, prefix_str, expected_meta,
                    store_dirty, final=True)
        logger.info("[dense] Phase A complete: %d vectors (dim=%d).",
                    len(dense.doc_ids), embedder.dimension)
        return len(dense.doc_ids)
    finally:
        del embedder
        gc.collect()
        logger.info("[dense] Embedding model released.")


# ---------------------------------------------------------------------------
# Phase B — BM25
# ---------------------------------------------------------------------------

def _build_bm25(ctx: dict, max_chunks=None,
                progress_interval: int = DEFAULT_PROGRESS_INTERVAL) -> int:
    from rank_bm25 import BM25Okapi

    prefix_str = ctx["prefix_str"]
    chunks_path = ctx["chunks_path"]
    expected_meta = ctx["expected_meta"]
    total = ctx["total"]

    phase_start = time.monotonic()
    logger.info("[bm25] Phase B starting | total=%d | model released | "
                "tokenising stream", total)

    tokenised: list[list[str]] = []
    doc_ids: list[str] = []
    lines_seen = 0
    last_progress = 0

    for chunk in _iter_chunks(chunks_path):
        lines_seen += 1
        tokenised.append(chunk["text"].lower().split())
        doc_ids.append(chunk["chunk_id"])

        if lines_seen - last_progress >= progress_interval:
            _log_progress("bm25", lines_seen, total, phase_start,
                          extra=f"docs={len(doc_ids)} streaming")
            last_progress = lines_seen

        if max_chunks and lines_seen >= max_chunks:
            break

    bm25 = BM25Okapi(tokenised)

    sparse = BM25SparseIndex()
    sparse.bm25 = bm25
    sparse._doc_ids = list(doc_ids)
    sparse.save(prefix_str + "_sparse", metadata=expected_meta)

    logger.info("[bm25] Phase B complete: %d documents saved.", len(sparse.doc_ids))
    return len(sparse.doc_ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml", max_chunks=None, phase: str = "all",
         force_rebuild: bool = False,
         checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL) -> None:
    """Build the judgment index.

    Args:
        config_path: Path to config.yaml.
        max_chunks: Process at most this many corpus chunks (diagnostics/tests).
        phase: ``"all"`` | ``"dense"`` | ``"bm25"``.
        force_rebuild: Ignore any existing checkpoint and rebuild dense from 0.
        checkpoint_interval: Chunks between dense checkpoints.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ctx = _build_context(cfg)
    logger.info("Judgment corpus: %s", ctx["chunks_path"])
    logger.info("Judgment index namespace: %s", ctx["prefix_str"])

    if phase in ("all", "dense"):
        _build_dense(ctx, force_rebuild, checkpoint_interval, max_chunks)
    if phase in ("all", "bm25"):
        _build_bm25(ctx, max_chunks)


def _cli(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the judgment dense (FAISS) and sparse (BM25) indexes."
    )
    parser.add_argument("config_path", nargs="?", default="config.yaml")
    parser.add_argument("max_chunks", nargs="?", type=int, default=None)
    parser.add_argument("--phase", choices=["all", "dense", "bm25"], default="all")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int,
                        default=DEFAULT_CHECKPOINT_INTERVAL)
    args = parser.parse_args(argv)

    config_path = args.config_path
    max_chunks = args.max_chunks
    # Backward compatibility: `build_judgments.py <n>` meant max_chunks.
    if config_path.isdigit():
        max_chunks = int(config_path)
        config_path = "config.yaml"

    main(config_path=config_path, max_chunks=max_chunks, phase=args.phase,
         force_rebuild=args.force_rebuild,
         checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    _cli()
