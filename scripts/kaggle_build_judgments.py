#!/usr/bin/env python3
"""Standalone Kaggle-compatible builder for the FULL LegalRAG judgment DENSE index.

This script is intentionally self-contained (it does NOT import the legalrag
package) so it can be dropped into a fresh Kaggle session and run against the
full corpus.  It writes files byte-compatible with
``legalrag.retrieval.index.FAISSDenseIndex`` / ``ChunkStore`` so they can be
copied back into ``data/index/judgments/`` on Windows:

    <out-dir>/judgments_nomic-embed-text_sac_dense.faiss
    <out-dir>/judgments_nomic-embed-text_sac_dense_ids.json
    <out-dir>/judgments_nomic-embed-text_sac_dense_meta.json
    <out-dir>/chunk_store.json

BM25 is intentionally NOT built here (build it separately on Windows).

Embedding contract (identical to production):
    model            = nomic-ai/nomic-embed-text-v1.5  (trust_remote_code=True)
    prefix           = "search_document: "
    max_seq_length   = 2048
    normalize_embeddings = True
    batch_size       = 32
    dimension        = 768

Resume safety: a checkpoint is only trusted after the stored dense ids are
verified to be an exact prefix of the current corpus (never a bare positional
offset), and the config/corpus fingerprints match.

Usage (Kaggle):
    python kaggle_build_judgments.py --chunks /kaggle/input/<ds>/chunks.jsonl \
        --out-dir /kaggle/working/judgment_index_checkpoint
    python kaggle_build_judgments.py --chunks ... --out-dir ... --mode resume
"""

import argparse
import gc
import hashlib
import json
import os
import time

# ---------------------------------------------------------------------------
# Contract constants (must match the Windows production config)
# ---------------------------------------------------------------------------
MODEL_NAME = "nomic-embed-text"
NOMIC_DOCUMENT_PREFIX = "search_document: "
MODEL_REGISTRY = {"nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5"}
INDEX_VERSION = "1.0"
DIM = 768
DEFAULT_CHECKPOINT_EVERY = 5000
DEFAULT_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 2048
EXCLUDE = frozenset({"text", "original_text"})
NAMESPACE = "judgments_nomic-embed-text_sac"


class ResumeMismatchError(RuntimeError):
    """Raised when an existing checkpoint cannot be safely resumed."""


# ---------------------------------------------------------------------------
# Fingerprints — exact replicas of legalrag.retrieval.index
# ---------------------------------------------------------------------------
def corpus_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def config_fingerprint(embedding_model, sac_enabled, document_prefix, index_version) -> str:
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


def iter_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def count_chunks(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Embedder (GPU)
# ---------------------------------------------------------------------------
class GPUEmbedder:
    def __init__(self, model, dim, batch_size):
        self._model = model
        self.dim = dim
        self.batch_size = batch_size

    def encode(self, texts):
        import numpy as np
        emb = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)


def get_embedder(model_name=MODEL_NAME, batch_size=DEFAULT_BATCH_SIZE,
                 max_seq_length=MAX_SEQ_LENGTH, device=None):
    import torch
    from sentence_transformers import SentenceTransformer

    hf = MODEL_REGISTRY.get(model_name, model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {hf} on {device}", flush=True)
    model = SentenceTransformer(hf, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    model.to(device)
    dim = model.get_sentence_embedding_dimension()
    print(f"[model] max_seq_length={model.max_seq_length} dim={dim}", flush=True)
    return GPUEmbedder(model, dim, batch_size)


# ---------------------------------------------------------------------------
# On-disk writers — byte-compatible with FAISSDenseIndex / ChunkStore
# ---------------------------------------------------------------------------
def _paths(out_dir):
    prefix = os.path.join(out_dir, NAMESPACE + "_dense")
    return {
        "faiss": prefix + ".faiss",
        "ids": prefix + "_ids.json",
        "meta": prefix + "_meta.json",
        "store": os.path.join(out_dir, "chunk_store.json"),
        "prefix": prefix,
    }


def save_dense(index, ids, out_dir, expected_meta):
    import faiss
    p = _paths(out_dir)
    faiss.write_index(index, p["faiss"])
    with open(p["ids"], "w", encoding="utf-8") as f:
        json.dump(ids, f)
    meta = dict(expected_meta)
    meta.update({"chunk_count": len(ids), "index_version": INDEX_VERSION})
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_store(store, out_dir):
    p = _paths(out_dir)
    with open(p["store"], "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------
def validate_resume(out_dir, chunks_path, expected_meta) -> int:
    """Return N if the first N corpus chunks match the checkpoint ids, else raise."""
    p = _paths(out_dir)
    if not (os.path.exists(p["meta"]) and os.path.exists(p["faiss"])
            and os.path.exists(p["ids"])):
        return 0

    with open(p["meta"], "r", encoding="utf-8") as f:
        stored = json.load(f)
    for k, v in expected_meta.items():
        if k in stored and stored[k] != v:
            raise ResumeMismatchError(
                f"Checkpoint does not match current config/corpus "
                f"(field '{k}': stored={stored[k]!r}, expected={v!r}). "
                f"Refusing to resume; use --force-rebuild for a clean build."
            )

    with open(p["ids"], "r", encoding="utf-8") as f:
        ids = json.load(f)
    n = len(ids)
    if n == 0:
        return 0

    seen = 0
    for ch in iter_chunks(chunks_path):
        if seen >= n:
            break
        if ch.get("chunk_id") != ids[seen]:
            raise ResumeMismatchError(
                f"Resume mismatch at position {seen}: stored id {ids[seen]!r} "
                f"!= corpus chunk_id {ch.get('chunk_id')!r}. Refusing to "
                f"resume; use --force-rebuild for a clean build."
            )
        seen += 1
    if seen < n:
        raise ResumeMismatchError(
            f"Corpus has only {seen} chunks but checkpoint holds {n} vectors. "
            f"Refusing to resume; use --force-rebuild."
        )
    return n


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def log_progress(phase, done, total, t0, ntotal, checkpoint=False):
    el = time.perf_counter() - t0
    pct = (done / total * 100.0) if total else 0.0
    cps = (done / el) if el > 0 else 0.0
    eta = ((total - done) / cps) if cps > 0 else float("inf")
    print(
        f"[{phase}] {done}/{total} ({pct:.2f}%) | {cps:.2f} chunks/s | "
        f"elapsed {el:.0f}s | ETA {eta:.0f}s | FAISS ntotal={ntotal} | "
        f"{'CHECKPOINT SAVED' if checkpoint else 'running'}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_index(chunks_path, out_dir, embedder, model_name=MODEL_NAME,
                checkpoint_every=DEFAULT_CHECKPOINT_EVERY, force_rebuild=False,
                max_chunks=None, log_every=20, mode="build"):
    import faiss
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    p = _paths(out_dir)
    total = count_chunks(chunks_path)
    if max_chunks:
        total = min(total, max_chunks)

    if embedder.dim != DIM:
        raise RuntimeError(f"Embedder dim {embedder.dim} != required {DIM}")

    expected_meta = {
        "corpus_fingerprint": corpus_fingerprint(chunks_path),
        "config_fingerprint": config_fingerprint(
            model_name, True, NOMIC_DOCUMENT_PREFIX, INDEX_VERSION),
        "embedding_model": model_name,
        "sac_enabled": True,
        "document_prefix": NOMIC_DOCUMENT_PREFIX,
        "index_version": INDEX_VERSION,
    }

    # ---- resume / completion / overwrite guard ----
    start = 0
    index = faiss.IndexFlatIP(embedder.dim)
    existing_ids = []
    if force_rebuild:
        print("[build] --force-rebuild: starting from zero vectors "
              "(existing files will be overwritten).", flush=True)
    else:
        start = validate_resume(out_dir, chunks_path, expected_meta)
        if start:
            if start >= total:
                print(f"[build] Index already COMPLETE ({start} vectors) — "
                      f"nothing to do. (Use --force-rebuild to rebuild.)", flush=True)
                return start
            if mode == "resume":
                pass
            index = faiss.read_index(p["faiss"])
            with open(p["ids"], "r", encoding="utf-8") as f:
                existing_ids = json.load(f)
            print(f"[build] Resuming from {start} existing vectors.", flush=True)

    if mode == "resume" and start == 0 and not force_rebuild:
        raise ResumeMismatchError("--mode resume requested but no valid checkpoint found.")

    ids = list(existing_ids)
    embedded = start
    lines_seen = 0
    store = {}
    batch_texts = []
    batch_ids = []
    last_bucket = embedded // checkpoint_every
    t0 = time.perf_counter()
    batches = 0

    for ch in iter_chunks(chunks_path):
        if max_chunks and lines_seen >= max_chunks:
            break
        lines_seen += 1

        cid = ch["chunk_id"]
        if cid not in store:
            store[cid] = {k: v for k, v in ch.items() if k not in EXCLUDE}

        if lines_seen <= start:
            continue

        batch_texts.append(NOMIC_DOCUMENT_PREFIX + ch["text"])
        batch_ids.append(cid)

        if len(batch_texts) >= embedder.batch_size:
            emb = embedder.encode(batch_texts)
            index.add(np.ascontiguousarray(emb, dtype=np.float32))
            ids.extend(batch_ids)
            embedded += len(batch_ids)
            batch_texts = []
            batch_ids = []
            batches += 1

            if batches % log_every == 0:
                log_progress("embed", embedded, total, t0, index.ntotal)

            bucket = embedded // checkpoint_every
            if bucket > last_bucket:
                save_dense(index, ids, out_dir, expected_meta)
                save_store(store, out_dir)
                last_bucket = bucket
                log_progress("embed", embedded, total, t0, index.ntotal, checkpoint=True)

    if batch_texts:
        emb = embedder.encode(batch_texts)
        index.add(np.ascontiguousarray(emb, dtype=np.float32))
        ids.extend(batch_ids)
        embedded += len(batch_ids)

    save_dense(index, ids, out_dir, expected_meta)
    save_store(store, out_dir)
    log_progress("embed", embedded, total, t0, index.ntotal, checkpoint=True)

    if embedded == total:
        print(f"[build] COMPLETE: {embedded} vectors written to {out_dir}", flush=True)
    else:
        print(f"[build] PARTIAL: {embedded}/{total} vectors (interrupted or --limit). "
              f"Re-run to resume.", flush=True)
    return embedded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the LegalRAG judgment dense index on GPU.")
    ap.add_argument("--chunks", required=True, help="path to chunks.jsonl")
    ap.add_argument("--out-dir", default="/kaggle/working/judgment_index_checkpoint")
    ap.add_argument("--mode", choices=["build", "resume"], default="build")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    ap.add_argument("--limit", type=int, default=None, help="max chunks (diagnostics only)")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    embedder = get_embedder(args.model, args.batch_size)
    build_index(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        embedder=embedder,
        model_name=args.model,
        checkpoint_every=args.checkpoint_every,
        force_rebuild=args.force_rebuild,
        max_chunks=args.limit,
        mode=args.mode,
    )
    gc.collect()


if __name__ == "__main__":
    main()
