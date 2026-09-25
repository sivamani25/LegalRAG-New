import os
import pytest
import numpy as np
import json

from legalrag.retrieval.index import (
    FAISSDenseIndex,
    BM25SparseIndex,
    ChunkStore,
    IndexStaleError,
    corpus_fingerprint,
    config_fingerprint,
)

def test_fingerprints(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text('{"chunk_id": "c1"}\n{"chunk_id": "c2"}')
    
    corp_fp = corpus_fingerprint(str(chunks_file))
    assert isinstance(corp_fp, str)
    assert len(corp_fp) == 64  # SHA-256
    
    conf_fp = config_fingerprint("nomic-embed-text", True, "doc_pref: ", "1.0")
    assert isinstance(conf_fp, str)
    assert len(conf_fp) == 64

def test_faiss_dense_index(tmp_path):
    dim = 4
    index = FAISSDenseIndex(dimension=dim)
    
    embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    doc_ids = ["doc1", "doc2"]
    
    index.add(embeddings, doc_ids)
    assert index.doc_ids == doc_ids
    
    metadata = {"embedding_model": "test-model", "category": "test"}
    prefix = str(tmp_path / "test_dense")
    index.save(prefix, metadata=metadata)
    
    new_index = FAISSDenseIndex(dimension=dim)
    new_index.load(prefix, expected_metadata={"embedding_model": "test-model"})
    assert new_index.doc_ids == doc_ids
    
    # Stale metadata should raise IndexStaleError
    with pytest.raises(IndexStaleError, match="Index metadata mismatch"):
        new_index.load(prefix, expected_metadata={"embedding_model": "wrong-model"})

def test_bm25_sparse_index(tmp_path):
    index = BM25SparseIndex()
    texts = ["hello world", "hello pytest"]
    doc_ids = ["doc1", "doc2"]
    
    index.add(texts, doc_ids)
    assert index.doc_ids == doc_ids
    
    metadata = {"category": "test"}
    prefix = str(tmp_path / "test_sparse")
    index.save(prefix, metadata=metadata)
    
    new_index = BM25SparseIndex()
    new_index.load(prefix, expected_metadata={"category": "test"})
    assert new_index.doc_ids == doc_ids

    with pytest.raises(IndexStaleError):
        new_index.load(prefix, expected_metadata={"category": "wrong-cat"})

def test_chunk_store(tmp_path):
    store = ChunkStore()
    chunks = [
        {"chunk_id": "c1", "category": "acts", "text": "large body of text", "section": "1"},
        {"chunk_id": "c2", "category": "cases", "original_text": "huge text", "section": "2"},
    ]
    store.add_chunks(chunks)
    
    assert len(store) == 2
    assert store.get("c1") == {"chunk_id": "c1", "category": "acts", "section": "1"}
    assert "text" not in store.get("c1")
    assert "original_text" not in store.get("c2")
    
    path = tmp_path / "store.json"
    store.save(str(path))
    
    new_store = ChunkStore()
    new_store.load(str(path))
    
    assert len(new_store) == 2
    assert new_store.get("c1")["category"] == "acts"
