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

def test_faiss_dense_index_search():
    dim = 4
    index = FAISSDenseIndex(dimension=dim)
    
    # Use float32 to match FAISS expectations
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)
    doc_ids = ["doc1", "doc2", "doc3"]
    index.add(embeddings, doc_ids)
    
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)
    
    assert len(results) == 2
    assert results[0][0] == "doc1"
    assert results[0][1] == pytest.approx(1.0)
    
    # Check dim mismatch
    bad_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Query dimension"):
        index.search(bad_query)
        
    # Check empty index safely returns empty list
    empty_index = FAISSDenseIndex(dimension=dim)
    assert empty_index.search(query) == []

def test_bm25_sparse_index_search():
    index = BM25SparseIndex()
    # Pad corpus so 'banana' appears in < 50% of docs, ensuring positive IDF in BM25Okapi
    texts = ["apple banana", "apple orange", "banana banana", "kiwi", "grape", "mango"]
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    
    index.add(texts, doc_ids)
    
    results = index.search("banana", top_k=2)
    # doc3 has "banana" twice, doc1 has it once
    assert len(results) == 2
    assert results[0][0] == "doc3"
    assert results[1][0] == "doc1"
    assert results[0][1] > results[1][1]
    
    # Check empty index safely returns empty list
    empty_index = BM25SparseIndex()
    assert empty_index.search("banana") == []
    
    # Check unknown token returns empty list (or no scores > 0)
    assert index.search("unknown") == []
