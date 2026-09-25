import os
import json
import pytest
import numpy as np

from legalrag.retrieval.index import FAISSDenseIndex, BM25SparseIndex, ChunkStore

def test_index_integration_roundtrip(tmp_path):
    """
    End-to-end integration test of index persistence and reloading without mocks.
    Uses tiny 4-dimensional embeddings instead of a real model.
    """
    
    # 1. Prepare fake data
    dim = 4
    texts = [
        "This is the first document about the constitution.",
        "The second document discusses fundamental rights.",
        "A third document on a completely different topic."
    ]
    doc_ids = ["doc1", "doc2", "doc3"]
    
    # Fake explicitly L2-normalized embeddings
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5]
    ])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    embeddings = vectors / norms
    
    metadata = {
        "embedding_model": "test-mock-model",
        "corpus_fingerprint": "abc123xyz",
        "config_fingerprint": "def456uvw",
        "sac_enabled": True,
        "document_prefix": "test_prefix: ",
        "index_version": "1.0",
    }
    
    prefix = str(tmp_path / "test_merged")
    
    # 2. Build and save indexes
    dense = FAISSDenseIndex(dimension=dim)
    dense.add(embeddings, doc_ids)
    dense.save(prefix + "_dense", metadata=metadata)
    
    sparse = BM25SparseIndex()
    sparse.add(texts, doc_ids)
    sparse.save(prefix + "_sparse", metadata=metadata)
    
    # 3. Build and save chunk store
    store = ChunkStore()
    store.add_chunks([
        {"chunk_id": "doc1", "category": "acts", "text": "...", "section": "1"},
        {"chunk_id": "doc2", "category": "acts", "text": "...", "section": "2"},
        {"chunk_id": "doc3", "category": "cases", "text": "...", "section": "3"},
    ])
    store.save(str(tmp_path / "chunk_store.json"))
    
    # 4. Verify disk contents
    assert os.path.exists(f"{prefix}_dense.faiss")
    assert os.path.exists(f"{prefix}_dense_ids.json")
    assert os.path.exists(f"{prefix}_dense_meta.json")
    assert os.path.exists(f"{prefix}_sparse_bm25.pkl")
    assert os.path.exists(f"{prefix}_sparse_meta.json")
    assert os.path.exists(str(tmp_path / "chunk_store.json"))
    
    # 5. Reload into new instances
    new_dense = FAISSDenseIndex(dimension=dim)
    new_dense.load(prefix + "_dense", expected_metadata=metadata)
    
    new_sparse = BM25SparseIndex()
    new_sparse.load(prefix + "_sparse", expected_metadata=metadata)
    
    new_store = ChunkStore()
    new_store.load(str(tmp_path / "chunk_store.json"))
    
    # 6. Verify contents
    assert new_dense.index.ntotal == 3
    assert new_dense.doc_ids == doc_ids
    assert new_sparse.doc_ids == doc_ids
    assert new_sparse.bm25.corpus_size == 3
    
    assert len(new_store) == 3
    assert new_store.get("doc1")["category"] == "acts"
