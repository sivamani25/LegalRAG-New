import json
import logging
import faiss
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_index")

def verify():
    base_dir = Path("data/index/judgments")
    prefix = base_dir / "judgments_nomic-embed-text_sac"
    
    # 1. Chunk Store
    with open(base_dir / "chunk_store.json", "r", encoding="utf-8") as f:
        chunk_store = json.load(f)
    logger.info(f"Chunk store count: {len(chunk_store)}")
    assert len(chunk_store) == 577552, "Chunk store count mismatch"
    
    # Check for duplicates (dict keys are unique, so len() verifies unique IDs)
    
    # 2. FAISS dense index
    dense_meta_path = f"{prefix}_dense_meta.json"
    dense_path = f"{prefix}_dense.faiss"
    dense_ids_path = f"{prefix}_dense_ids.json"
    
    with open(dense_meta_path, "r", encoding="utf-8") as f:
        dense_meta = json.load(f)
    logger.info(f"Dense metadata exists, config fingerprint: {dense_meta.get('config_fingerprint')}")
    assert "config_fingerprint" in dense_meta
    
    index = faiss.read_index(dense_path)
    logger.info(f"FAISS vector count: {index.ntotal}")
    assert index.ntotal == 577552, "FAISS count mismatch"
    logger.info(f"FAISS dimension: {index.d}")
    assert index.d == 768, "FAISS dimension mismatch"
    
    with open(dense_ids_path, "r", encoding="utf-8") as f:
        dense_ids = json.load(f)
    assert len(dense_ids) == 577552
    assert len(set(dense_ids)) == 577552, "Duplicate chunk IDs in dense_ids"
    
    # 3. BM25 sparse index
    sparse_meta_path = f"{prefix}_sparse_meta.json"
    sparse_pkl_path = f"{prefix}_sparse_bm25.pkl"
    
    with open(sparse_meta_path, "r", encoding="utf-8") as f:
        sparse_meta = json.load(f)
    assert "config_fingerprint" in sparse_meta
    
    with open(sparse_pkl_path, "rb") as f:
        sparse_data = pickle.load(f)
    bm25 = sparse_data["bm25"]
    doc_ids = sparse_data["doc_ids"]
    logger.info(f"BM25 document count: {len(doc_ids)}")
    assert len(doc_ids) == 577552, "BM25 doc count mismatch"
    assert len(bm25.corpus_size) == 577552 if hasattr(bm25, 'corpus_size') else True
    
    logger.info("Verification passed successfully.")

if __name__ == '__main__':
    verify()
