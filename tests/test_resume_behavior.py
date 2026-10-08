import json
import pytest
from pathlib import Path
import sys
import os

# Add scripts dir to path to import build_judgments
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
import build_judgments
import yaml
from legalrag.retrieval.index import FAISSDenseIndex

def test_resume_behavior(tmp_path, monkeypatch):
    # Setup mock config
    config_data = {
        "retrieval": {
            "embedding_model": "mock-model",
            "embedding_batch_size": 2,
            "index_dir": str(tmp_path / "index")
        },
        "ingest": {
            "processed_dir": str(tmp_path / "processed"),
            "chunks_file": "chunks.jsonl"
        },
        "chunking": {
            "use_sac": True
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    processed_dir = tmp_path / "processed" / "judgments-full"
    processed_dir.mkdir(parents=True)
    
    # 10 chunks total
    chunks = [
        {"chunk_id": f"c{i}", "category": "judgments", "text": f"text {i}"}
        for i in range(10)
    ]
    with open(processed_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    monkeypatch.setattr(build_judgments, "Path", lambda x: tmp_path / x if "data" in str(x) else Path(x))
    
    class MockEmbedder:
        dimension = 4
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.ones((len(texts), self.dimension), dtype=np.float32)

    monkeypatch.setattr(build_judgments.Embedder, "from_config", lambda cfg: MockEmbedder())
    
    # Run first pass: only 4 chunks
    build_judgments.main(config_path=str(config_path), max_chunks=4)
    
    prefix = tmp_path / "index" / "judgments" / "judgments_mock-model_sac"
    dense = FAISSDenseIndex(4)
    dense.load(str(prefix) + "_dense")
    assert len(dense.doc_ids) == 4
    
    # Check that FAISS vectors match unique chunks
    assert len(set(dense.doc_ids)) == 4
    
    # Run second pass: 7 chunks
    build_judgments.main(config_path=str(config_path), max_chunks=7)
    
    dense.load(str(prefix) + "_dense")
    assert len(dense.doc_ids) == 7
    assert len(set(dense.doc_ids)) == 7
    assert "c6" in dense.doc_ids
    
    # Final pass: run all 10 chunks
    build_judgments.main(config_path=str(config_path), max_chunks=10)
    
    dense.load(str(prefix) + "_dense")
    assert len(dense.doc_ids) == 10
    assert len(set(dense.doc_ids)) == 10
