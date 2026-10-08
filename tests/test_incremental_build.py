import json
import pytest
from pathlib import Path
import sys

# Add scripts dir to path to import build_judgments
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
import build_judgments
import yaml
from legalrag.retrieval.index import FAISSDenseIndex, BM25SparseIndex

def test_incremental_judgment_build(tmp_path, monkeypatch):
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
    
    chunks = [
        {"chunk_id": f"c{i}", "category": "judgments", "text": f"text {i}"}
        for i in range(5)
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
    
    # Run first time with max_chunks = 3
    build_judgments.main(config_path=str(config_path), max_chunks=3)
    
    # Verify index has 3
    prefix = tmp_path / "index" / "judgments" / "judgments_mock-model_sac"
    dense = FAISSDenseIndex(4)
    dense.load(str(prefix) + "_dense")
    assert len(dense.doc_ids) == 3
    assert "c0" in dense.doc_ids
    
    # Run second time with max_chunks = 5
    build_judgments.main(config_path=str(config_path), max_chunks=5)
    
    dense.load(str(prefix) + "_dense")
    assert len(dense.doc_ids) == 5
    
    sparse = BM25SparseIndex()
    sparse.load(str(prefix) + "_sparse")
    assert len(sparse.doc_ids) == 5

