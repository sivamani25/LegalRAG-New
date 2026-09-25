import os
import json
import pytest
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

from legalrag.retrieval.builder import IndexBuilder
from legalrag.retrieval.embedding import NOMIC_DOCUMENT_PREFIX

@pytest.fixture
def mock_config(tmp_path):
    config_data = {
        "retrieval": {
            "embedding_model": "mock-model",
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
        
    return config_path, config_data

@pytest.fixture
def mock_chunks(tmp_path, mock_config):
    _, config_data = mock_config
    processed_dir = Path(config_data["ingest"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Must have >= MIN_CORPUS_SIZE (10) chunks to pass the safety guard.
    chunks = (
        [{"chunk_id": f"acts_{i}", "category": "acts", "text": f"acts text {i}"} for i in range(6)]
        + [{"chunk_id": f"cases_{i}", "category": "cases", "text": f"cases text {i}"} for i in range(5)]
        + [{"chunk_id": f"misc_{i}", "category": "misc", "text": f"misc text {i}"} for i in range(4)]
    )

    chunks_path = processed_dir / config_data["ingest"]["chunks_file"]
    with open(chunks_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    return chunks

@patch("legalrag.retrieval.builder.Embedder.from_config")
@patch("legalrag.retrieval.builder.FAISSDenseIndex")
@patch("legalrag.retrieval.builder.BM25SparseIndex")
def test_index_builder(MockSparse, MockDense, MockEmbedder, mock_config, mock_chunks):
    config_path, config_data = mock_config

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.dimension = 128
    MockEmbedder.return_value = mock_embedder_instance

    mock_dense_instance = MagicMock()
    MockDense.return_value = mock_dense_instance

    mock_sparse_instance = MagicMock()
    MockSparse.return_value = mock_sparse_instance

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    builder = IndexBuilder(cfg)
    builder.load_or_build(force_rebuild=True)

    MockEmbedder.assert_called_once()

    # Categories: 'acts' (6), 'cases' (5), 'misc' (4), plus '_merged' (15) = 4 encode calls
    assert mock_embedder_instance.encode.call_count == 4

    # _merged encode is the last one
    _, kwargs = mock_embedder_instance.encode.call_args_list[-1]
    assert kwargs["normalize"] is True

    assert mock_dense_instance.save.call_count == 4
    assert mock_sparse_instance.save.call_count == 4

    _, kwargs = mock_dense_instance.save.call_args_list[0]
    meta = kwargs["metadata"]
    assert meta["embedding_model"] == "mock-model"
    assert meta["sac_enabled"] is True
    assert "corpus_fingerprint" in meta
    assert "config_fingerprint" in meta

    # Verify ChunkStore is saved
    store_path = Path(config_data["retrieval"]["index_dir"]) / "chunk_store.json"
    assert store_path.exists()


def test_corpus_size_guard_raises(tmp_path, mock_config):
    """load_or_build must raise RuntimeError when corpus < MIN_CORPUS_SIZE."""
    _, config_data = mock_config
    processed_dir = Path(config_data["ingest"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Write only 2 chunks — well below MIN_CORPUS_SIZE=10
    tiny_chunks = [
        {"chunk_id": "c1", "category": "acts", "text": "text 1"},
        {"chunk_id": "c2", "category": "acts", "text": "text 2"},
    ]
    chunks_path = processed_dir / config_data["ingest"]["chunks_file"]
    with open(chunks_path, "w") as f:
        for c in tiny_chunks:
            f.write(json.dumps(c) + "\n")

    import yaml as _yaml
    with open(tmp_path / "config.yaml") as cf:
        cfg = _yaml.safe_load(cf)

    builder = IndexBuilder(cfg)
    with pytest.raises(RuntimeError, match="Corpus too small"):
        builder.load_or_build(force_rebuild=True)
