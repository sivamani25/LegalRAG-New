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
            "embedding_batch_size": 16,          # non-default so tests can assert it
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

    # 3 distinct categories → 3 encode calls.
    # The _merged corpus differs from all per-category corpora, so it also needs
    # one encode call → total 4 in this multi-category test.
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


@patch("legalrag.retrieval.builder.Embedder.from_config")
@patch("legalrag.retrieval.builder.FAISSDenseIndex")
@patch("legalrag.retrieval.builder.BM25SparseIndex")
def test_batch_size_flows_from_config_to_encode(MockSparse, MockDense, MockEmbedder, mock_config, mock_chunks):
    """FIX 1: embedding_batch_size from config must arrive at embedder.encode(batch_size=)."""
    config_path, config_data = mock_config

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.dimension = 128
    MockEmbedder.return_value = mock_embedder_instance

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Confirm the fixture set a non-default value
    assert cfg["retrieval"]["embedding_batch_size"] == 16

    builder = IndexBuilder(cfg)
    assert builder.batch_size == 16

    builder.load_or_build(force_rebuild=True)

    # Every encode call must have received batch_size=16 (from config)
    for call in mock_embedder_instance.encode.call_args_list:
        _, kwargs = call
        assert kwargs.get("batch_size") == 16, (
            f"Expected batch_size=16 but got {kwargs.get('batch_size')} "
            f"in call: {call}"
        )


@patch("legalrag.retrieval.builder.Embedder.from_config")
@patch("legalrag.retrieval.builder.FAISSDenseIndex")
@patch("legalrag.retrieval.builder.BM25SparseIndex")
def test_merged_index_reuses_embeddings_when_single_category(
    MockSparse, MockDense, MockEmbedder, tmp_path
):
    """FIX 2: When corpus has one category, merged == category corpus.
    The embedder.encode() must be called ONCE (not twice) — the merged index
    reuses the cached embeddings from the category build.
    """
    import json as _json, yaml as _yaml

    # Build a single-category corpus with >= MIN_CORPUS_SIZE chunks
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    single_cat_chunks = [
        {"chunk_id": f"const_{i}", "category": "constitution", "text": f"Article {i} text"}
        for i in range(12)
    ]
    chunks_path = processed_dir / "chunks.jsonl"
    with open(chunks_path, "w") as f:
        for c in single_cat_chunks:
            f.write(_json.dumps(c) + "\n")

    config_data = {
        "retrieval": {
            "embedding_model": "mock-model",
            "embedding_batch_size": 4,
            "index_dir": str(tmp_path / "index"),
        },
        "ingest": {
            "processed_dir": str(processed_dir),
            "chunks_file": "chunks.jsonl",
        },
        "chunking": {"use_sac": False},
    }

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.dimension = 8
    MockEmbedder.return_value = mock_embedder_instance

    builder = IndexBuilder(config_data)
    builder.load_or_build(force_rebuild=True)

    # With one category:
    #   - category "constitution" → 1 encode call (result cached)
    #   - merged corpus == constitution corpus → cache HIT, 0 additional encode calls
    # Total: exactly 1 encode call.
    assert mock_embedder_instance.encode.call_count == 1, (
        f"Expected exactly 1 encode call (merged reuses cache), "
        f"got {mock_embedder_instance.encode.call_count}"
    )

    # Both the category index and the merged index must still produce saved artifacts.
    # dense.save should be called twice (once per index namespace).
    mock_dense_instance = MockDense.return_value
    assert mock_dense_instance.save.call_count == 2
    assert MockSparse.return_value.save.call_count == 2

    # Embedding cache must be cleared after load_or_build (FIX 3).
    assert len(builder._embedding_cache) == 0


@patch("legalrag.retrieval.builder.Embedder.from_config")
@patch("legalrag.retrieval.builder.FAISSDenseIndex")
@patch("legalrag.retrieval.builder.BM25SparseIndex")
def test_embedding_cache_cleared_after_build(MockSparse, MockDense, MockEmbedder, mock_config, mock_chunks):
    """FIX 3: _embedding_cache must be empty after load_or_build completes."""
    config_path, _ = mock_config
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.dimension = 128
    MockEmbedder.return_value = mock_embedder_instance

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    builder = IndexBuilder(cfg)
    assert len(builder._embedding_cache) == 0  # starts empty

    builder.load_or_build(force_rebuild=True)

    assert len(builder._embedding_cache) == 0, (
        "_embedding_cache must be cleared after load_or_build() to release memory"
    )




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
