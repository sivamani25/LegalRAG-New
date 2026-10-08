"""Pre-production tests for the canonical judgment builder.

These exercise the checkpoint/resume behaviour of ``scripts/build_judgments.py``
with a tiny synthetic corpus and a mock embedder, without touching any real
indexes:

1. checkpoints fire when processing crosses bucket boundaries
2. resume continues from a validated partial index without duplicates
3. a mismatched corpus / index refuses to resume (no silent duplication)
4. stored ids that are not a corpus prefix are rejected
5. the dense and BM25 phases can run independently
6. Constitution / ``_merged`` paths are never touched
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
import build_judgments  # noqa: E402
from legalrag.retrieval.index import BM25SparseIndex, FAISSDenseIndex  # noqa: E402


class _MockEmbedder:
    dimension = 4
    model_name = "mock-model"

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), self.dimension), dtype=np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = {
        "retrieval": {
            "embedding_model": "mock-model",
            "embedding_batch_size": 2,
            "index_dir": str(tmp_path / "index"),
        },
        "ingest": {
            "processed_dir": str(tmp_path / "processed"),
            "chunks_file": "chunks.jsonl",
        },
        "chunking": {"use_sac": True},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    processed_dir = tmp_path / "processed" / "judgments-full"
    processed_dir.mkdir(parents=True)
    corpus_path = processed_dir / "chunks.jsonl"

    # Same Path shim the existing judgment tests rely on.
    monkeypatch.setattr(
        build_judgments, "Path",
        lambda x: tmp_path / x if "data" in str(x) else Path(x),
    )
    monkeypatch.setattr(
        build_judgments.Embedder, "from_config", lambda c: _MockEmbedder()
    )

    index_dir = tmp_path / "index" / "judgments"
    prefix = index_dir / "judgments_mock-model_sac"

    def write_corpus(ids, texts=None):
        texts = texts or [f"text for {i}" for i in ids]
        with open(corpus_path, "w", encoding="utf-8") as f:
            for cid, txt in zip(ids, texts):
                f.write(json.dumps(
                    {"chunk_id": cid, "category": "judgments", "text": txt}
                ) + "\n")

    return {
        "config_path": str(config_path),
        "corpus_path": corpus_path,
        "index_dir": index_dir,
        "prefix": prefix,
        "write_corpus": write_corpus,
        "cfg": cfg,
    }


def _load_dense(prefix):
    d = FAISSDenseIndex(4)
    d.load(str(prefix) + "_dense")
    return d


def test_checkpoint_crosses_bucket_boundaries(env, monkeypatch):
    env["write_corpus"]([f"c{i}" for i in range(12)])

    records = []
    original = build_judgments._checkpoint

    def spy(dense, *args, **kwargs):
        records.append(len(dense.doc_ids))
        return original(dense, *args, **kwargs)

    monkeypatch.setattr(build_judgments, "_checkpoint", spy)

    # batch_size=2, interval=4 -> buckets advance at 4, 8, 12.
    build_judgments.main(config_path=env["config_path"], checkpoint_interval=4)

    assert 4 in records
    assert 8 in records
    assert records[-1] == 12

    dense = _load_dense(env["prefix"])
    assert len(dense.doc_ids) == 12
    assert len(set(dense.doc_ids)) == 12


def test_resume_completes_without_duplicates(env):
    env["write_corpus"]([f"c{i}" for i in range(12)])

    build_judgments.main(config_path=env["config_path"], max_chunks=6,
                         checkpoint_interval=4)
    assert len(_load_dense(env["prefix"]).doc_ids) == 6

    build_judgments.main(config_path=env["config_path"], checkpoint_interval=4)

    dense = _load_dense(env["prefix"])
    assert len(dense.doc_ids) == 12
    assert len(set(dense.doc_ids)) == 12
    assert dense.doc_ids == [f"c{i}" for i in range(12)]

    with open(env["index_dir"] / "chunk_store.json", encoding="utf-8") as f:
        assert len(json.load(f)) == 12


def test_changed_corpus_refuses_resume(env):
    env["write_corpus"]([f"c{i}" for i in range(12)])
    build_judgments.main(config_path=env["config_path"], max_chunks=6)
    assert len(_load_dense(env["prefix"]).doc_ids) == 6

    # Different corpus content -> different fingerprint.
    env["write_corpus"]([f"x{i}" for i in range(12)])

    with pytest.raises(build_judgments.ResumeMismatchError):
        build_judgments.main(config_path=env["config_path"])

    # The existing index is untouched: no silent duplication / misalignment.
    assert len(_load_dense(env["prefix"]).doc_ids) == 6


def test_stored_ids_must_be_a_corpus_prefix(env):
    env["write_corpus"](["c0", "c1", "c2", "c3"])
    ctx = build_judgments._build_context(env["cfg"])
    prefix = ctx["prefix_str"]

    # Metadata matches, but the stored ids are not the corpus prefix.
    Path(prefix + "_dense.faiss").write_bytes(b"")
    with open(prefix + "_dense_meta.json", "w", encoding="utf-8") as f:
        json.dump(ctx["expected_meta"], f)
    with open(prefix + "_dense_ids.json", "w", encoding="utf-8") as f:
        json.dump(["z0", "z1"], f)

    with pytest.raises(build_judgments.ResumeMismatchError, match="Resume mismatch"):
        build_judgments._validate_resume(
            prefix, ctx["chunks_path"], ctx["expected_meta"]
        )


def test_dense_and_bm25_phases_are_independent(env):
    env["write_corpus"]([f"c{i}" for i in range(6)])
    p = str(env["prefix"])

    build_judgments.main(config_path=env["config_path"], phase="dense", max_chunks=6)
    assert Path(p + "_dense.faiss").exists()
    assert not Path(p + "_sparse_bm25.pkl").exists()
    assert len(_load_dense(env["prefix"]).doc_ids) == 6

    build_judgments.main(config_path=env["config_path"], phase="bm25", max_chunks=6)
    assert Path(p + "_sparse_bm25.pkl").exists()

    sparse = BM25SparseIndex()
    sparse.load(p + "_sparse")
    assert len(sparse.doc_ids) == 6
    # Dense artifact was not disturbed by the BM25 phase.
    assert len(_load_dense(env["prefix"]).doc_ids) == 6


def test_constitution_and_merged_paths_untouched(env):
    env["write_corpus"]([f"c{i}" for i in range(6)])

    root = env["index_dir"].parent
    root.mkdir(parents=True, exist_ok=True)
    const = root / "constitution_mock.faiss"
    merged = root / "_merged_mock.faiss"
    const.write_bytes(b"CONSTITUTION")
    merged.write_bytes(b"MERGED")
    const_mtime = const.stat().st_mtime_ns
    merged_mtime = merged.stat().st_mtime_ns

    build_judgments.main(config_path=env["config_path"], max_chunks=6)

    assert const.read_bytes() == b"CONSTITUTION"
    assert merged.read_bytes() == b"MERGED"
    assert const.stat().st_mtime_ns == const_mtime
    assert merged.stat().st_mtime_ns == merged_mtime

    # Only judgment-namespace artifacts may be written here.
    for f in env["index_dir"].iterdir():
        assert f.name.startswith("judgments_") or f.name == "chunk_store.json", f.name
