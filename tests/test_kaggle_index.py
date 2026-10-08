"""Local tests for the standalone Kaggle judgment builder/verifier.

Uses a fake 768-d embedder and a tiny synthetic corpus — the Nomic model is
NEVER loaded here, so this is safe for the normal test suite.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import kaggle_build_judgments as kb  # noqa: E402
import kaggle_verify_judgments as kv  # noqa: E402


class FakeEmbedder:
    dim = 768

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def encode(self, texts):
        v = np.ones((len(texts), self.dim), dtype=np.float32)
        return v / np.sqrt(self.dim)


def write_corpus(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({
                "chunk_id": cid, "category": "judgments",
                "source_doc": cid.split("/")[0], "text": f"text for {cid}",
                "start_offset": 0, "end_offset": 10,
            }) + "\n")


def test_build_format_and_counts(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    ids = [f"judgments/doc{i}/0" for i in range(12)]
    write_corpus(chunks, ids)
    out = tmp_path / "out"

    kb.build_index(str(chunks), str(out), FakeEmbedder(2), checkpoint_every=5)

    ok, results, info = kv.verify(str(out), str(chunks))
    assert ok, results
    assert info["faiss_dim"] == 768
    assert info["faiss_ntotal"] == 12
    assert info["ids"] == 12
    assert info["unique_ids"] == 12
    assert info["chunk_store"] == 12

    # chunk_store format: metadata present, text excluded
    store = json.load(open(out / "chunk_store.json", encoding="utf-8"))
    entry = store[ids[0]]
    assert "text" not in entry and "chunk_id" in entry

    # meta keys match the production format
    meta = json.load(open(out / "judgments_nomic-embed-text_sac_dense_meta.json",
                          encoding="utf-8"))
    for k in ("corpus_fingerprint", "config_fingerprint", "embedding_model",
              "sac_enabled", "document_prefix", "index_version", "chunk_count"):
        assert k in meta


def test_checkpoint_resume_no_duplicates(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    ids = [f"c{i}" for i in range(12)]
    write_corpus(chunks, ids)
    out = tmp_path / "out"

    n1 = kb.build_index(str(chunks), str(out), FakeEmbedder(2),
                        checkpoint_every=5, max_chunks=6)
    assert n1 == 6
    n2 = kb.build_index(str(chunks), str(out), FakeEmbedder(2), checkpoint_every=5)
    assert n2 == 12

    ok, results, info = kv.verify(str(out), str(chunks))
    assert ok, results
    assert info["unique_ids"] == 12
    assert info["ids_first"] == "c0" and info["ids_last"] == "c11"


def test_resume_rejects_mismatched_corpus(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write_corpus(chunks, [f"c{i}" for i in range(12)])
    out = tmp_path / "out"
    kb.build_index(str(chunks), str(out), FakeEmbedder(2), checkpoint_every=5,
                   max_chunks=6)

    # Different corpus content -> different fingerprint -> must refuse.
    write_corpus(chunks, [f"x{i}" for i in range(12)])
    with pytest.raises(kb.ResumeMismatchError):
        kb.build_index(str(chunks), str(out), FakeEmbedder(2))


def test_completed_index_is_not_overwritten(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write_corpus(chunks, [f"c{i}" for i in range(12)])
    out = tmp_path / "out"
    kb.build_index(str(chunks), str(out), FakeEmbedder(2))
    before = (out / "judgments_nomic-embed-text_sac_dense_meta.json").read_text()

    n = kb.build_index(str(chunks), str(out), FakeEmbedder(2))  # no force-rebuild
    assert n == 12
    assert (out / "judgments_nomic-embed-text_sac_dense_meta.json").read_text() == before


def test_verify_detects_corrupt_ids(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    ids = [f"c{i}" for i in range(12)]
    write_corpus(chunks, ids)
    out = tmp_path / "out"
    kb.build_index(str(chunks), str(out), FakeEmbedder(2))

    ids_path = out / "judgments_nomic-embed-text_sac_dense_ids.json"
    bad = json.load(open(ids_path, encoding="utf-8"))
    bad[-1] = bad[0]  # duplicate
    json.dump(bad, open(ids_path, "w", encoding="utf-8"))

    ok, results, _ = kv.verify(str(out), str(chunks))
    assert not ok
    assert not results["no_duplicate_ids"]
    assert not results["last_id_matches"]
