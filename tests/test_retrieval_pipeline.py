import pytest
import numpy as np
from legalrag.retrieval.pipeline import (
    extract_explicit_references,
    is_structural_match,
    is_hierarchy_match,
    HybridRetriever
)
from legalrag.retrieval.index import ChunkStore

class MockDenseIndex:
    def __init__(self, doc_ids, vecs):
        self.doc_ids = doc_ids
        self.vecs = vecs
        self.index = self
    def reconstruct(self, idx):
        return self.vecs[idx]
    def search(self, query, top_k):
        # Deterministic mock search
        return [(self.doc_ids[i], 1.0 - i*0.1) for i in range(min(top_k, len(self.doc_ids)))]

class MockSparseIndex:
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids
    def search(self, query, top_k):
        # Reverse order for sparse to test RRF
        return [(self.doc_ids[i], 10.0 - i) for i in range(min(top_k, len(self.doc_ids)))][::-1]

class MockEmbedder:
    dimension = 4
    model_name = "mock"
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)

@pytest.fixture
def mock_retriever(monkeypatch):
    config = {
        "retrieval": {
            "top_k": 3,
            "rrf_k": 1,
            "mmr_lambda": 0.5,
            "structural_boost_strength": 1.0,
            "hierarchy_boost_strength": 0.5,
            "citation_boost_strength": 0.5,
        }
    }
    
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.config = config
    retriever.retrieval_cfg = config["retrieval"]
    retriever.category = "_merged"
    retriever.top_k = 3
    retriever.rrf_k = 1
    retriever.mmr_lambda = 0.5
    retriever.use_mmr = True
    retriever.use_structural_boost = True
    retriever.structural_boost = 1.0
    retriever.use_hierarchy_boost = True
    retriever.hierarchy_boost = 0.5
    retriever.use_citation_boost = True
    retriever.citation_boost = 0.5
    retriever.model_name = "mock"
    
    doc_ids = ["d1", "d2", "d3", "d4"]
    vecs = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    retriever.dense = MockDenseIndex(doc_ids, vecs)
    retriever.sparse = MockSparseIndex(doc_ids)
    retriever.embedder = MockEmbedder()
    
    cs = ChunkStore()
    cs._store = {
        "d1": {"section": "21", "hierarchy_path": ["PART III", "21"], "source_doc": "doc1"},
        "d2": {"hierarchy_path": ["Part III", "Article 21"], "source_doc": "doc1"},
        "d3": {"cites": [{"doc": "doc1", "section": "21"}], "source_doc": "doc2"},
        "d4": {"section": "14", "source_doc": "doc1"}
    }
    retriever.chunk_store = cs
    retriever._doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    
    return retriever


def test_extract_explicit_references():
    assert extract_explicit_references("What does Article 21 say?") == [("Article", "21")]
    assert extract_explicit_references("Is Section 302 related to Part III?") == [("Section", "302"), ("Part", "III")]
    assert extract_explicit_references("No reference here") == []

def test_is_structural_match():
    meta = {"section": "21", "hierarchy_path": ["PART III", "21"]}
    assert is_structural_match(meta, [("Article", "21")])
    assert not is_structural_match(meta, [("Article", "14")])
    
def test_is_hierarchy_match():
    meta = {"hierarchy_path": ["Part III", "Article 21"], "parent": "Part III"}
    assert is_hierarchy_match(meta, [("Part", "III")])
    assert is_hierarchy_match(meta, [("Article", "21")])
    assert not is_hierarchy_match(meta, [("Article", "14")])

def test_rrf_and_score_ordering(mock_retriever):
    mock_retriever.use_mmr = False
    mock_retriever.use_structural_boost = False
    mock_retriever.use_hierarchy_boost = False
    mock_retriever.use_citation_boost = False
    
    res = mock_retriever.search("test")
    # Dense: d1(rank 0), d2(rank 1), d3(rank 2), d4(rank 3)
    # Sparse: d4(rank 0), d3(rank 1), d2(rank 2), d1(rank 3)
    # RRF(k=1) d1: 1/1 + 1/4 = 1.25
    # RRF(k=1) d2: 1/2 + 1/3 = 0.833
    # RRF(k=1) d3: 1/3 + 1/2 = 0.833
    # RRF(k=1) d4: 1/4 + 1/1 = 1.25
    
    assert res[0][0] in ["d1", "d4"]
    assert res[1][0] in ["d1", "d4"]
    
def test_structural_boosting(mock_retriever):
    mock_retriever.use_mmr = False
    mock_retriever.use_citation_boost = False
    res = mock_retriever.search("Article 21")
    # d1 should get structural boost of +1.0
    # d1 base RRF 1.25 + 1.0 = 2.25, should be rank 1
    assert res[0][0] == "d1"

def test_hierarchy_boosting(mock_retriever):
    mock_retriever.use_mmr = False
    mock_retriever.use_citation_boost = False
    mock_retriever.use_structural_boost = False
    res = mock_retriever.search("Article 21")
    # d2 should get hierarchy boost of +0.5
    # d2 base RRF 0.833 + 0.5 = 1.333
    assert "d2" in [r[0] for r in res]

def test_citation_boosting(mock_retriever):
    mock_retriever.use_mmr = False
    mock_retriever.use_structural_boost = False
    mock_retriever.use_hierarchy_boost = False
    res = mock_retriever.search("test")
    # d3 cites d1. So d1 gets citation boost (+0.5)
    assert res[0][0] == "d1"

def test_mmr(mock_retriever):
    mock_retriever.use_structural_boost = False
    mock_retriever.use_hierarchy_boost = False
    mock_retriever.use_citation_boost = False
    mock_retriever.use_mmr = True
    
    # vectors for d1 and d2 are identical [1,0,0,0], d3 is [0,1,0,0], d4 is [0,0,1,0]
    # base RRF: d1/d4 are tied highest. If d1 is picked first, d2 is highly similar to d1.
    res = mock_retriever.search("test")
    # d1 and d4 should be picked. d2 should be suppressed due to mmr penalty.
    ids = [r[0] for r in res]
    assert "d2" not in ids[:2]

def test_empty_candidates():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.top_k = 3
    retriever.mmr_lambda = 0.5
    retriever.dense = MockDenseIndex([], [])
    retriever.sparse = MockSparseIndex([])
    retriever.embedder = MockEmbedder()
    retriever.chunk_store = ChunkStore()
    retriever._doc_id_to_idx = {}
    
    assert retriever._apply_mmr([], {}) == []
    
def test_category_filtering():
    # Pipeline's `__init__` takes `category` and loads the correct index namespace
    pass

def test_metadata_candidate_expansion_article_21a(mock_retriever):
    mock_retriever.use_mmr = False
    
    # We add a doc that is NOT returned by the mock dense/sparse
    # It has section 21A, hierarchy_path=['PART III', '21A']
    mock_retriever.chunk_store._store["d5"] = {
        "section": "21A",
        "hierarchy_path": ["PART III", "21A"],
        "category": "_merged"
    }
    
    res = mock_retriever.search("What is Article 21A?")
    
    # d5 should be structurally boosted because extract_explicit_references finds it
    assert "d5" in [r[0] for r in res]
    
def test_metadata_candidate_expansion_schedule_entry(mock_retriever):
    mock_retriever.use_mmr = False
    
    mock_retriever.chunk_store._store["d6"] = {
        "section": "21",
        "hierarchy_path": ["SEVENTH SCHEDULE", "21"],
        "category": "_merged"
    }
    mock_retriever.chunk_store._store["d7"] = {
        "section": "21",
        "hierarchy_path": ["PART III", "21"],
        "category": "_merged"
    }
    mock_retriever.chunk_store._store["d10"] = {
        "section": "21",
        "hierarchy_path": ["SIXTH SCHEDULE", "21"],
        "category": "_merged"
    }
    
    # 1. Unqualified "Entry 21" should retrieve both Sixth and Seventh Schedule entries
    res_unqual = mock_retriever.search("What does Entry 21 concern?")
    ids_unqual = [r[0] for r in res_unqual]
    assert "d6" in ids_unqual
    assert "d10" in ids_unqual
    assert "d7" not in ids_unqual  # Part III rejected
    
    # 2. "Seventh Schedule Entry 21" should isolate Seventh Schedule
    res_seventh = mock_retriever.search("What does the Seventh Schedule Entry 21 concern?")
    ids_seventh = [r[0] for r in res_seventh]
    assert "d6" in ids_seventh
    assert "d10" not in ids_seventh
    
    # 3. "Sixth Schedule Entry 21" should isolate Sixth Schedule
    res_sixth = mock_retriever.search("What does the Sixth Schedule Entry 21 concern?")
    ids_sixth = [r[0] for r in res_sixth]
    assert "d10" in ids_sixth
    assert "d6" not in ids_sixth

def test_metadata_candidate_expansion_article_21(mock_retriever):
    mock_retriever.use_mmr = False
    
    mock_retriever.chunk_store._store["d8"] = {
        "section": "21",
        "hierarchy_path": ["PART III", "21"],
        "category": "_merged"
    }
    
    res = mock_retriever.search("What does Article 21 provide?")
    
    ids = [r[0] for r in res]
    assert "d8" in ids

def test_metadata_candidate_expansion_article_368(mock_retriever):
    mock_retriever.use_mmr = False
    
    mock_retriever.chunk_store._store["d9"] = {
        "section": "368",
        "hierarchy_path": ["PART XX", "368"],
        "category": "_merged"
    }
    
    res = mock_retriever.search("What is Article 368?")
    
    ids = [r[0] for r in res]
    assert "d9" in ids
