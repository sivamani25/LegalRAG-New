import pytest
from src.legalrag.retrieval.adaptive import AdaptiveRetriever
from src.legalrag.retrieval.classifier import QueryClass

class MockHybridRetriever:
    def __init__(self, category):
        self.category = category
        self.last_query = None
        
    def search(self, query):
        self.last_query = query
        if self.category == "_merged":
            return [("c1", 0.9, {"category": "constitution"}), ("j1", 0.8, {"category": "judgments"})]
        elif self.category == "judgments":
            return [("j1", 0.8, {"category": "judgments"}), ("j2", 0.7, {"category": "judgments"})]
        return []

@pytest.fixture
def base_retriever():
    return MockHybridRetriever(category="_merged")

@pytest.fixture
def judgments_retriever():
    return MockHybridRetriever(category="judgments")

@pytest.fixture
def adaptive_retriever(base_retriever, judgments_retriever):
    return AdaptiveRetriever(base_retriever=base_retriever, judgments_retriever=judgments_retriever)

def test_adaptive_fact_pattern_query(adaptive_retriever):
    query = "A person was arrested and kept in custody for several days without being produced before a magistrate. What legal remedy is available?"
    
    res = adaptive_retriever.search(query)
    
    # 1. Classifier identifies fact pattern (implies judgements path used)
    # 2. Retrieval restricted to judgments category
    # 3. Returned results come from judgment chunks
    for doc_id, score, meta in res:
        assert meta["category"] == "judgments"
        assert doc_id in ["j1", "j2"]
        assert doc_id != "c1"  # Constitution chunk is isolated
        
    # 4. Rewriter invoked and rewritten query passed to retrieval
    used_query = adaptive_retriever.judgments_retriever.last_query
    assert used_query is not None
    assert "arrest" in used_query
    assert "remedy" in used_query
    assert "a person" not in used_query.lower()
    
    # Ensure base_retriever was untouched
    assert adaptive_retriever.base_retriever.last_query is None

def test_adaptive_normal_legal_query(adaptive_retriever):
    query = "What does Article 21 provide?"
    
    res = adaptive_retriever.search(query)
    
    # 1. Preserves existing normal M3 retrieval path
    used_query = adaptive_retriever.base_retriever.last_query
    assert used_query == query
    
    # 2. Fact-pattern rewriting path NOT used
    assert adaptive_retriever.judgments_retriever.last_query is None
    
    # Result contains mixed categories typical of base retrieval
    categories = [meta["category"] for _, _, meta in res]
    assert "constitution" in categories
    assert "judgments" in categories

