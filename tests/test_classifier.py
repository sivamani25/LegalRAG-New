import pytest
from legalrag.retrieval.classifier import RuleBasedQueryClassifier, QueryClass

@pytest.fixture
def classifier():
    return RuleBasedQueryClassifier()

def test_normal_legal_queries(classifier):
    queries = [
        "What does Article 21 provide?",
        "What is the punishment under Section 302?",
        "What is the difference between a lease and a license?",
        "How is the President of India elected?",
        "Can a company be sued for breach of contract?",
        "What are the fundamental rights guaranteed by the Constitution?"
    ]
    for q in queries:
        assert classifier.classify(q) == QueryClass.NORMAL_LEGAL_QUERY, f"Failed on: {q}"

def test_fact_pattern_queries(classifier):
    queries = [
        "A person was arrested and kept in custody for several days without being produced before a magistrate. What legal remedy is available?",
        "A company terminated an employee without notice after the employee complained about discrimination. What legal rights may apply?",
        "A person signed a contract after being threatened with violence. Can the contract be challenged?",
        "If a tenant refuses to vacate the premises after the lease expires, what can the landlord do?",
        "My landlord refused to return my security deposit and locked me out. Can I sue?",
        "Suppose a doctor operated on a patient and caused severe injuries due to negligence. What is the liability?"
    ]
    for q in queries:
        assert classifier.classify(q) == QueryClass.FACT_PATTERN_QUERY, f"Failed on: {q}"

