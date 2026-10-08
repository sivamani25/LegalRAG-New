import pytest
from src.legalrag.retrieval.rewriter import RuleBasedLegalIssueRewriter
from src.legalrag.retrieval.classifier import QueryClass

@pytest.fixture
def rewriter():
    return RuleBasedLegalIssueRewriter()

def test_preserves_normal_legal_query(rewriter):
    q = "What does Article 21 provide?"
    result = rewriter.rewrite(q, QueryClass.NORMAL_LEGAL_QUERY)
    assert result == q

def test_rewrite_arrest_scenario(rewriter):
    q = "A person was arrested and kept in custody for several days without being produced before a magistrate. What legal remedy is available?"
    res = rewriter.rewrite(q, QueryClass.FACT_PATTERN_QUERY)
    
    assert "arrest" in res
    assert "custody" in res or "detention" in res
    assert "failure" in res and "magistrate" in res
    assert "remedy" in res
    
    # Check that narrative noise is removed
    assert "a person" not in res.lower()
    assert "was" not in res.lower()

def test_rewrite_termination_scenario(rewriter):
    q = "A company terminated an employee without notice after the employee complained about discrimination. What legal rights may apply?"
    res = rewriter.rewrite(q, QueryClass.FACT_PATTERN_QUERY)
    
    assert "termination" in res
    assert "employee" in res
    assert "notice" in res
    assert "discrimination" in res
    assert "rights" in res
    
    assert "a company" not in res.lower()

def test_rewrite_contract_duress_scenario(rewriter):
    q = "A person signed a contract after being threatened with violence. Can the contract be challenged?"
    res = rewriter.rewrite(q, QueryClass.FACT_PATTERN_QUERY)
    
    assert "contract" in res
    assert "threat" in res or "violence" in res
    assert "coercion" in res or "duress" in res
    assert "challenge" in res or "validity" in res
    
    assert "can the" not in res.lower()

