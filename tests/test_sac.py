import pytest
from legalrag.ingest.sac import (
    summarize_document, apply_sac_to_document, SummaryResult, SAC_SEPARATOR
)
from legalrag.ingest.chunker import Chunk

def test_heuristic_returns_summary_result():
    res = summarize_document("Test", use_llm=False)
    assert isinstance(res, SummaryResult)
    assert res.strategy == "heuristic"
    assert res.summary == "Test"

def test_heuristic_acts_category():
    text = "THE INDIAN PENAL CODE\n\nACT NO. 45 OF 1860\n\n[6th October, 1860.]\n\nCHAPTER I"
    res = summarize_document(text, category="acts")
    assert "INDIAN PENAL CODE" in res.summary
    assert "(1860)" in res.summary

def test_heuristic_judgments_category():
    text = "IN THE SUPREME COURT OF INDIA\n\nKeshava vs. State\n\nJudgment 1973"
    res = summarize_document(text, category="judgments")
    assert "SUPREME COURT OF INDIA" in res.summary
    assert "Keshava v. State" in res.summary
    assert "(1973)" in res.summary

def test_heuristic_contracts_category():
    text = "NON-DISCLOSURE AGREEMENT\n\nBetween \"Alice Corp\" and \"Bob Inc\"."
    res = summarize_document(text, category="contracts")
    assert "Agreement" in res.summary
    assert "Alice Corp" in res.summary
    assert "Bob Inc" in res.summary

def test_heuristic_respects_max_chars():
    res = summarize_document("A" * 200, max_chars=50)
    assert len(res.summary) <= 50

def test_heuristic_empty_text():
    assert summarize_document("").summary == ""

def test_llm_path_calls_func_once():
    called = []
    def fake_llm(text):
        called.append(text)
        return "LLM Summary"
    
    res = summarize_document("Doc text", use_llm=True, llm_func=fake_llm)
    assert res.strategy == "llm"
    assert res.summary == "LLM Summary"
    assert called == ["Doc text"]

def test_llm_path_requires_func():
    with pytest.raises(ValueError):
        summarize_document("Doc text", use_llm=True, llm_func=None)

def _make_chunk():
    return Chunk("id", "doc", "path", "cat", "jur", "Body", "Body", "", None, None, None, [], [], 0, 4, False)

def test_apply_prepends_summary_and_separator():
    chunks = [_make_chunk()]
    apply_sac_to_document(chunks, "Sum")
    assert chunks[0].text == f"Sum{SAC_SEPARATOR}Body"

def test_apply_sets_doc_summary_and_flag():
    chunks = [_make_chunk()]
    apply_sac_to_document(chunks, "Sum")
    assert chunks[0].doc_summary == "Sum"
    assert chunks[0].sac_applied is True

def test_apply_preserves_original_text():
    chunks = [_make_chunk()]
    apply_sac_to_document(chunks, "Sum")
    assert chunks[0].original_text == "Body"
    assert chunks[0].text != chunks[0].original_text

def test_apply_double_raises():
    chunks = [_make_chunk()]
    apply_sac_to_document(chunks, "Sum")
    with pytest.raises(ValueError):
        apply_sac_to_document(chunks, "Sum2")
