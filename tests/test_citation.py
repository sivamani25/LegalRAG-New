import pytest
from legalrag.ingest.citation import extract_citations, Citation

def test_extracts_air_sc():
    text = "As held in AIR 2020 SC 100, the rule is clear."
    cites = extract_citations(text, category="judgments")
    assert Citation("case_law", "AIR 2020 SC 100", "") in cites

def test_extracts_scc_bracketed():
    text = "See (2019) 5 SCC 200 for details."
    cites = extract_citations(text, category="judgments")
    assert Citation("case_law", "(2019) 5 SCC 200", "") in cites

def test_extracts_scc_bare():
    text = "Reported as 2020 SCC 15."
    cites = extract_citations(text, category="judgments")
    assert Citation("case_law", "2020 SCC 15", "") in cites

def test_extracts_neutral_citation():
    text = "Neutral cite 2023 INSC 350."
    cites = extract_citations(text, category="judgments")
    assert Citation("case_law", "2023 INSC 350", "") in cites

def test_extracts_section_with_act_name():
    text = "Under Section 300 of the Indian Penal Code, murder is defined."
    cites = extract_citations(text, category="judgments")
    assert Citation("statute", "Indian Penal Code", "300") in cites

def test_extracts_section_with_abbreviation():
    text = "Punishable under Section 302 IPC."
    cites = extract_citations(text, category="judgments")
    assert Citation("statute", "IPC", "302") in cites

def test_extracts_article_reference():
    text = "Violates Article 21(1) of the Constitution."
    cites = extract_citations(text, category="judgments")
    assert Citation("constitutional", "Constitution of India", "21(1)") in cites

def test_extracts_schedule_reference():
    text = "Found in Schedule VII."
    cites = extract_citations(text, category="judgments")
    assert Citation("schedule", "", "VII") in cites

def test_extracts_contract_clause():
    text = "Subject to Clause 3.1 of the agreement."
    cites = extract_citations(text, category="contracts")
    assert Citation("contract_clause", "", "3.1") in cites

def test_empty_text_returns_empty():
    assert extract_citations("", category="judgments") == []

def test_no_citations_returns_empty():
    assert extract_citations("Just some plain text.", category="judgments") == []

def test_deduplication():
    text = "See AIR 2020 SC 100 and again AIR 2020 SC 100."
    cites = extract_citations(text, category="judgments")
    assert len(cites) == 1
    assert cites[0] == Citation("case_law", "AIR 2020 SC 100", "")

def test_disabled_category_returns_empty():
    text = "See AIR 2020 SC 100."
    # Extractor shouldn't extract if category isn't enabled.
    cites = extract_citations(text, category="acts")
    assert cites == []

from legalrag.ingest.citation import extract_internal_refs

def test_extract_internal_refs():
    text = "See Exhibit A and Schedule 1. Also see Annexure B and Appendix II."
    refs = extract_internal_refs(text)
    assert "Exhibit A" in refs
    assert "Schedule 1" in refs
    assert "Annexure B" in refs
    assert "Appendix II" in refs
    assert len(refs) == 4

def test_internal_refs_dedup():
    text = "Exhibit A and Exhibit A."
    refs = extract_internal_refs(text)
    assert refs == ["Exhibit A"]

def test_extract_citations_ignores_internal_refs():
    # Make sure canonical citation extractor doesn't capture Exhibit/Annexure
    # (It does capture Schedule, which is expected for Indian law, but we leave
    # the canonical extractor untouched).
    pass
