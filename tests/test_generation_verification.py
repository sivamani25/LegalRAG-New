from legalrag.generation.verification import validate_citations, normalize_source_labels


evidence = [
    {"chunk_id": "judgments/test/0", "text": "Passage one."},
    {"chunk_id": "judgments/test/1", "text": "Passage two."},
]


def test_valid_citations():
    result = validate_citations("The court considered the facts [S1] and law [S2].", evidence)
    assert result["valid"] is True
    assert result["used_sources"] == ["S1", "S2"]
    assert result["unknown_sources"] == []


def test_missing_citations():
    result = validate_citations("The court considered the facts.", evidence)
    assert result["valid"] is False
    assert result["has_citations"] is False


def test_fabricated_source_label():
    result = validate_citations("The rule is clear [S9].", evidence)
    assert result["valid"] is False
    assert result["unknown_sources"] == ["S9"]


def test_empty_answer():
    result = validate_citations("  ", evidence)
    assert result["valid"] is False
    assert "The answer is empty." in result["warnings"]



def test_citation_validation_does_not_claim_legal_support_verified():
    result = validate_citations(
        "The court considered the facts [S1].",
        evidence,
    )

    assert result["valid"] is True
    assert result["support_verified"] is False
    assert any(
        "substantive support" in warning.lower()
        for warning in result["warnings"]
    )


# ---------------------------------------------------------------------------
# Parenthesised-label compatibility (M8)
# ---------------------------------------------------------------------------

def test_normalize_parenthesized_labels():
    assert normalize_source_labels("See (S1) and ( s2 ).") == "See [S1] and [S2]."


def test_normalize_leaves_non_labels_untouched():
    # Only bare S<digits> inside parentheses is a label.
    assert normalize_source_labels("section (5) and (SX) stay") == "section (5) and (SX) stay"
    assert normalize_source_labels("plain S1 stays") == "plain S1 stays"


def test_parenthesized_valid_citation_is_validated():
    result = validate_citations(normalize_source_labels("The court held so (S1)."), evidence)
    assert result["valid"] is True
    assert result["used_sources"] == ["S1"]
    assert result["unknown_sources"] == []


def test_parenthesized_fabricated_label_is_flagged():
    result = validate_citations(normalize_source_labels("A rule is clear (S9)."), evidence)
    assert result["valid"] is False
    assert result["unknown_sources"] == ["S9"]


def test_missing_citations_remain_flagged():
    result = validate_citations("No labels here.", evidence)
    assert result["has_citations"] is False
    assert result["valid"] is False
