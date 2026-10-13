"""Tests for the federated cross-category merge (M8 tie-break fix)."""

from legalrag.app.retrieval import MultiCategoryRetriever


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query):
        return list(self.rows)


def make(acts, const, judg=None):
    retrievers = {"acts": FakeRetriever(acts), "constitution": FakeRetriever(const)}
    if judg is not None:
        retrievers["judgments"] = FakeRetriever(judg)
    return MultiCategoryRetriever(retrievers)


def test_rank_rrf_ties_break_toward_stronger_category():
    # Both categories contribute one rank-0 hit -> equal RRF; the category with
    # the higher internal top score must win the tie.
    merged = make(
        [("acts/a/0", 0.02, {"category": "acts"})],
        [("constitution/c/0", 0.10, {"category": "constitution"})],
    ).search("q")

    assert merged[0][0] == "constitution/c/0"
    assert merged[1][0] == "acts/a/0"


def test_lower_rank_document_stays_last():
    merged = make(
        [("acts/a/0", 0.5, {}), ("acts/a/1", 0.4, {})],
        [("constitution/c/0", 0.9, {})],
    ).search("q")

    ids = [chunk_id for chunk_id, _, _ in merged]
    assert ids[0] == "constitution/c/0"   # rank-0 tie broken by category strength
    assert ids[-1] == "acts/a/1"          # rank-1 hit is strictly lower


def test_related_cases_come_from_judgments():
    multi = make([], [], [("judgments/j/0", 0.5, {"source_doc": "j"})])
    related = multi.related_cases("q")
    assert related[0][0] == "judgments/j/0"
    assert related[0][2]["category"] == "judgments"


def test_empty_retrievers_return_no_results():
    assert make([], []).search("q") == []
