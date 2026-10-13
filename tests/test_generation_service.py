import pytest

from legalrag.generation.service import LegalAnswerService


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, question):
        return self.results


class FakeTextStore:
    def __init__(self, records):
        self.records = records

    def get_many(self, chunk_ids):
        return {cid: self.records[cid] for cid in chunk_ids if cid in self.records}


class FakeClient:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or "The evidence discusses the circumstances of the case. [S1]"

    def generate(self, messages):
        self.calls.append(messages)
        return self.answer


def make_service(client=None):
    cid = "judgments/test/0"
    service = LegalAnswerService(
        FakeRetriever([(cid, 0.25, {"category": "judgments"})]),
        FakeTextStore({
            cid: {
                "chunk_id": cid,
                "source_doc": "test-case",
                "original_text": "Courts consider the circumstances of each case.",
            }
        }),
        client or FakeClient(),
    )
    return service


def test_successful_answer_includes_evidence():
    client = FakeClient()
    service = make_service(client)

    result = service.answer("How is bail considered?")

    assert result["retrieved_count"] == 1
    assert result["used_count"] == 1
    assert result["evidence"][0]["chunk_id"] == "judgments/test/0"
    assert "Courts consider the circumstances" in client.calls[0][-1]["content"]
    assert "[S1]" in result["answer"]
    assert result["citation_validation"]["valid"] is True
    assert result["citation_validation"]["used_sources"] == ["S1"]
    assert result["citation_validation"]["unknown_sources"] == []
    assert result["citation_validation"]["support_verified"] is False
    assert any(
        "substantive support" in warning.lower()
        for warning in result["citation_validation"]["warnings"]
)


def test_fabricated_citation_is_flagged():
    client = FakeClient("The rule is established. [S9]")
    service = make_service(client)

    result = service.answer("What is the rule?")

    assert result["citation_validation"]["valid"] is False
    assert result["citation_validation"]["unknown_sources"] == ["S9"]
    assert result["citation_validation"]["warnings"]


def test_answer_without_citations_is_flagged():
    client = FakeClient("The evidence discusses the case circumstances.")
    service = make_service(client)

    result = service.answer("What happened in the case?")

    assert result["citation_validation"]["valid"] is False
    assert result["citation_validation"]["has_citations"] is False


def test_empty_retrieval_does_not_call_generation():
    client = FakeClient()
    service = LegalAnswerService(FakeRetriever([]), FakeTextStore({}), client)

    result = service.answer("Question without matches")

    assert result["used_count"] == 0
    assert result["evidence"] == []
    assert client.calls == []
    assert result["citation_validation"]["valid"] is False


def test_missing_full_text_does_not_call_generation():
    client = FakeClient()
    service = LegalAnswerService(
        FakeRetriever([("judgments/missing/0", 0.1, {})]),
        FakeTextStore({}),
        client,
    )

    result = service.answer("Question with missing text")

    assert result["retrieved_count"] == 1
    assert result["used_count"] == 0
    assert client.calls == []
    assert result["citation_validation"]["valid"] is False


def test_empty_question_is_rejected():
    service = LegalAnswerService(FakeRetriever([]), FakeTextStore({}), FakeClient())

    with pytest.raises(ValueError, match="question must not be empty"):
        service.answer("   ")
