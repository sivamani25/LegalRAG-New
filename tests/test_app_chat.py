"""Focused tests for the M7 /chat endpoint (no models, no network)."""

import pytest
from fastapi.testclient import TestClient

from legalrag.app.main import create_app
from legalrag.app.service import ChatService
from legalrag.generation.client import OpenRouterError


class FakeMultiRetriever:
    def __init__(self, results, related=None):
        self.results = results
        self.related = related or []
        self.related_calls = []

    def search(self, question):
        return self.results

    def related_cases(self, question, k):
        self.related_calls.append((question, k))
        return self.related[:k]


class FakeTextStore:
    def __init__(self, records):
        self.records = records

    def get_many(self, chunk_ids):
        return {cid: self.records[cid] for cid in chunk_ids if cid in self.records}


class FakeClient:
    def __init__(self, answer=None, raise_error=None):
        self.calls = []
        self.answer = answer or "The right is guaranteed. [S1]"
        self.raise_error = raise_error

    def generate(self, messages):
        self.calls.append(messages)
        if self.raise_error:
            raise self.raise_error
        return self.answer


CHUNK = "judgments/doc1/0"
RELEVANT_RECORD = {
    "chunk_id": CHUNK,
    "category": "judgments",
    "source_doc": "doc1",
    "original_text": "Courts consider the circumstances of each case.",
}
RELATED_CHUNK = "judgments/doc2/0"
RELATED_RECORD = {
    "chunk_id": RELATED_CHUNK,
    "category": "judgments",
    "source_doc": "doc2",
    "original_text": "A related case discusses the same question.",
}


def make_client(**kwargs):
    retriever = kwargs.pop("retriever", None) or FakeMultiRetriever(
        [(CHUNK, 0.25, {"category": "judgments"})],
        related=[(RELATED_CHUNK, 0.5, {"category": "judgments", "source_doc": "doc2"})],
    )
    text_store = kwargs.pop("text_store", None) or FakeTextStore(
        {CHUNK: RELEVANT_RECORD, RELATED_CHUNK: RELATED_RECORD}
    )
    generation = kwargs.pop("client", None) or FakeClient()
    service = ChatService(retriever, text_store, generation)
    app = create_app(chat_service=service, frontend_dir="__no_frontend__")
    return TestClient(app), generation


def test_rejects_blank_question():
    client, _ = make_client()
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_successful_response_shape():
    client, generation = make_client()

    response = client.post("/chat", json={"question": "How is bail considered?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"][0]["chunk_id"] == CHUNK
    assert body["evidence"][0]["category"] == "judgments"
    assert len(generation.calls) == 1
    assert "[S1]" in body["answer"]
    assert body["citation_validation"]["valid"] is True
    assert body["citation_validation"]["support_verified"] is False
    assert body["related_cases"][0]["chunk_id"] == RELATED_CHUNK
    assert any("substantive legal support" in item.lower() for item in body["limitations"])
    assert any("not legal advice" in item.lower() for item in body["limitations"])


def test_fabricated_citation_is_flagged():
    client, _ = make_client(client=FakeClient(answer="The rule is settled. [S9]"))

    body = client.post("/chat", json={"question": "What is the rule?"}).json()

    assert body["citation_validation"]["valid"] is False
    assert body["citation_validation"]["unknown_sources"] == ["S9"]


def test_parenthesized_citation_is_normalized_and_validated():
    client, _ = make_client(client=FakeClient(answer="The right is guaranteed (S1)."))

    body = client.post("/chat", json={"question": "Which right?"}).json()

    assert "[S1]" in body["answer"]
    assert "(S1)" not in body["answer"]
    assert body["citation_validation"]["valid"] is True
    assert body["citation_validation"]["used_sources"] == ["S1"]


def test_no_evidence_skips_generation():
    retriever = FakeMultiRetriever([])
    client, generation = make_client(retriever=retriever, text_store=FakeTextStore({}))

    body = client.post("/chat", json={"question": "Unmatched question"}).json()

    assert body["used_count"] == 0
    assert body["evidence"] == []
    assert generation.calls == []
    assert any("no supporting passages" in item.lower() for item in body["limitations"])


def test_retrieval_failure_returns_502():
    class BoomRetriever(FakeMultiRetriever):
        def search(self, question):
            raise RuntimeError("index exploded")

    client, _ = make_client(retriever=BoomRetriever([]))

    response = client.post("/chat", json={"question": "Anything"})

    assert response.status_code == 502


def test_generation_failure_returns_503():
    client, _ = make_client(client=FakeClient(raise_error=OpenRouterError("no key")))

    response = client.post("/chat", json={"question": "Anything"})

    assert response.status_code == 503


def test_response_does_not_leak_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET-KEY-12345")
    client, _ = make_client()

    response = client.post("/chat", json={"question": "Article 21?"})

    assert "SECRET-KEY-12345" not in response.text
