"""Focused tests for the M8 PDF export (no models, no network)."""

from fastapi.testclient import TestClient

from legalrag.app.main import create_app
from legalrag.app.pdf import build_pdf
from legalrag.app.service import ChatService

CHUNK = "constitution/constitution_of_india/581"
EVIDENCE = {
    "chunk_id": CHUNK,
    "category": "constitution",
    "source_doc": "constitution_of_india",
    "section": "21",
    "heading": "21",
    "text": "Protection of life and personal liberty.",
}
RELATED = {
    "chunk_id": "judgments/j/0",
    "category": "judgments",
    "source_doc": "j",
    "text": "A related case about Article 21.",
}

PAYLOAD = {
    "question": "What does Article 21 guarantee?",
    "answer": "It guarantees life and personal liberty. [S1]",
    "evidence": [EVIDENCE],
    "related_cases": [RELATED],
    "limitations": ["Citation labels were checked; support is not verified.", "Not legal advice."],
    "citation_validation": {
        "valid": True, "has_citations": True, "used_sources": ["S1"],
        "unknown_sources": [], "support_verified": False, "warnings": [],
    },
    "retrieved_count": 1,
    "used_count": 1,
}


class FakeMultiRetriever:
    def search(self, question):
        return [(CHUNK, 0.5, {"category": "constitution"})]

    def related_cases(self, question, k):
        return [("judgments/j/0", 0.4, {"category": "judgments"})]


class FakeTextStore:
    def get_many(self, chunk_ids):
        return {cid: rec for cid, rec in
                [(CHUNK, {**EVIDENCE, "original_text": EVIDENCE["text"]}),
                 ("judgments/j/0", {**RELATED, "original_text": RELATED["text"]})]
                if cid in chunk_ids}


class FakeClient:
    def generate(self, messages):
        return "It guarantees life and personal liberty. [S1]"


def test_build_pdf_is_readable_and_complete():
    raw = build_pdf(PAYLOAD)

    assert raw.startswith(b"%PDF")
    assert b"%%EOF" in raw
    for needle in (b"Article 21", b"[S1]", b"constitution", b"constitution_of_india",
                   b"Cited authorities", b"Related cases", b"Limitations",
                   b"not legal advice"):
        assert needle in raw, needle


def test_build_pdf_paginates_long_answers():
    long_payload = dict(PAYLOAD, answer="word " * 3000)
    raw = build_pdf(long_payload)
    # More than one /Page object means the content was paginated.
    assert raw.count(b"/Type /Page ") >= 2


def make_client():
    service = ChatService(FakeMultiRetriever(), FakeTextStore(), FakeClient())
    app = create_app(chat_service=service, frontend_dir="__no_frontend__")
    return TestClient(app)


def test_pdf_endpoint_returns_pdf():
    response = make_client().post("/chat/pdf", json={"question": "What does Article 21 guarantee?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert b"life and personal liberty" in response.content
    assert response.headers["content-disposition"].startswith("attachment")


def test_pdf_endpoint_validates_empty_question():
    assert make_client().post("/chat/pdf", json={"question": ""}).status_code == 422


def test_pdf_does_not_leak_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET-PDF-123")
    response = make_client().post("/chat/pdf", json={"question": "Article 21?"})
    assert b"SECRET-PDF-123" not in response.content
