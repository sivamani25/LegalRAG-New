"""Chat orchestration: retrieval -> grounded answer -> API response (M7).

Reuses ``legalrag.generation.service.LegalAnswerService`` for retrieval,
prompting, OpenRouter generation, and citation-label validation, then adds
related cases and explicit limitations for the API response.
"""

from __future__ import annotations

from typing import Any

from legalrag.generation.service import LegalAnswerService

EVIDENCE_EXCERPT_CHARS = 1500

SUPPORT_LIMITATION = (
    "Citation labels were checked against the supplied passages; substantive "
    "legal support for each claim has not been verified."
)
ADVICE_LIMITATION = (
    "This response is general legal information, not legal advice or a guarantee "
    "of any outcome. Consult a qualified lawyer for your situation."
)
NO_EVIDENCE_LIMITATION = (
    "No supporting passages were retrieved from the indexed corpus, so the "
    "answer is not evidence-grounded."
)
RETRIEVED_NO_TEXT_LIMITATION = (
    "Passages were retrieved but their full text could not be loaded, so no "
    "grounded answer was produced."
)


def _first(record: dict, *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return None


def _merge(base: dict, overlay: dict) -> dict:
    merged = dict(base or {})
    for key, value in (overlay or {}).items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def _excerpt(record: dict, limit: int) -> str:
    text = str(record.get("original_text") or record.get("text") or "").strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class ChatService:
    """Produce a grounded, citation-annotated chat response."""

    def __init__(
        self,
        retriever: Any,
        text_store: Any,
        client: Any,
        related_k: int = 5,
        excerpt_chars: int = EVIDENCE_EXCERPT_CHARS,
    ) -> None:
        self.retriever = retriever
        self.text_store = text_store
        self.answer_service = LegalAnswerService(retriever, text_store, client)
        self.related_k = related_k
        self.excerpt_chars = excerpt_chars

    def chat(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        result = self.answer_service.answer(question, conversation_history)

        evidence = [self._evidence_item(item) for item in result["evidence"]]
        related = self._related_cases(question)
        limitations = self._limitations(result, evidence)

        return {
            "question": question,
            "answer": result["answer"],
            "evidence": evidence,
            "related_cases": related,
            "limitations": limitations,
            "citation_validation": result["citation_validation"],
            "retrieved_count": result["retrieved_count"],
            "used_count": result["used_count"],
        }

    # ------------------------------------------------------------------
    def _related_cases(self, question: str) -> list[dict[str, Any]]:
        related_fn = getattr(self.retriever, "related_cases", None)
        if related_fn is None:
            return []
        raw = related_fn(question, self.related_k)
        ids = [chunk_id for chunk_id, _, _ in raw]
        texts = self.text_store.get_many(ids) if ids else {}

        cases: list[dict[str, Any]] = []
        for chunk_id, score, meta in raw:
            combined = _merge(meta, texts.get(chunk_id) or {})
            text = _excerpt(combined, self.excerpt_chars)
            if not text:
                continue
            cases.append({
                "chunk_id": chunk_id,
                "source_doc": _first(combined, "source_doc"),
                "section": _first(combined, "section"),
                "heading": _first(combined, "heading"),
                "court": _first(combined, "court"),
                "date": _first(combined, "date"),
                "retrieval_score": float(score),
                "text": text,
            })
        return cases

    def _evidence_item(self, record: dict) -> dict[str, Any]:
        return {
            "chunk_id": record.get("chunk_id"),
            "category": _first(record, "category"),
            "source_doc": _first(record, "source_doc"),
            "section": _first(record, "section"),
            "heading": _first(record, "heading"),
            "parent": _first(record, "parent"),
            "hierarchy_path": record.get("hierarchy_path") or [],
            "court": _first(record, "court"),
            "date": _first(record, "date"),
            "retrieval_score": record.get("retrieval_score"),
            "text": _excerpt(record, self.excerpt_chars),
        }

    def _limitations(self, result: dict, evidence: list[dict]) -> list[str]:
        limitations: list[str] = []

        if not evidence:
            if result.get("retrieved_count", 0) == 0:
                limitations.append(NO_EVIDENCE_LIMITATION)
            else:
                limitations.append(RETRIEVED_NO_TEXT_LIMITATION)

        for warning in result.get("citation_validation", {}).get("warnings", []):
            if warning not in limitations:
                limitations.append(warning)

        if not any("substantive legal support" in item.lower() for item in limitations):
            limitations.append(SUPPORT_LIMITATION)

        limitations.append(ADVICE_LIMITATION)
        return limitations
