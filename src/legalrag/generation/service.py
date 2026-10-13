"""Evidence-grounded legal answer service."""

from __future__ import annotations

from typing import Any

from .prompts import build_legal_messages
from .verification import normalize_source_labels, validate_citations


class LegalAnswerService:
    """Connect retrieval, full-text lookup, prompting, and generation."""

    def __init__(self, retriever: Any, text_store: Any, client: Any):
        self.retriever = retriever
        self.text_store = text_store
        self.client = client

    def answer(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Answer a legal question and return evidence and citation checks."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        retrieved = self.retriever.search(question)
        if not retrieved:
            answer_text = (
                "I could not find relevant passages in the available corpus. "
                "I cannot provide a reliable answer without supporting evidence."
            )
            return {
                "answer": answer_text,
                "evidence": [],
                "retrieved_count": 0,
                "used_count": 0,
                "citation_validation": validate_citations(answer_text, []),
            }

        chunk_ids = [chunk_id for chunk_id, _, _ in retrieved]
        full_records = self.text_store.get_many(chunk_ids)

        evidence: list[dict[str, Any]] = []
        for chunk_id, score, retrieval_meta in retrieved:
            record = full_records.get(chunk_id)
            if not record:
                continue

            item = dict(record)
            for key, value in (retrieval_meta or {}).items():
                if key not in item or item[key] in (None, "", []):
                    item[key] = value

            text = str(item.get("original_text") or item.get("text") or "").strip()
            if not text:
                continue

            item["chunk_id"] = chunk_id
            item["retrieval_score"] = float(score)
            evidence.append(item)

        if not evidence:
            answer_text = (
                "Relevant records were retrieved, but their full text could not "
                "be loaded. I cannot provide a reliable answer until the evidence "
                "can be inspected."
            )
            return {
                "answer": answer_text,
                "evidence": [],
                "retrieved_count": len(retrieved),
                "used_count": 0,
                "citation_validation": validate_citations(answer_text, []),
            }

        messages = build_legal_messages(
            question,
            evidence,
            conversation_history=conversation_history,
        )
        answer_text = self.client.generate(messages)
        answer_text = normalize_source_labels(answer_text)
        citation_validation = validate_citations(answer_text, evidence)

        return {
            "answer": answer_text,
            "evidence": evidence,
            "retrieved_count": len(retrieved),
            "used_count": len(evidence),
            "citation_validation": citation_validation,
        }
