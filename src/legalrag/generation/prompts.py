"""Evidence-grounded prompt construction for Indian LegalRAG."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are the answer-generation component of LegalRAG, an Indian
legal information retrieval system.

Rules:
1. Answer using only the supplied evidence. Do not invent laws, sections,
   case names, courts, dates, quotations, facts, or outcomes.
2. Cite supporting passages with square-bracketed labels exactly as given,
   e.g. [S1], [S2]. Always use square brackets — never (S1), [s1], or plain S1.
3. Do not treat a retrieved passage as support for a claim it does not establish.
4. If evidence is incomplete, conflicting, outdated, or irrelevant, say so clearly.
5. Distinguish what a cited judgment says from a general legal rule.
6. If the question assumes a fact or legal premise not established by the
   evidence, identify that limitation instead of accepting it as true.
7. Do not claim to have checked current law unless current authoritative
   material is included in the evidence.
8. Explain the answer clearly and identify practical uncertainty where relevant.
9. Provide general legal information, not a guarantee of a legal outcome.
10. End with a short note that the response is informational and not a
    substitute for advice from a qualified lawyer.

Use this structure when appropriate:
- Answer
- Legal reasoning
- Relevant authorities
- Limitations or missing information

Every legal proposition that depends on retrieved evidence should have one
or more source labels. Write every label as [S#] with square brackets.
Never fabricate a source label."""


def build_legal_messages(
    question: str,
    evidence: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages from a question and retrieved evidence.

    Evidence items should contain a chunk_id and readable text. Optional
    metadata such as source_doc, court, date, section, and category is included
    only when present.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty.")

    if not evidence:
        evidence_text = (
            "No retrieved legal evidence was provided. Explain that the "
            "available evidence is insufficient and do not guess."
        )
    else:
        blocks = []
        for i, item in enumerate(evidence, start=1):
            text = str(item.get("text") or item.get("original_text") or "").strip()
            if not text:
                continue

            label = f"S{i}"
            metadata_fields = (
                ("category", "category"),
                ("source_doc", "source_doc"),
                ("court", "court"),
                ("date", "date"),
                ("section", "section"),
                ("heading", "heading"),
            )
            metadata = [
                f"{display}: {item[key]}"
                for key, display in metadata_fields
                if item.get(key) not in (None, "")
            ]

            header = f"[{label}]"
            if metadata:
                header += "\n" + "\n".join(metadata)

            blocks.append(f"{header}\nPassage:\n{text}")

        evidence_text = "\n\n".join(blocks) if blocks else (
            "Retrieved records were provided, but none contained readable text. "
            "Explain that the available evidence is insufficient and do not guess."
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for turn in (conversation_history or []):
        role = turn.get("role")
        content = turn.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({
        "role": "user",
        "content": (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED LEGAL EVIDENCE:\n{evidence_text}\n\n"
            "Answer the question using the rules in the system message. "
            "Cite the evidence labels exactly as provided. If the evidence "
            "does not establish the answer, say what is missing."
        ),
    })
    return messages
