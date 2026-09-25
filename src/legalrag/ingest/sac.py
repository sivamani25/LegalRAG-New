"""Summary-Augmented Chunking (SAC).

SAC prepends a short per-document fingerprint (≈150 chars) to every chunk
before embedding.  This addresses Document-Level Retrieval Mismatch (DRM):
the tendency of dense retrievers to return chunks from the wrong document
when many documents share boilerplate text.

Pipeline contract
-----------------
1. Call ``summarize_document(text, ...)`` once per source document to obtain
   a ``SummaryResult``.
2. Call ``apply_sac_to_document(chunks, summary)`` to mutate each chunk's
   ``text``, ``doc_summary``, and ``sac_applied`` fields in place.

LLM injection (M5)
------------------
The LLM path accepts an arbitrary ``Callable[[str], str]`` (``llm_func``).
M1 never imports or calls any LLM client.  In M5, the caller injects an
OpenRouter-backed callable.  Tests inject a simple fake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .chunker import Chunk

SAC_SEPARATOR: str = "\n\n[Document summary]\n"

_DOC_TYPE_HINTS = (
    "agreement", "contract", "non-disclosure", "confidentiality", "nda",
    "licen", "employment", "purchase", "lease", "privacy policy",
)

# Generic names that appear in contracts but are not useful as party identifiers.
_GENERIC_QUOTED: frozenset[str] = frozenset({
    "agreement", "the agreement", "this agreement", "contract", "the contract",
    "party", "the party", "the parties", "parties",
    "confidential information", "effective date", "termination",
    "the termination", "the employee", "the employer",
    "document", "the document", "shall", "secret",
})


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    """Result of a document summary call.

    Attributes:
        summary:  The summary string (may be empty for empty docs).
        strategy: ``"heuristic"`` (default) or ``"llm"`` when an LLM was used.
    """

    summary: str
    strategy: str = "heuristic"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_document(
    text: str,
    max_chars: int = 150,
    use_llm: bool = False,
    llm_func: Callable[[str], str] | None = None,
    category: str = "",
) -> SummaryResult:
    """Generate a short document fingerprint summary.

    Args:
        text:      Full document text.
        max_chars: Approximate maximum length for the heuristic summary.
                   LLM summaries are *not* truncated by this value.
        use_llm:   If ``True``, call *llm_func* and return its result.
                   Raises ``ValueError`` if *llm_func* is ``None``.
        llm_func:  Callable ``(doc_text) -> summary_string``.  M1 never
                   provides a real implementation; tests inject a fake callable.
        category:  Corpus category hint for the heuristic (e.g. ``"acts"``).

    Returns:
        A ``SummaryResult`` with the summary string and strategy label.

    Raises:
        ValueError: If ``use_llm=True`` but ``llm_func`` is ``None``.
    """
    if use_llm:
        if llm_func is None:
            raise ValueError(
                "llm_func must be provided when use_llm=True; "
                "inject an OpenRouter callable from the generation module (M5)"
            )
        return SummaryResult(summary=llm_func(text), strategy="llm")

    return SummaryResult(
        summary=_heuristic_summary(text, max_chars, category),
        strategy="heuristic",
    )


def apply_sac_to_document(
    chunks: list[Chunk],
    summary: str,
    separator: str = SAC_SEPARATOR,
) -> list[Chunk]:
    """Prepend the document summary to every chunk of one document, in place.

    After this call:
    - ``chunk.text = summary + separator + original_text``
    - ``chunk.doc_summary = summary``
    - ``chunk.original_text`` is unchanged
    - ``chunk.sac_applied = True``

    Args:
        chunks:    All chunks from a single source document.
        summary:   Summary string from ``summarize_document``.
        separator: String inserted between summary and chunk body.

    Returns:
        The same *chunks* list (mutated in place).

    Raises:
        ValueError: If any chunk already has ``sac_applied=True``.
    """
    for chunk in chunks:
        if chunk.sac_applied:
            raise ValueError(
                f"SAC has already been applied to chunk {chunk.chunk_id!r}; "
                "call apply_sac_to_document at most once per document"
            )
        chunk.sac_applied = True
        chunk.doc_summary = summary
        chunk.text = summary + separator + chunk.text
    return chunks


# ---------------------------------------------------------------------------
# Heuristic summarizer — deterministic, offline, category-aware
# ---------------------------------------------------------------------------

def _heuristic_summary(text: str, max_chars: int, category: str) -> str:
    text = text.strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if category == "acts":
        return _act_summary(lines, max_chars)
    if category == "constitution":
        return _truncate("Constitution of India — supreme law of the land", max_chars)
    if category == "judgments":
        return _judgment_summary(lines, max_chars)
    if category in ("contracts",):
        return _contract_summary(lines, max_chars)
    return _generic_summary(lines, max_chars)


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _act_summary(lines: list[str], max_chars: int) -> str:
    title = ""
    year = ""
    joined = " ".join(lines[:10])

    for line in lines[:6]:
        if re.search(r"\b(?:act|code)\b", line, re.IGNORECASE):
            title = line.strip()
            break
    if not title:
        title = lines[0][:80] if lines else ""

    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", joined)
    if m:
        year = m.group(1)

    parts = [title]
    if year and year not in title:
        parts.append(f"({year})")
    return _truncate(" ".join(p for p in parts if p), max_chars)


def _judgment_summary(lines: list[str], max_chars: int) -> str:
    court = ""
    parties = ""
    year = ""
    joined = " ".join(lines[:15])

    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", joined)
    if m:
        year = m.group(1)

    for line in lines[:10]:
        if re.search(r"supreme court|high court", line, re.IGNORECASE):
            court = line.strip()[:60]
            break

    pm = re.search(
        r"([A-Z][A-Za-z ]{2,40}?)\s+[Vv](?:s\.?|ersus)\s+([A-Z][A-Za-z ]{2,40})",
        joined,
    )
    if pm:
        parties = f"{pm.group(1).strip()} v. {pm.group(2).strip()}"

    parts: list[str] = []
    if court:
        parts.append(court)
    if parties:
        parts.append(parties)
    if year:
        parts.append(f"({year})")
    summary = " — ".join(parts) if parts else (lines[0][:max_chars] if lines else "")
    return _truncate(summary, max_chars)


def _contract_summary(lines: list[str], max_chars: int) -> str:
    doc_type = ""
    lower_head = " ".join(lines[:5]).lower()
    for hint in _DOC_TYPE_HINTS:
        if hint in lower_head:
            doc_type = hint.replace("-", " ").title()
            break

    parties = _extract_quoted_parties(" ".join(lines[:20]))

    parts: list[str] = []
    if doc_type:
        parts.append(doc_type)
    if parties:
        parts.append("between " + " and ".join(parties))

    summary = " ".join(parts).strip() or (lines[0] if lines else "")
    return _truncate(summary, max_chars)


def _generic_summary(lines: list[str], max_chars: int) -> str:
    doc_type = ""
    lower_head = " ".join(lines[:5]).lower()
    for hint in _DOC_TYPE_HINTS:
        if hint in lower_head:
            doc_type = hint.replace("-", " ").title()
            break
    summary = doc_type or (lines[0] if lines else "")
    return _truncate(summary, max_chars)


def _extract_quoted_parties(text: str) -> list[str]:
    parties: list[str] = []
    for name in re.findall(r'"([^"]{2,60})"', text):
        name = name.strip()
        if name and name.lower() not in _GENERIC_QUOTED:
            parties.append(name)
        if len(parties) >= 2:
            break
    return parties[:2]
