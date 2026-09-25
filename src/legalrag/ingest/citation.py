"""Citation extraction for legal documents.

Extracts structured citations from chunk text using compiled regex patterns.
All pattern matching is offline — no network calls, no model downloads.

The ``extract_citations`` function is the primary public API.  It returns a
list of ``Citation`` objects each carrying three fields:

    category  — citation type (``"case_law"``, ``"statute"``, etc.)
    doc       — referenced document identifier or reporter string
    section   — specific provision/article/clause; ``""`` when not applicable

Extraction is enabled only for categories listed in *enabled_categories*
(default: ``{"judgments", "contracts"}``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """A normalized citation extracted from legal text.

    Attributes:
        category: Citation type:
            ``"case_law"``        — court decisions (AIR, SCC, SCR, INSC, …)
            ``"statute"``         — references to Acts/Codes (Section N of Act)
            ``"constitutional"``  — references to Articles of the Indian Constitution
            ``"schedule"``        — references to Schedules
            ``"contract_clause"`` — intra-document clause references
        doc: Referenced document id (reporter string, act name, or abbreviation).
             ``""`` when not determinable from context (e.g. bare Schedule refs).
        section: Specific provision, article, clause, or schedule number.
                 ``""`` when not applicable (e.g. bare case citations).
    """

    category: str
    doc: str
    section: str

    def to_dict(self) -> dict[str, str]:
        return {"category": self.category, "doc": self.doc, "section": self.section}


# ---------------------------------------------------------------------------
# Compiled patterns — module-level (compiled once at import)
# ---------------------------------------------------------------------------

# Case law — AIR reporter (covers both Supreme Court "SC" and High Courts)
_AIR = re.compile(r"\bAIR\s+\d{4}\s+[A-Z]{2,6}\s+\d+\b")

# Case law — SCC with bracketed year: "(2019) 5 SCC 200"
_SCC_BRACKETED = re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+")

# Case law — SCC bare: "2020 SCC 15"  (word-boundary to avoid matching inside bracketed)
_SCC_BARE = re.compile(r"(?<!\()\b\d{4}\s+SCC\s+\d+\b")

# Case law — SCR: "1985 SCR 400"
_SCR = re.compile(r"\b\d{4}\s+SCR\s+\d+\b")

# Case law — neutral Indian citation: "2023 INSC 350"
_INSC = re.compile(r"\b\d{4}\s+INSC\s+\d+\b")

# Statute — "Section 300 of the Indian Penal Code"
_SECTION_OF_ACT = re.compile(
    r"\b[Ss]ection\s+(\d+[A-Z]?)\s+of\s+(?:the\s+)?"
    r"([A-Z][A-Za-z ]{2,60}?(?:Act|Code|Rules|Regulations|Order))\b"
)

# Statute — "Section 302 IPC"  (short abbreviation, 2–6 uppercase letters)
_SECTION_ABBR = re.compile(
    r"\b[Ss]ection\s+(\d+[A-Z]?)\s+([A-Z]{2,6})\b"
)

# Constitutional — "Article 21(1)", "Article 226"
_ARTICLE = re.compile(r"\b[Aa]rticle\s+(\d+[A-Z]?)(?:\s*\((\d+)\))?")

# Schedule references — "Schedule VII", "Schedule 1"
_SCHEDULE = re.compile(r"\b[Ss]chedule\s+([IVXLCDM]+|\d+)\b")

# Contract clause references — "Clause 3.1", "Clause 7"
_CLAUSE = re.compile(r"\b[Cc]lause\s+(\d+(?:\.\d+)*)\b")

# Default enabled categories
_DEFAULT_ENABLED: frozenset[str] = frozenset({"judgments", "contracts"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dedup(citations: list[Citation]) -> list[Citation]:
    """Remove duplicates, keeping first-occurrence order."""
    seen: set[tuple[str, str, str]] = set()
    result: list[Citation] = []
    for c in citations:
        key = (c.category, c.doc.lower(), c.section.lower())
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_citations(
    text: str,
    category: str = "",
    enabled_categories: frozenset[str] | None = None,
) -> list[Citation]:
    """Extract and normalize citations from legal text.

    Args:
        text:               Chunk or document text to scan.
        category:           Corpus category of the source document.
        enabled_categories: Categories for which extraction is active.
                            Defaults to ``{"judgments", "contracts"}``.

    Returns:
        Deduplicated list of ``Citation`` objects in order of first appearance.
        Returns ``[]`` immediately if *category* is not in *enabled_categories*,
        or if the text contains no recognizable citation patterns.
    """
    if enabled_categories is None:
        enabled_categories = _DEFAULT_ENABLED
    if category and category not in enabled_categories:
        return []

    found: list[Citation] = []

    # -- Case law --
    for m in _AIR.finditer(text):
        found.append(Citation("case_law", m.group(0).strip(), ""))
    for m in _SCC_BRACKETED.finditer(text):
        found.append(Citation("case_law", m.group(0).strip(), ""))
    for m in _SCC_BARE.finditer(text):
        found.append(Citation("case_law", m.group(0).strip(), ""))
    for m in _SCR.finditer(text):
        found.append(Citation("case_law", m.group(0).strip(), ""))
    for m in _INSC.finditer(text):
        found.append(Citation("case_law", m.group(0).strip(), ""))

    # -- Statutes --
    for m in _SECTION_OF_ACT.finditer(text):
        found.append(Citation("statute", m.group(2).strip(), m.group(1)))
    for m in _SECTION_ABBR.finditer(text):
        # Only capture if not already covered by _SECTION_OF_ACT at same position
        found.append(Citation("statute", m.group(2), m.group(1)))

    # -- Constitutional --
    for m in _ARTICLE.finditer(text):
        art = m.group(1)
        sub = m.group(2)
        section = f"{art}({sub})" if sub else art
        found.append(Citation("constitutional", "Constitution of India", section))

    # -- Schedules --
    for m in _SCHEDULE.finditer(text):
        found.append(Citation("schedule", "", m.group(1)))

    # -- Contract clauses --
    for m in _CLAUSE.finditer(text):
        found.append(Citation("contract_clause", "", m.group(1)))

    return _dedup(found)

# ---------------------------------------------------------------------------
# Internal References
# ---------------------------------------------------------------------------

_INTERNAL_REF = re.compile(
    r"\b(Exhibit|Schedule|Annexure|Appendix)\s+([A-Z0-9IVXLCDM]+)\b",
    re.IGNORECASE
)

def extract_internal_refs(text: str) -> list[str]:
    """Extract internal document references (e.g., 'Exhibit A', 'Annexure 1').

    Returns a deduplicated list of strings.
    """
    found = []
    seen = set()
    for m in _INTERNAL_REF.finditer(text):
        ref_type = m.group(1).capitalize()
        ref_id = m.group(2).upper()
        ref_str = f"{ref_type} {ref_id}"
        if ref_str not in seen:
            seen.add(ref_str)
            found.append(ref_str)
    return found
