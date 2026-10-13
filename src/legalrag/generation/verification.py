"""Deterministic citation-reference checks for generated legal answers."""

from __future__ import annotations

import re
from typing import Any


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

# Clearly-equivalent source labels written with parentheses, e.g. "(S1)" or
# "( s2 )". Only a bare ``S<digits>`` inside parentheses is normalized; other
# parenthesised text is left untouched.
_PAREN_SOURCE_PATTERN = re.compile(r"\(\s*([Ss]\d+)\s*\)")


def normalize_source_labels(answer: str) -> str:
    """Rewrite parenthesised source labels ``(S1)`` as bracketed ``[S1]``.

    This only touches unambiguous ``S<digits>`` labels; it never promotes
    arbitrary text to a citation.
    """
    if not isinstance(answer, str):
        return answer
    return _PAREN_SOURCE_PATTERN.sub(lambda m: f"[{m.group(1).upper()}]", answer)


def validate_citations(
    answer: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check source-label references against the evidence supplied to the prompt."""
    if not isinstance(answer, str) or not answer.strip():
        return {
            "valid": False,
            "has_citations": False,
            "used_sources": [],
            "unknown_sources": [],
            "warnings": ["The answer is empty."],
        }

    # Prompt labels are assigned in evidence order, starting at S1.
    valid_sources = {f"S{i}" for i in range(1, len(evidence) + 1)}
    used_sources = sorted(
        set(CITATION_PATTERN.findall(answer)),
        key=lambda value: int(value[1:]),
    )
    unknown_sources = [source for source in used_sources if source not in valid_sources]

    warnings = []
    if not used_sources:
        warnings.append("The answer contains no source citations.")
    if unknown_sources:
        warnings.append(
            "The answer references source labels not supplied to the prompt: "
            + ", ".join(f"[{source}]" for source in unknown_sources)
        )

   
    if used_sources and not unknown_sources:
        warnings.append(
            "Citation labels are valid, but substantive support "
            "for each legal claim has not been verified."
        )

    return {
        "valid": bool(used_sources) and not unknown_sources,
        "has_citations": bool(used_sources),
        "used_sources": used_sources,
        "unknown_sources": unknown_sources,
        "support_verified": False,
        "warnings": warnings,
    }
