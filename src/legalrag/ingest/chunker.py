"""Chunking strategies for legal documents.

Two strategies are provided:

``SentenceChunker``
    Splits on sentence-ending punctuation (``[.!?]``), groups sentences into
    windows bounded by *max_chars*, and enforces a *min_chars* floor before
    flushing.  Best for narrative text (Judgments, Legal_QA).

``PatternChunker``
    Structure-aware chunker that detects legal section delimiters (e.g.
    "Section 3", "Article 21(1)", "Chapter II") via configurable regex patterns.
    Preserves the full section hierarchy on each chunk as ``section``,
    ``parent``, ``heading``, and ``hierarchy_path``.  Best for structured
    statutes, contracts, and constitutional text.

``Chunker``
    Facade that selects an implementation from ``strategy`` and an optional
    named *preset* (read from the ``chunking.presets`` config block).

Chunk IDs are globally unique across all corpus categories:
    ``"{category}/{source_doc}/{chunk_index}"``
    Components are normalised (lowercase, unsafe chars removed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Chunk ID helpers
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r"[^a-z0-9_.-]")


def _normalize_id_component(s: str) -> str:
    """Lowercase; spaces/slashes/backslashes → ``_``; strip remaining unsafe chars."""
    s = s.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return _UNSAFE_RE.sub("", s) or "unknown"


def make_chunk_id(category: str, source_doc: str, index: int) -> str:
    """Build a globally-unique, normalized chunk ID.

    Format: ``"{category}/{source_doc}/{index}"``

    Example: ``"acts/ipc_1860/42"``
    """
    return (
        f"{_normalize_id_component(category)}"
        f"/{_normalize_id_component(source_doc)}"
        f"/{index}"
    )


# ---------------------------------------------------------------------------
# Chunk dataclass — full M1 schema
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk derived from a legal source document.

    Attributes:
        chunk_id:       Globally unique ID: ``"{category}/{source_doc}/{index}"``.
        source_doc:     File stem of the source file (stable document id).
        source_path:    Absolute path to the source file.
        category:       Corpus category (``"acts"``, ``"judgments"``, …).
        jurisdiction:   Jurisdiction string (``"India"`` or ``""``).
        text:           Embedded text.  When SAC is applied this is
                        ``doc_summary + SAC_SEPARATOR + original_text``.
        original_text:  Raw chunk text before any SAC prepending; always preserved.
        doc_summary:    SAC summary string; ``""`` when ``sac_applied`` is False.
        section:        Section label (e.g. ``"300"``, ``"21"``, ``"Part III"``);
                        ``None`` for SentenceChunker or preamble chunks.
        parent:         Label of the immediately enclosing section; ``None`` at root.
        heading:        First-line text of the section heading; ``None`` for
                        SentenceChunker or numeric sub-section labels.
        hierarchy_path: Labels from root to this section,
                        e.g. ``["Part III", "21"]``.
        cites:          Extracted citations as dicts (``{category, doc, section}``);
                        ``[]`` when extraction is disabled for this category.
        start_offset:   Character offset of this chunk in the source document text.
        end_offset:     Character offset of the end of this chunk.
        sac_applied:    True once ``apply_sac_to_document`` has been called.
    """

    # Identity
    chunk_id: str

    # Provenance
    source_doc: str
    source_path: str
    category: str
    jurisdiction: str

    # Content
    text: str
    original_text: str
    doc_summary: str

    # Hierarchy (PatternChunker fills; SentenceChunker leaves None/[])
    section: str | None
    parent: str | None
    heading: str | None
    hierarchy_path: list[str] = field(default_factory=list)

    # Citations
    cites: list[dict] = field(default_factory=list)

    # Offsets + SAC flag
    start_offset: int = 0
    end_offset: int = 0
    sac_applied: bool = False

    # Optional contract-specific metadata
    contract_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "source_path": self.source_path,
            "category": self.category,
            "jurisdiction": self.jurisdiction,
            "text": self.text,
            "original_text": self.original_text,
            "doc_summary": self.doc_summary,
            "section": self.section,
            "parent": self.parent,
            "heading": self.heading,
            "hierarchy_path": list(self.hierarchy_path),
            "cites": list(self.cites),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "sac_applied": self.sac_applied,
            "contract_meta": dict(self.contract_meta),
        }


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"(?<=[.!?])(?=\s+)")


class SentenceChunker:
    """Chunks text by grouping sentences up to *max_chars*.

    Sentences are split on ``[.!?]`` followed by whitespace.  Short sentences
    (below *min_chars*) are accumulated before flushing; long ones flush
    immediately if the window is already non-empty.
    """

    def __init__(self, max_chars: int = 1000, min_chars: int = 50) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        self.max_chars = max_chars
        self.min_chars = min_chars

    # ------------------------------------------------------------------
    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Return ``(sentence_text, char_offset)`` pairs, in order."""
        parts = _SENT_END_RE.split(text)
        result: list[tuple[str, int]] = []
        cursor = 0
        for part in parts:
            if not part:
                continue
            idx = text.find(part, cursor)
            result.append((part, idx))
            cursor = idx + len(part)
        return result

    def chunk(
        self,
        source_doc: str,
        text: str,
        category: str = "",
        jurisdiction: str = "",
        source_path: str = "",
    ) -> list[Chunk]:
        """Produce ``Chunk`` objects from *text*."""
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        pending: list[tuple[str, int]] = []   # (sentence, offset)
        pending_len = 0

        def _flush() -> None:
            nonlocal pending, pending_len
            if not pending:
                return
            start = pending[0][1]
            end = pending[-1][1] + len(pending[-1][0])
            body = "".join(s for s, _ in pending)
            chunk_id = make_chunk_id(category, source_doc, len(chunks))
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_doc=source_doc,
                source_path=source_path,
                category=category,
                jurisdiction=jurisdiction,
                text=body,
                original_text=body,
                doc_summary="",
                section=None,
                parent=None,
                heading=None,
                hierarchy_path=[],
                cites=[],
                start_offset=start,
                end_offset=end,
            ))
            pending = []
            pending_len = 0

        for sentence, offset in sentences:
            s_len = len(sentence)
            # If adding this sentence would exceed max_chars, flush first
            if pending and (pending_len + s_len) > self.max_chars:
                _flush()
            pending.append((sentence, offset))
            pending_len += s_len

        _flush()
        return chunks


# ---------------------------------------------------------------------------
# PatternChunker — structure-aware
# ---------------------------------------------------------------------------

# Named heading keywords (Section/Article/Chapter/Part/Schedule)
_NAMED_HEADING_RE = re.compile(
    r"^(?:Section|Article|Chapter|Part|Schedule)\s+",
    re.IGNORECASE,
)

# Parenthetical label: "(1)", "(a)", "(iv)"
_PAREN_LABEL_RE = re.compile(r"^\(([a-z0-9ivx]+)\)$", re.IGNORECASE)

# Numeric label possibly ending in dot: "3", "3.", "3A"
_NUMERIC_LABEL_RE = re.compile(r"^(\d+(?:\.\d+)*[A-Z]?)\.?\s*$")


def _normalize_label(raw: str) -> str:
    """Normalise a raw matched label string to a clean canonical form."""
    raw = raw.strip()
    # Parenthetical: "(1)" → "1", "(a)" → "a"
    m = _PAREN_LABEL_RE.match(raw)
    if m:
        return m.group(1)
    # Numeric with optional trailing dot: "3." → "3", "3.2" → "3.2"
    m = _NUMERIC_LABEL_RE.match(raw)
    if m:
        return m.group(1)
    # Named heading: keep as-is ("Section 3", "Part III", …)
    return raw


def _is_named_heading(label: str) -> bool:
    return bool(_NAMED_HEADING_RE.match(label))


def _line_end(text: str, pos: int) -> int:
    """Return the index one past the next newline after *pos*, or len(text)."""
    nl = text.find("\n", pos)
    return nl + 1 if nl != -1 else len(text)


class PatternChunker:
    """Structure-aware chunker using configurable delimiter patterns.

    Each delimiter entry in *delimiters* is a dict with keys:

    - ``pattern``: regex string (matched with ``re.MULTILINE``)
    - ``level``:   int hierarchy depth (lower = higher in tree; 1 = top)

    When multiple patterns match at the same position, the entry with the
    *highest* level number wins (deepest wins).

    Named headings (Section/Article/Chapter/Part/Schedule N) contribute their
    entire first line as ``heading``; the body starts on the *next* line.
    Parenthetical / numeric labels contribute the label text as ``heading``;
    the body starts immediately after the match on the *same* line.
    """

    _DEFAULT_DELIMITERS: list[dict] = [
        {"pattern": r"^[ \t]*(?:Section|Article)\s+\d+", "level": 1},
        {"pattern": r"^[ \t]*\d+\.\d+\.\d+",             "level": 3},
        {"pattern": r"^[ \t]*\d+\.\d+",                  "level": 2},
        {"pattern": r"^[ \t]*\d+\.",                      "level": 1},
        {"pattern": r"^[ \t]*\(\w+\)",                    "level": 3},
    ]

    def __init__(self, delimiters: Sequence[dict] | None = None) -> None:
        raw = list(delimiters) if delimiters is not None else self._DEFAULT_DELIMITERS
        self._compiled: list[tuple[re.Pattern, int]] = [
            (re.compile(d["pattern"], re.MULTILINE), int(d["level"]))
            for d in raw
        ]
        self._counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    def _find_boundaries(
        self, text: str
    ) -> list[tuple[str, int, int, int]]:
        """Find all delimiter matches in *text*.

        Returns a list of ``(label, start, label_end, level)`` tuples,
        sorted by position.  When two patterns match at the same start
        position the one with the highest level wins.
        """
        by_start: dict[int, tuple[str, int, int]] = {}  # start → (label, end, level)
        for pattern, level in self._compiled:
            for m in pattern.finditer(text):
                label = _normalize_label(m.group(0))
                start = m.start()
                existing = by_start.get(start)
                if existing is None or level > existing[2]:
                    by_start[start] = (label, m.end(), level)

        return [
            (label, start, end, level)
            for start, (label, end, level) in sorted(by_start.items())
        ]

    # ------------------------------------------------------------------
    def chunk(
        self,
        source_doc: str,
        text: str,
        category: str = "",
        jurisdiction: str = "",
        source_path: str = "",
    ) -> list[Chunk]:
        """Produce ``Chunk`` objects from *text*, preserving section hierarchy."""
        self._counter[source_doc] = 0
        boundaries = self._find_boundaries(text)
        chunks: list[Chunk] = []

        # Empty document
        if not text.strip():
            return chunks

        # No delimiters found → single unsectioned chunk
        if not boundaries:
            body = text.strip()
            chunks.append(self._make_chunk(
                source_doc, body, 0, len(text),
                section=None, parent=None, path=[], heading=None,
                category=category, jurisdiction=jurisdiction,
                source_path=source_path,
            ))
            return chunks

        # Preamble: text before the first delimiter
        first_start = boundaries[0][1]
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                chunks.append(self._make_chunk(
                    source_doc, preamble, 0, first_start,
                    section=None, parent=None, path=[], heading=None,
                    category=category, jurisdiction=jurisdiction,
                    source_path=source_path,
                ))

        # Stack: list of (label, level) for open ancestors
        stack: list[tuple[str, int]] = []

        def _pop_to(level: int) -> None:
            while stack and stack[-1][1] >= level:
                stack.pop()

        for i, (label, start, label_end, level) in enumerate(boundaries):
            _pop_to(level)
            parent = stack[-1][0] if stack else None
            path = [s for s, _ in stack] + [label]

            # Determine heading and body_start
            if _is_named_heading(label):
                heading_end = _line_end(text, start)
                heading = text[start:heading_end].strip()
                body_start = heading_end
            else:
                heading = label
                body_start = label_end

            body_end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
            body = text[body_start:body_end].strip()

            if body:
                chunks.append(self._make_chunk(
                    source_doc, body, body_start, body_end,
                    section=label, parent=parent, path=path, heading=heading,
                    category=category, jurisdiction=jurisdiction,
                    source_path=source_path,
                ))

            stack.append((label, level))

        return chunks

    # ------------------------------------------------------------------
    def _make_chunk(
        self,
        source_doc: str,
        body: str,
        start: int,
        end: int,
        section: str | None,
        parent: str | None,
        path: list[str],
        heading: str | None,
        category: str = "",
        jurisdiction: str = "",
        source_path: str = "",
    ) -> Chunk:
        idx = self._counter.get(source_doc, 0)
        self._counter[source_doc] = idx + 1
        return Chunk(
            chunk_id=make_chunk_id(category, source_doc, idx),
            source_doc=source_doc,
            source_path=source_path,
            category=category,
            jurisdiction=jurisdiction,
            text=body,
            original_text=body,
            doc_summary="",
            section=section,
            parent=parent,
            heading=heading,
            hierarchy_path=list(path),
            cites=[],
            start_offset=start,
            end_offset=end,
        )


# ---------------------------------------------------------------------------
# Chunker facade
# ---------------------------------------------------------------------------

class Chunker:
    """Selects a chunking implementation from config.

    Args:
        strategy:    ``"sentence"`` or ``"pattern"``.
        config:      ``chunking`` config block (dict). Used for sentence
                     ``max_chars``/``min_chars`` and pattern ``delimiters``
                     fallback.
        preset_name: Name of a preset in *presets* to use for the pattern chunker.
        presets:     ``chunking.presets`` config block (dict of preset dicts).

    Raises:
        ValueError: If *strategy* is not ``"sentence"`` or ``"pattern"``.
    """

    def __init__(
        self,
        strategy: str,
        config: dict | None = None,
        preset_name: str | None = None,
        presets: dict | None = None,
    ) -> None:
        config = config or {}
        strategy = strategy.lower().strip()
        if strategy not in {"sentence", "pattern"}:
            raise ValueError(
                f"unknown chunking strategy {strategy!r}; choose 'sentence' or 'pattern'"
            )
        self.strategy = strategy

        if strategy == "sentence":
            sent_cfg = config.get("sentence", {})
            self._impl: SentenceChunker | PatternChunker = SentenceChunker(
                max_chars=int(sent_cfg.get("max_chars", 1000)),
                min_chars=int(sent_cfg.get("min_chars", 50)),
            )
        else:
            # Resolve delimiters: preset > config.pattern.delimiters > built-in default
            delimiters = None
            if preset_name and presets and preset_name in presets:
                delimiters = presets[preset_name].get("delimiters")
            elif "pattern" in config:
                delimiters = config["pattern"].get("delimiters")
            self._impl = PatternChunker(delimiters)

    def chunk(
        self,
        source_doc: str,
        text: str,
        category: str = "",
        jurisdiction: str = "",
        source_path: str = "",
    ) -> list[Chunk]:
        """Delegate to the underlying chunker implementation."""
        return self._impl.chunk(
            source_doc=source_doc,
            text=text,
            category=category,
            jurisdiction=jurisdiction,
            source_path=source_path,
        )
