"""Load raw legal documents (PDF/TXT/MD) from a directory into Document objects.

Supports utf-8, utf-8-sig, and latin-1 encodings (tried in order).
PDF text is extracted via pdfplumber (imported lazily so the rest of the
package works even if pdfplumber is not yet installed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is not a supported document format."""


@dataclass(frozen=True)
class Document:
    """A single loaded source document.

    Attributes:
        source_path:  Absolute path to the original file.
        source_doc:   File stem — stable document id (e.g. ``"ipc_1860"``).
        text:         Full plain-text content of the document.
        format:       File extension: ``".pdf"``, ``".txt"``, or ``".md"``.
        category:     Corpus category inferred by the pipeline
                      (e.g. ``"acts"``, ``"judgments"``). May be ``""`` if unknown.
        jurisdiction: Jurisdiction string (e.g. ``"India"``). May be ``""`` if unknown.
    """

    source_path: str
    source_doc: str
    text: str
    format: str
    category: str
    jurisdiction: str


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    """Read a plain-text file, trying common encodings in order."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Should not be reached — latin-1 decodes any byte sequence
    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1,
        f"could not decode {path.name} with any supported encoding",
    )


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber (lazy import)."""
    import pdfplumber  # noqa: PLC0415 — lazy import keeps package importable without pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(
    path: str | Path,
    category: str = "",
    jurisdiction: str = "",
) -> Document:
    """Load a single file into a Document, dispatching on file extension.

    Args:
        path:         Path to the file to load.
        category:     Corpus category to attach (set by pipeline after inference).
        jurisdiction: Jurisdiction string to attach.

    Returns:
        A Document object.

    Raises:
        UnsupportedFormatError: If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"unsupported format {ext!r} for {path.name}; "
            f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    text = _read_pdf(path) if ext == ".pdf" else _read_text(path)
    return Document(
        source_path=str(path),
        source_doc=path.stem,
        text=text,
        format=ext,
        category=category,
        jurisdiction=jurisdiction,
    )


def load_documents(
    raw_dir: str | Path,
    extensions: set[str] | frozenset[str] | None = None,
    category: str = "",
    jurisdiction: str = "",
) -> list[Document]:
    """Load all supported documents from *raw_dir* (non-recursive, sorted).

    Sidecar files (``*.meta.yaml``) are skipped automatically because
    ``.yaml`` is not in the supported-extensions set.

    Args:
        raw_dir:      Directory containing raw documents.
        extensions:   Extension filter. Defaults to all supported extensions.
        category:     Category attached to every loaded document.
        jurisdiction: Jurisdiction attached to every loaded document.

    Returns:
        List of Document objects, alphabetically sorted by filename.
        Returns ``[]`` if *raw_dir* does not exist.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []

    allowed = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    documents: list[Document] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        documents.append(load_document(path, category=category, jurisdiction=jurisdiction))
    return documents
