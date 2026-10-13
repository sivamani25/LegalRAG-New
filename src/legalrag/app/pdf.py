"""Minimal, dependency-free PDF export for LegalRAG answers (M8).

Renders the same data returned by ``/chat`` into a simple, readable,
multi-page PDF using only the standard library (Helvetica, ASCII-sanitised
text). It never includes secrets — only the answer, citations/metadata,
limitations, and related cases.
"""

from __future__ import annotations

from typing import Any

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 50
LINE_HEIGHT = 14
FONT_SIZE = 10
MAX_CHARS = 92
LINES_PER_PAGE = 52
FOOTER = (
    "Citation labels were checked only for format and existence; substantive "
    "legal support is not verified. Informational only - not legal advice."
)


def _ascii(value: Any) -> str:
    return str(value if value is not None else "").encode("ascii", "replace").decode("ascii")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int = MAX_CHARS) -> list[str]:
    lines: list[str] = []
    for raw in _ascii(text).split("\n"):
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            if cut <= 0:
                cut = width
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip(" ")
        lines.append(raw)
    return lines


def build_lines(data: dict[str, Any]) -> list[str]:
    """Flatten a /chat response payload into wrapped PDF text lines."""
    out: list[str] = []
    out.append("LegalRAG - Answer")
    out.append("=" * 60)
    out.append("")
    out.extend(_wrap("Question: " + str(data.get("question", ""))))
    out.append("")
    out.append("Answer")
    out.append("-" * 60)
    out.extend(_wrap(str(data.get("answer", ""))))
    out.append("")

    evidence = data.get("evidence") or []
    out.append("Cited authorities")
    out.append("-" * 60)
    if not evidence:
        out.append("No cited passages were returned.")
    else:
        for i, item in enumerate(evidence, start=1):
            meta_bits = [
                f"S{i}",
                _ascii(item.get("category")),
                _ascii(item.get("source_doc")),
            ]
            if item.get("section"):
                meta_bits.append("section " + _ascii(item.get("section")))
            if item.get("heading"):
                meta_bits.append(_ascii(item.get("heading")))
            if item.get("court"):
                meta_bits.append(_ascii(item.get("court")))
            if item.get("date"):
                meta_bits.append(_ascii(item.get("date")))
            out.extend(_wrap("[" + "] [".join(b for b in meta_bits if b) + "]"))
            out.extend(_wrap("  " + str(item.get("text", ""))))
            out.append("")

    related = data.get("related_cases") or []
    out.append("Related cases")
    out.append("-" * 60)
    if not related:
        out.append("No related cases were found.")
    else:
        for item in related:
            bits = [_ascii(item.get("source_doc"))]
            if item.get("section"):
                bits.append("section " + _ascii(item.get("section")))
            if item.get("court"):
                bits.append(_ascii(item.get("court")))
            if item.get("date"):
                bits.append(_ascii(item.get("date")))
            out.extend(_wrap("- " + " | ".join(b for b in bits if b)))
            out.extend(_wrap("  " + str(item.get("text", ""))))
            out.append("")

    out.append("Limitations")
    out.append("-" * 60)
    for limitation in (data.get("limitations") or ["(none reported)"]):
        out.extend(_wrap("- " + str(limitation)))
    out.append("")
    out.extend(_wrap(FOOTER))
    return out


def _content_stream(page_lines: list[str]) -> bytes:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{MARGIN} {PAGE_HEIGHT - MARGIN} Td"]
    for index, line in enumerate(page_lines):
        if index:
            parts.append(f"0 -{LINE_HEIGHT} Td")
        parts.append(f"({_escape(_ascii(line))}) Tj")
    parts.append("ET")
    body = "\n".join(parts).encode("latin-1", "replace")
    return b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream"


def build_pdf(data: dict[str, Any]) -> bytes:
    """Return a minimal valid PDF (bytes) for a /chat response payload."""
    lines = build_lines(data)
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)] or [[]]
    n_pages = len(pages)

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{4 + 2 * k} 0 R" for k in range(n_pages))
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for k, page_lines in enumerate(pages):
        page_num = 4 + 2 * k
        content_num = page_num + 1
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>"
        ).encode()
        objects[content_num] = _content_stream(page_lines)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in sorted(objects):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)
