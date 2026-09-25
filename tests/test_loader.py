import pytest
from pathlib import Path
from legalrag.ingest.loader import (
    load_document, load_documents, UnsupportedFormatError, Document
)

def test_load_txt(tmp_path: Path):
    p = tmp_path / "test_doc.txt"
    p.write_text("Hello World", encoding="utf-8")
    doc = load_document(p, category="acts", jurisdiction="India")
    assert doc.source_doc == "test_doc"
    assert doc.text == "Hello World"
    assert doc.format == ".txt"
    assert doc.category == "acts"
    assert doc.jurisdiction == "India"

def test_load_md(tmp_path: Path):
    p = tmp_path / "test.md"
    p.write_text("# Title", encoding="utf-8")
    doc = load_document(p)
    assert doc.format == ".md"

def test_load_pdf_malformed_raises(tmp_path: Path):
    pytest.importorskip("pdfplumber")
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        load_document(p)

def test_load_unsupported_ext(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c")
    with pytest.raises(UnsupportedFormatError):
        load_document(p)

def test_load_latin1_encoding(tmp_path: Path):
    p = tmp_path / "latin.txt"
    # Write some bytes that are valid latin-1 but invalid utf-8
    p.write_bytes(b"Caf\xe9")
    doc = load_document(p)
    assert doc.text == "Café"

def test_load_documents_sorted(tmp_path: Path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "c.txt").write_text("c")
    docs = load_documents(tmp_path)
    assert [d.source_doc for d in docs] == ["a", "b", "c"]

def test_load_documents_filter_extensions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.md").write_text("b")
    docs = load_documents(tmp_path, extensions={".md"})
    assert len(docs) == 1
    assert docs[0].source_doc == "b"

def test_load_documents_missing_dir(tmp_path: Path):
    docs = load_documents(tmp_path / "nonexistent")
    assert docs == []
