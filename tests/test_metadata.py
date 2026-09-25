import pytest
from pathlib import Path
from legalrag.ingest.metadata import (
    infer_category, infer_jurisdiction, load_sidecar, resolve_doc_meta
)

@pytest.fixture
def categories_cfg():
    return [
        {"name": "acts", "path": "acts", "default_jurisdiction": "India"},
        {"name": "judgments", "path": "judgments", "default_jurisdiction": "India"},
        {"name": "contracts", "path": "contracts", "default_jurisdiction": ""}
    ]

def test_infer_category_from_folder(categories_cfg):
    assert infer_category(Path("data/raw/acts/x.txt"), categories_cfg) == "acts"

def test_category_case_insensitive(categories_cfg):
    assert infer_category(Path("data/raw/Acts/x.txt"), categories_cfg) == "acts"

def test_category_sidecar_override(categories_cfg):
    assert infer_category(Path("data/raw/acts/x.txt"), categories_cfg, {"category": "judgments"}) == "judgments"

def test_jurisdiction_from_category_config(categories_cfg):
    assert infer_jurisdiction("acts", categories_cfg) == "India"
    assert infer_jurisdiction("contracts", categories_cfg) == ""

def test_jurisdiction_sidecar_override(categories_cfg):
    assert infer_jurisdiction("acts", categories_cfg, {"jurisdiction": "UK"}) == "UK"

def test_load_sidecar_present(tmp_path: Path):
    doc_path = tmp_path / "doc.txt"
    sidecar_path = tmp_path / "doc.meta.yaml"
    sidecar_path.write_text("category: acts\njurisdiction: UK\nyear: 2020", encoding="utf-8")
    data = load_sidecar(doc_path)
    assert data == {"category": "acts", "jurisdiction": "UK", "year": 2020}

def test_load_sidecar_absent(tmp_path: Path):
    assert load_sidecar(tmp_path / "doc.txt") == {}

def test_load_sidecar_malformed_yaml(tmp_path: Path):
    doc_path = tmp_path / "doc.txt"
    sidecar_path = tmp_path / "doc.meta.yaml"
    sidecar_path.write_text("category: [unclosed list", encoding="utf-8")
    assert load_sidecar(doc_path) == {}

def test_unknown_folder_returns_empty_category(categories_cfg):
    assert infer_category(Path("data/raw/unknown/x.txt"), categories_cfg) == ""

def test_resolve_doc_meta(tmp_path: Path, categories_cfg):
    d = tmp_path / "acts"
    d.mkdir()
    doc_path = d / "doc.txt"
    doc_path.write_text("hello")
    sidecar_path = d / "doc.meta.yaml"
    sidecar_path.write_text("short_title: 'My Act'", encoding="utf-8")
    
    meta = resolve_doc_meta(doc_path, categories_cfg)
    assert meta.category == "acts"
    assert meta.jurisdiction == "India"
    assert meta.short_title == "My Act"
    assert meta.sidecar_path == str(sidecar_path)

def test_resolve_doc_meta_contract_fields(tmp_path, categories_cfg):
    d = tmp_path / "contracts"
    d.mkdir()
    doc_path = d / "doc.txt"
    doc_path.write_text("hello")
    sidecar_path = d / "doc.meta.yaml"
    sidecar_path.write_text(
        "parties: ['Alice', 'Bob']\n"
        "effective_date: '2023-01-01'\n"
        "governing_law: 'New York'\n",
        encoding="utf-8"
    )
    
    meta = resolve_doc_meta(doc_path, categories_cfg)
    assert meta.category == "contracts"
    assert meta.parties == ["Alice", "Bob"]
    assert meta.effective_date == "2023-01-01"
    assert meta.governing_law == "New York"

def test_resolve_doc_meta_optional_fields_are_none(tmp_path, categories_cfg):
    d = tmp_path / "contracts"
    d.mkdir()
    doc_path = d / "empty.txt"
    doc_path.write_text("hello")
    meta = resolve_doc_meta(doc_path, categories_cfg)
    assert meta.parties is None
    assert meta.effective_date is None
    assert meta.governing_law is None
