import json
import pytest
from pathlib import Path
from legalrag.ingest.pipeline import run_ingestion
import yaml

@pytest.fixture
def config():
    return {
        "ingest": {
            "categories": [
                {
                    "name": "acts",
                    "path": "acts",
                    "default_jurisdiction": "India",
                    "chunking_strategy": "pattern",
                    "chunking_preset": "generic",
                    "citation_extraction": False
                },
                {
                    "name": "judgments",
                    "path": "judgments",
                    "default_jurisdiction": "India",
                    "chunking_strategy": "sentence",
                    "citation_extraction": True
                }
            ],
            "supported_extensions": [".txt"]
        },
        "chunking": {
            "strategy": "sentence",
            "use_sac": True,
            "sac_use_llm": False,
            "presets": {
                "generic": {
                    "delimiters": [{"pattern": r"^[ \t]*Section\s+\d+", "level": 1}]
                }
            },
            "sentence": {"max_chars": 100, "min_chars": 10}
        }
    }

def test_run_ingestion_writes_chunks_jsonl(tmp_path, config):
    raw = tmp_path / "raw"
    acts = raw / "acts"
    acts.mkdir(parents=True)
    (acts / "doc.txt").write_text("Section 1\nHello")
    
    proc = tmp_path / "proc"
    run_ingestion(config, raw_dir=raw, processed_dir=proc)
    
    out_file = proc / "chunks.jsonl"
    assert out_file.is_file()
    lines = out_file.read_text().splitlines()
    assert len(lines) > 0
    data = json.loads(lines[0])
    assert data["category"] == "acts"

def test_chunk_record_has_all_m1_fields(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Section 1\nHello")
    
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert len(chunks) > 0
    d = chunks[0].to_dict()
    for field in [
        "chunk_id", "source_doc", "source_path", "category", "jurisdiction",
        "text", "original_text", "doc_summary", "section", "parent", "heading",
        "hierarchy_path", "cites", "start_offset", "end_offset", "sac_applied"
    ]:
        assert field in d

def test_chunk_id_format(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Hello")
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert chunks[0].chunk_id == "acts/doc/0"

def test_sac_on_sets_flag_and_summary(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Hello")
    config["chunking"]["use_sac"] = True
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert chunks[0].sac_applied is True
    assert chunks[0].doc_summary != ""

def test_sac_off_clears_flag_and_summary(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Hello")
    config["chunking"]["use_sac"] = False
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert chunks[0].sac_applied is False
    assert chunks[0].doc_summary == ""

def test_citation_enabled_category_has_cites(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "judgments").mkdir(parents=True)
    (raw / "judgments" / "doc.txt").write_text("See AIR 2020 SC 100.")
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert len(chunks[0].cites) == 1
    assert chunks[0].cites[0]["category"] == "case_law"

def test_citation_disabled_category_empty_cites(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("See AIR 2020 SC 100.")
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert len(chunks[0].cites) == 0

def test_sidecar_jurisdiction_propagates_to_chunk(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Hello")
    (raw / "acts" / "doc.meta.yaml").write_text("jurisdiction: UK")
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert chunks[0].jurisdiction == "UK"

def test_empty_category_dir_produces_no_chunks(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    assert len(chunks) == 0

def test_return_chunks_flag(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "acts").mkdir(parents=True)
    (raw / "acts" / "doc.txt").write_text("Hello")
    ret = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=False)
    assert ret is None

def test_pipeline_propagates_contract_meta(tmp_path, config):
    raw = tmp_path / "raw"
    (raw / "contracts").mkdir(parents=True)
    (raw / "contracts" / "doc.txt").write_text("See Exhibit A.")
    (raw / "contracts" / "doc.meta.yaml").write_text(
        "parties: ['Alice', 'Bob']\n"
        "effective_date: '2023-01-01'\n"
        "governing_law: 'NY'\n"
    )
    
    # We need to make sure the config has 'contracts' category
    if not any(c["name"] == "contracts" for c in config["ingest"]["categories"]):
        config["ingest"]["categories"].append({
            "name": "contracts",
            "path": "contracts",
            "default_jurisdiction": "",
            "chunking_strategy": "sentence",
        })
        
    chunks = run_ingestion(config, raw_dir=raw, processed_dir=tmp_path, return_chunks=True)
    
    # Find the contract chunk
    chunk = chunks[0]
    assert chunk.category == "contracts"
    assert chunk.contract_meta["parties"] == ["Alice", "Bob"]
    assert chunk.contract_meta["effective_date"] == "2023-01-01"
    assert chunk.contract_meta["governing_law"] == "NY"
    assert chunk.contract_meta["internal_refs"] == ["Exhibit A"]
