"""Validate the M6 retrieval evaluation cases (schema, categories, chunk IDs).

Lightweight: uses the small acts/constitution corpora directly and the built
judgments index ID list for existence checks. Does not load the Nomic model or
the 1.31 GB judgment corpus.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval" / "retrieval_cases.jsonl"
CORPORA = {
    "acts": ROOT / "data" / "pilot" / "processed" / "acts" / "chunks.jsonl",
    "constitution": ROOT / "data" / "processed" / "chunks.jsonl",
}
JUDGMENT_IDS_PATH = (
    ROOT / "data" / "index" / "judgments"
    / "judgments_nomic-embed-text_sac_dense_ids.json"
)
VALID_CATEGORIES = {"acts", "constitution", "judgments"}
REQUIRED_KEYS = {"query", "relevant_ids", "category", "expected_evidence", "source_doc"}


def _load_cases():
    assert CASES_PATH.is_file(), f"missing cases file: {CASES_PATH}"
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunk_ids(path):
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["chunk_id"])
    return ids


def test_cases_schema_and_category_balance():
    cases = _load_cases()
    assert len(cases) == 30, f"expected 30 cases, found {len(cases)}"

    counts = Counter(c["category"] for c in cases)
    assert counts == {"acts": 10, "constitution": 10, "judgments": 10}, counts

    for i, c in enumerate(cases):
        assert REQUIRED_KEYS <= set(c), f"case {i} missing keys: {c.keys()}"
        assert isinstance(c["query"], str) and c["query"].strip(), f"case {i} query"
        assert isinstance(c["relevant_ids"], list) and c["relevant_ids"], f"case {i} ids"
        assert all(isinstance(r, str) and r for r in c["relevant_ids"]), f"case {i} ids"
        assert c["category"] in VALID_CATEGORIES, f"case {i} category {c['category']!r}"
        assert isinstance(c["expected_evidence"], str) and c["expected_evidence"].strip()
        assert isinstance(c["source_doc"], str) and c["source_doc"].strip()


def test_acts_and_constitution_ids_exist():
    cases = _load_cases()
    for category, path in CORPORA.items():
        ids = _chunk_ids(path)
        for c in cases:
            if c["category"] != category:
                continue
            for rid in c["relevant_ids"]:
                assert rid in ids, f"{rid} not found in {category} corpus"


def test_judgment_ids_exist():
    if not JUDGMENT_IDS_PATH.is_file():
        pytest.skip("judgments index id list not available")
    known = set(json.loads(JUDGMENT_IDS_PATH.read_text(encoding="utf-8")))
    for c in _load_cases():
        if c["category"] != "judgments":
            continue
        for rid in c["relevant_ids"]:
            assert rid in known, f"{rid} not found in judgments index id list"


def test_no_duplicate_queries():
    queries = [c["query"] for c in _load_cases()]
    assert len(queries) == len(set(queries)), "duplicate queries in the evaluation set"
