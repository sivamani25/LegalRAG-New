"""Tests for scripts/prepare_injudgements.py."""
import sys
import os
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath("scripts"))
import prepare_injudgements


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_parquet_dir(tmp_path):
    """
    Write a tiny *.parquet fixture (3 rows) into a temp directory.
    Uses pyarrow directly – no HF datasets library, no production files.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "Text": [
            "This is judgment number one.",
            "This is judgment number two.",
            "",  # empty text → should be skipped by process_row
        ],
        "Doc_url": [
            "https://indiankanoon.org/doc/111/",
            "https://indiankanoon.org/doc/222/",
            "https://indiankanoon.org/doc/333/",
        ],
        "Titles": [
            "Case Alpha v. Beta",
            "Case Gamma v. Delta",
            None,
        ],
        "Court_Name_Normalized": [
            "Supreme Court of India",
            None,
            None,
        ],
        "Court_Name": [
            None,
            "Delhi High Court",
            None,
        ],
        "Date": [
            "2023-01-15",
            None,
            None,
        ],
        "date": [
            None,
            "2022-06-01",
            None,
        ],
    }

    table = pa.table(data)
    pq_path = tmp_path / "test_shard.parquet"
    pq.write_table(table, pq_path)
    return tmp_path


@pytest.fixture()
def multi_parquet_dir(tmp_path):
    """Two separate parquet shards to verify glob discovery."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _make_shard(name: str, doc_id: str, text: str):
        table = pa.table({
            "Text": [text],
            "Doc_url": [f"https://indiankanoon.org/doc/{doc_id}/"],
            "Titles": ["Shard Title"],
            "Court_Name_Normalized": [None],
            "Court_Name": ["Some Court"],
            "Date": [None],
            "date": [None],
        })
        pq.write_table(table, tmp_path / name)

    _make_shard("shard_a.parquet", "AAA", "Text from shard A.")
    _make_shard("shard_b.parquet", "BBB", "Text from shard B.")
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests for pure-function helpers (unchanged from original)
# ---------------------------------------------------------------------------

def test_extract_case_id():
    # URL extraction
    assert prepare_injudgements.extract_case_id("https://indiankanoon.org/doc/12345/", "text") == "12345"
    assert prepare_injudgements.extract_case_id("https://indiankanoon.org/doc/67890", "text") == "67890"
    # Fallback hashing
    h = prepare_injudgements.extract_case_id("", "Sample judgment text")
    assert len(h) == 32
    assert h != prepare_injudgements.extract_case_id("", "Different text")


def test_safe_filename():
    assert prepare_injudgements.safe_filename("doc/123.pdf") == "doc_123_pdf"
    assert prepare_injudgements.safe_filename("  test@name!  ") == "test_name_"


def test_process_row():
    row = {
        "Doc_url": "http://indiankanoon.org/doc/999/",
        "Text": "This is a judgment.",
        "Titles": "Kesavananda Bharati v. State of Kerala",
        "Court_Name_Normalized": "Supreme Court of India",
    }

    case_id, text, meta = prepare_injudgements.process_row(row)

    assert case_id == "999"
    assert text == "This is a judgment."
    assert meta["category"] == "judgments"
    assert meta["jurisdiction"] == "India"
    assert meta["short_title"] == "Kesavananda Bharati v. State of Kerala"
    assert meta["court"] == "Supreme Court of India"
    assert "date" not in meta


def test_process_row_missing_fields():
    row = {"Text": "Judgment with missing metadata."}
    case_id, text, meta = prepare_injudgements.process_row(row)

    assert len(case_id) == 32  # hash
    assert meta["category"] == "judgments"
    assert "court" not in meta
    assert "short_title" not in meta


def test_process_row_empty_text():
    row = {"Titles": "Empty Judgment", "Text": ""}
    case_id, text, meta = prepare_injudgements.process_row(row)
    assert not case_id
    assert not text
    assert not meta


def test_process_row_none_text():
    """pyarrow may decode missing strings as None – must be handled safely."""
    row = {"Text": None, "Doc_url": None, "Titles": None}
    case_id, text, meta = prepare_injudgements.process_row(row)
    assert not case_id
    assert not text
    assert not meta


# ---------------------------------------------------------------------------
# Tests for _iter_parquet_dir
# ---------------------------------------------------------------------------

def test_iter_parquet_dir_yields_rows(tiny_parquet_dir):
    rows = list(prepare_injudgements._iter_parquet_dir(tiny_parquet_dir))
    assert len(rows) == 3, "Should yield all 3 rows from the single parquet file"


def test_iter_parquet_dir_row_schema(tiny_parquet_dir):
    rows = list(prepare_injudgements._iter_parquet_dir(tiny_parquet_dir))
    first = rows[0]
    assert "Text" in first
    assert "Doc_url" in first
    assert "Titles" in first


def test_iter_parquet_dir_multi_shards(multi_parquet_dir):
    rows = list(prepare_injudgements._iter_parquet_dir(multi_parquet_dir))
    assert len(rows) == 2, "Should yield one row per shard = 2 total"
    doc_ids = {r["Doc_url"] for r in rows}
    assert "https://indiankanoon.org/doc/AAA/" in doc_ids
    assert "https://indiankanoon.org/doc/BBB/" in doc_ids


def test_iter_parquet_dir_empty_raises(tmp_path):
    """Should raise FileNotFoundError when no *.parquet files exist."""
    with pytest.raises(FileNotFoundError, match=r"No \*.parquet files found"):
        list(prepare_injudgements._iter_parquet_dir(tmp_path))


# ---------------------------------------------------------------------------
# Integration: local Parquet mode end-to-end (no HF access)
# ---------------------------------------------------------------------------

def test_local_parquet_mode_writes_output(tiny_parquet_dir, tmp_path):
    """
    End-to-end: feed a tiny parquet dir through the main() pipeline and verify
    that .txt and .meta.yaml files are written correctly.
    """
    out_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    txt_files = list(out_dir.glob("*.txt"))
    meta_files = list(out_dir.glob("*.meta.yaml"))

    # Row 3 has empty text → skipped; rows 1 & 2 should be written
    assert len(txt_files) == 2, f"Expected 2 .txt files, got {len(txt_files)}"
    assert len(meta_files) == 2, f"Expected 2 .meta.yaml files, got {len(meta_files)}"


def test_local_parquet_mode_correct_ids(tiny_parquet_dir, tmp_path):
    """Case IDs are derived from the Doc_url."""
    out_dir = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (out_dir / "111.txt").exists()
    assert (out_dir / "222.txt").exists()


def test_local_parquet_mode_metadata_content(tiny_parquet_dir, tmp_path):
    """Verify YAML metadata fields are correctly populated."""
    import yaml as _yaml

    out_dir = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    with open(out_dir / "111.meta.yaml", encoding="utf-8") as f:
        meta1 = _yaml.safe_load(f)

    assert meta1["category"] == "judgments"
    assert meta1["jurisdiction"] == "India"
    assert meta1["short_title"] == "Case Alpha v. Beta"
    assert meta1["court"] == "Supreme Court of India"
    assert meta1["date"] == "2023-01-15"

    with open(out_dir / "222.meta.yaml", encoding="utf-8") as f:
        meta2 = _yaml.safe_load(f)

    # Court_Name_Normalized is None for row 2 → fall back to Court_Name
    assert meta2["court"] == "Delhi High Court"
    # date falls back to lowercase 'date' field
    assert meta2["date"] == "2022-06-01"


def test_local_parquet_mode_limit(tiny_parquet_dir, tmp_path):
    """--limit should cap the number of rows processed."""
    out_dir = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
            "--limit", "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    txt_files = list(out_dir.glob("*.txt"))
    assert len(txt_files) == 1, "With --limit 1 only 1 file should be written"


def test_local_parquet_mode_no_overwrite(tiny_parquet_dir, tmp_path):
    """Without --overwrite, existing files should not be re-written."""
    out_dir = tmp_path / "output"

    # First run
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
        ],
        check=True, capture_output=True,
    )

    # Record mtime
    existing = list(out_dir.glob("*.txt"))[0]
    mtime_before = existing.stat().st_mtime

    # Second run – should skip existing
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
        ],
        check=True, capture_output=True,
    )

    mtime_after = existing.stat().st_mtime
    assert mtime_before == mtime_after, "File should NOT have been overwritten"


def test_local_parquet_mode_dry_run(tiny_parquet_dir, tmp_path):
    """--dry-run should print metadata but write NO files."""
    out_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(tiny_parquet_dir),
            "--output-dir", str(out_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "DRY RUN" in result.stderr
    assert not out_dir.exists() or not list(out_dir.glob("*.txt")), \
        "--dry-run must not write any files"


def test_local_parquet_mode_missing_dir(tmp_path):
    """Passing a non-existent --input-dir should exit without crash."""
    missing = tmp_path / "does_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_injudgements.py",
            "--input-dir", str(missing),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    # Script logs an error and returns 0 (graceful exit)
    assert "does not exist" in result.stderr
