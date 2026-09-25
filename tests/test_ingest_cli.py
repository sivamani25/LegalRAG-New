"""Tests for scripts/ingest.py CLI entrypoint.

Covers: argument parsing, config validation, successful run (empty corpus),
clear failure modes (missing config, bad YAML, missing keys, bad import path).
Does NOT test ingestion logic — that belongs to test_pipeline.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers: load the script as a module without installing it
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "scripts" / "ingest.py"


def _load_ingest_module():
    """Dynamically load scripts/ingest.py so we can call its functions."""
    spec = importlib.util.spec_from_file_location("scripts.ingest", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ingest_mod():
    return _load_ingest_module()


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults(ingest_mod):
    ns = ingest_mod._parse_args([])
    assert ns.config == "config.yaml"
    assert ns.raw_dir is None
    assert ns.processed_dir is None
    assert ns.verbose is False


def test_parse_args_overrides(ingest_mod):
    ns = ingest_mod._parse_args([
        "--config", "alt.yaml",
        "--raw-dir", "/raw",
        "--processed-dir", "/proc",
        "--verbose",
    ])
    assert ns.config == "alt.yaml"
    assert ns.raw_dir == "/raw"
    assert ns.processed_dir == "/proc"
    assert ns.verbose is True


def test_parse_args_short_verbose(ingest_mod):
    ns = ingest_mod._parse_args(["-v"])
    assert ns.verbose is True


# ---------------------------------------------------------------------------
# _load_config — error paths
# ---------------------------------------------------------------------------

def test_load_config_missing_file_exits(ingest_mod, tmp_path):
    with pytest.raises(SystemExit) as exc:
        ingest_mod._load_config(tmp_path / "nonexistent.yaml")
    assert exc.value.code != 0


def test_load_config_bad_yaml_exits(ingest_mod, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(": : invalid: [", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ingest_mod._load_config(bad)
    assert exc.value.code != 0


def test_load_config_non_mapping_yaml_exits(ingest_mod, tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ingest_mod._load_config(bad)
    assert exc.value.code != 0


def test_load_config_missing_ingest_section_exits(ingest_mod, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("chunking: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ingest_mod._load_config(cfg)
    assert exc.value.code != 0


def test_load_config_missing_ingest_key_exits(ingest_mod, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    # ingest section exists but raw_dir is absent
    cfg.write_text(
        "ingest:\n  processed_dir: x\n  chunks_file: y\nchunking: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        ingest_mod._load_config(cfg)
    assert exc.value.code != 0


def test_load_config_valid(ingest_mod, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump({
            "ingest": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "chunks_file": "chunks.jsonl",
                "categories": [],
                "supported_extensions": [".txt"],
            },
            "chunking": {"strategy": "sentence", "use_sac": False},
        }),
        encoding="utf-8",
    )
    result = ingest_mod._load_config(cfg)
    assert result["ingest"]["raw_dir"] == "data/raw"


# ---------------------------------------------------------------------------
# main() — successful run with empty corpus
# ---------------------------------------------------------------------------

def test_main_empty_corpus(ingest_mod, tmp_path):
    """main() should succeed (exit 0) when raw_dir has no documents."""
    cfg = tmp_path / "config.yaml"
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "proc"

    cfg.write_text(
        yaml.dump({
            "ingest": {
                "raw_dir": str(raw),
                "processed_dir": str(proc),
                "chunks_file": "chunks.jsonl",
                "categories": [],
                "supported_extensions": [".txt"],
            },
            "chunking": {
                "strategy": "sentence",
                "use_sac": False,
                "sentence": {"max_chars": 1000, "min_chars": 50},
                "presets": {},
            },
        }),
        encoding="utf-8",
    )

    rc = ingest_mod.main(["--config", str(cfg)])
    assert rc == 0
    assert (proc / "chunks.jsonl").is_file()


def test_main_writes_chunks_jsonl_for_nonempty_dir(ingest_mod, tmp_path):
    """main() with a real document writes at least one chunk."""
    import json

    cfg_path = tmp_path / "config.yaml"
    raw = tmp_path / "raw"
    acts = raw / "acts"
    acts.mkdir(parents=True)
    (acts / "sample.txt").write_text("Section 1\nSome statutory text.", encoding="utf-8")
    proc = tmp_path / "proc"

    cfg_path.write_text(
        yaml.dump({
            "ingest": {
                "raw_dir": str(raw),
                "processed_dir": str(proc),
                "chunks_file": "chunks.jsonl",
                "categories": [
                    {
                        "name": "acts",
                        "path": "acts",
                        "default_jurisdiction": "India",
                        "chunking_strategy": "sentence",
                        "chunking_preset": None,
                        "citation_extraction": False,
                    }
                ],
                "supported_extensions": [".txt"],
            },
            "chunking": {
                "strategy": "sentence",
                "use_sac": False,
                "sentence": {"max_chars": 1000, "min_chars": 10},
                "presets": {},
            },
        }),
        encoding="utf-8",
    )

    rc = ingest_mod.main(["--config", str(cfg_path)])
    assert rc == 0
    lines = (proc / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert rec["category"] == "acts"


def test_main_bad_config_path_exits_nonzero(ingest_mod, tmp_path):
    # _load_config calls sys.exit() with a message string, which raises SystemExit.
    with pytest.raises(SystemExit) as exc:
        ingest_mod.main(["--config", str(tmp_path / "missing.yaml")])
    assert exc.value.code != 0


def test_main_respects_raw_dir_override(ingest_mod, tmp_path):
    """--raw-dir override takes priority over config value."""
    cfg_path = tmp_path / "config.yaml"
    override_raw = tmp_path / "override_raw"
    override_raw.mkdir()
    proc = tmp_path / "proc"

    cfg_path.write_text(
        yaml.dump({
            "ingest": {
                "raw_dir": str(tmp_path / "nonexistent_raw"),
                "processed_dir": str(proc),
                "chunks_file": "chunks.jsonl",
                "categories": [],
                "supported_extensions": [".txt"],
            },
            "chunking": {
                "strategy": "sentence",
                "use_sac": False,
                "sentence": {"max_chars": 1000, "min_chars": 50},
                "presets": {},
            },
        }),
        encoding="utf-8",
    )

    rc = ingest_mod.main(["--config", str(cfg_path), "--raw-dir", str(override_raw)])
    assert rc == 0
