#!/usr/bin/env python3
"""CLI entry point for the M1 ingestion pipeline.

Loads config.yaml, validates the essential keys, then delegates entirely to
``legalrag.ingest.pipeline.run_ingestion``.  No ingestion logic lives here.

Usage
-----
    python scripts/ingest.py [--config PATH] [--raw-dir PATH] [--processed-dir PATH] [-v]

Exit codes
----------
    0  success
    1  configuration error or pipeline failure
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Run the LegalRAG M1 ingestion pipeline: load raw documents from "
            "data/raw/, chunk them, apply SAC, extract citations, and write "
            "data/processed/chunks.jsonl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All paths are resolved relative to the current working directory.\n"
            "Place source documents under data/raw/<category>/ before running.\n\n"
            "Examples:\n"
            "  python scripts/ingest.py\n"
            "  python scripts/ingest.py --config config.yaml --verbose\n"
            "  python scripts/ingest.py --raw-dir /data/corpus --processed-dir /out\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config.yaml in cwd)",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        metavar="PATH",
        help="Override ingest.raw_dir from config",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        metavar="PATH",
        help="Override ingest.processed_dir from config",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> dict:
    """Load and minimally validate config.yaml.  Exits with code 1 on error."""
    import yaml  # already a project dependency

    if not config_path.is_file():
        sys.exit(
            f"[ingest] ERROR: config file not found: {config_path}\n"
            "       Run from the project root or pass --config PATH."
        )

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"[ingest] ERROR: could not parse {config_path}: {exc}")

    if not isinstance(config, dict):
        sys.exit(f"[ingest] ERROR: {config_path} did not produce a mapping — check YAML syntax.")

    # Validate required top-level sections
    missing = [k for k in ("ingest", "chunking") if k not in config]
    if missing:
        sys.exit(
            f"[ingest] ERROR: config is missing required section(s): {missing}\n"
            f"       Check {config_path}."
        )

    # Validate ingest sub-keys
    ingest = config.get("ingest", {})
    for key in ("raw_dir", "processed_dir", "chunks_file"):
        if key not in ingest:
            sys.exit(
                f"[ingest] ERROR: config['ingest'] is missing required key '{key}'.\n"
                f"       Check {config_path}."
            )

    return config


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an integer exit code."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("ingest")

    config_path = Path(args.config)
    log.debug("loading config from %s", config_path.resolve())
    config = _load_config(config_path)

    # Lazy import so the module is only required when the environment is ready.
    try:
        from legalrag.ingest.pipeline import run_ingestion
    except ImportError as exc:
        log.error("could not import legalrag — is the package installed? (%s)", exc)
        return 1

    raw_dir = args.raw_dir or config["ingest"]["raw_dir"]
    processed_dir = args.processed_dir or config["ingest"]["processed_dir"]
    chunks_file = config["ingest"]["chunks_file"]

    log.info("raw_dir       : %s", raw_dir)
    log.info("processed_dir : %s", processed_dir)
    log.info("chunks_file   : %s", chunks_file)

    try:
        run_ingestion(
            config,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            chunks_file=chunks_file,
        )
    except Exception as exc:
        log.error("ingestion failed: %s", exc)
        return 1

    out = Path(processed_dir) / chunks_file
    log.info("done — output written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
