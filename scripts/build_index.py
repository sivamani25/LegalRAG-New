#!/usr/bin/env python3
"""Build dense and sparse vector indexes from ingested chunks. (Milestone 2)"""

import argparse
import logging
import sys

import yaml

from legalrag.retrieval.builder import IndexBuilder


def _load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        sys.exit(f"Failed to read config {config_path}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS and BM25 indexes from chunks.jsonl"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild indexes even if fingerprints match",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO logging"
    )

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("build_index")

    cfg = _load_config(args.config)

    try:
        builder = IndexBuilder(cfg)
        builder.load_or_build(force_rebuild=args.force_rebuild)
        logger.info("Indexing complete.")
    except Exception as e:
        logger.error("Build failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
