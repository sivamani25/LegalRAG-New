import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterator

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Sanitize string to prevent path traversal and collisions."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())


def extract_case_id(url: str, text: str) -> str:
    """Extract ID from IndianKanoon URL, fallback to MD5 of text."""
    if url:
        # e.g., https://indiankanoon.org/doc/12345/
        parts = [p for p in url.strip("/").split("/") if p]
        if parts:
            # Usually the last part is the doc ID
            return safe_filename(parts[-1])

    # Fallback if URL is missing or unparseable
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def process_row(row: dict) -> tuple[str, str, dict]:
    """Extract case_id, text, and meta from a raw dataset row."""
    text = row.get("Text", "") or ""
    text = str(text).strip() if text else ""
    if not text:
        return "", "", {}

    url = row.get("Doc_url", "") or ""
    case_id = extract_case_id(str(url).strip(), text)

    meta = {
        "category": "judgments",
        "jurisdiction": "India",
    }

    title = row.get("Titles")
    if title is not None and str(title).strip():
        meta["short_title"] = str(title).strip()

    court = row.get("Court_Name_Normalized") or row.get("Court_Name")
    if court is not None and str(court).strip():
        meta["court"] = str(court).strip()

    date = row.get("Date") or row.get("date")
    if date is not None and str(date).strip():
        meta["date"] = str(date).strip()

    return case_id, text, meta


# ---------------------------------------------------------------------------
# Local Parquet mode
# ---------------------------------------------------------------------------

def _iter_parquet_dir(input_dir: Path) -> Iterator[dict]:
    """
    Yield rows (as plain dicts) from all *.parquet files in *input_dir*,
    one row at a time, without loading the full dataset into memory.

    Uses pyarrow's ParquetFile.iter_batches() for memory-efficient streaming.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for --input-dir mode. "
            "Run: pip install pyarrow"
        )

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No *.parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")

    for pq_path in parquet_files:
        logger.info(f"Reading {pq_path.name} ...")
        pf = pq.ParquetFile(pq_path)
        for batch in pf.iter_batches(batch_size=256):
            # Convert RecordBatch -> list of row dicts without loading all at once
            batch_dict = batch.to_pydict()
            col_names = list(batch_dict.keys())
            num_rows = batch.num_rows
            for i in range(num_rows):
                yield {col: batch_dict[col][i] for col in col_names}


# ---------------------------------------------------------------------------
# Hugging Face mode
# ---------------------------------------------------------------------------

def _iter_hf_dataset(split: str, streaming: bool) -> Iterator[dict]:
    """Load and iterate the InJudgements dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HF mode. "
            "Run: pip install datasets"
        )

    logger.info(f"Loading opennyaiorg/InJudgements_dataset (split={split})...")
    ds = load_dataset(
        "opennyaiorg/InJudgements_dataset", split=split, streaming=streaming
    )
    yield from ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare InJudgements dataset for LegalRAG ingestion."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/judgments",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Directory containing one or more *.parquet files to load locally. "
            "When supplied, Hugging Face is NOT accessed."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (HF mode only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mapped metadata for the first record and exit",
    )
    args = parser.parse_args()

    # Choose source
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            logger.error(f"--input-dir '{input_dir}' does not exist or is not a directory.")
            return
        row_iter = _iter_parquet_dir(input_dir)
        logger.info("Mode: local Parquet")
    else:
        streaming = bool(args.dry_run or (args.limit and args.limit < 1000))
        row_iter = _iter_hf_dataset(args.split, streaming)
        logger.info("Mode: Hugging Face")

    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")

    processed = 0
    skipped = 0
    total_seen = 0

    for row in row_iter:
        if args.limit and total_seen >= args.limit:
            break
        total_seen += 1

        case_id, text, meta = process_row(row)
        if not case_id:
            skipped += 1
            continue

        if args.dry_run:
            logger.info("--- DRY RUN ---")
            logger.info(f"Raw Row Keys: {list(row.keys())}")
            logger.info(f"Case ID: {case_id}")
            logger.info(f"Metadata:\n{yaml.dump(meta, sort_keys=False)}")
            logger.info(f"Text Preview: {text[:100]}...")
            break

        txt_path = out_dir / f"{case_id}.txt"
        meta_path = out_dir / f"{case_id}.meta.yaml"

        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta, f, sort_keys=False, allow_unicode=True)

        processed += 1
        if processed % 100 == 0:
            logger.info(f"Processed {processed} records...")

    if not args.dry_run:
        logger.info(
            f"Done. Processed: {processed}, Skipped: {skipped}, "
            f"Total written: {processed}"
        )


if __name__ == "__main__":
    main()
