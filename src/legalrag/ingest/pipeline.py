"""End-to-end ingestion pipeline.

Orchestrates all M1 ingestion stages for each configured corpus category:
  1. Load documents from ``data/raw/{category}/``
  2. Resolve category + jurisdiction metadata (sidecar override supported)
  3. Chunk with the category-appropriate strategy and preset
  4. Optionally apply Summary-Augmented Chunking (SAC)
  5. Optionally extract citations (Judgments + Contracts by default)
  6. Write all chunks to ``data/processed/chunks.jsonl``

LLM-injectable SAC
------------------
If ``chunking.sac_use_llm: true`` is set in config but no ``llm_func`` is
supplied, the pipeline logs a warning and falls back to the heuristic
summarizer.  In M5, the caller will inject an OpenRouter-backed callable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from .chunker import Chunk, Chunker
from .citation import extract_citations, extract_internal_refs
from .loader import load_documents
from .metadata import resolve_doc_meta
from .sac import apply_sac_to_document, summarize_document

logger = logging.getLogger(__name__)


def _get_category_cfg(categories_cfg: list[dict], category: str) -> dict:
    for c in categories_cfg:
        if c["name"] == category:
            return c
    return {}


def _citation_enabled(category_cfg: dict) -> bool:
    return bool(category_cfg.get("citation_extraction", False))


def run_ingestion(
    config: dict,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    chunks_file: str | None = None,
    llm_func: Callable[[str], str] | None = None,
    return_chunks: bool = False,
) -> list[Chunk] | None:
    """Run the full M1 ingestion pipeline.

    Args:
        config:        Full application config dict (from ``config.yaml``).
        raw_dir:       Override for ``ingest.raw_dir``.
        processed_dir: Override for ``ingest.processed_dir``.
        chunks_file:   Override for ``ingest.chunks_file``.
        llm_func:      Optional callable ``(doc_text) -> summary`` for LLM SAC.
                       Only called when ``chunking.sac_use_llm: true``.
                       M1 never provides a real implementation; M5 will inject
                       an OpenRouter callable.
        return_chunks: If ``True``, return the ``list[Chunk]`` in addition
                       to writing the JSONL file.

    Returns:
        ``list[Chunk]`` when *return_chunks* is ``True``, otherwise ``None``.
        Always writes the JSONL output file (even when empty).
    """
    ingest_cfg = config.get("ingest", {})
    chunking_cfg = config.get("chunking", {})

    raw_dir = Path(raw_dir or ingest_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(
        processed_dir or ingest_cfg.get("processed_dir", "data/processed")
    )
    chunks_filename = chunks_file or ingest_cfg.get("chunks_file", "chunks.jsonl")

    categories_cfg: list[dict] = ingest_cfg.get("categories", [])
    global_strategy: str = chunking_cfg.get("strategy", "pattern")
    presets: dict = chunking_cfg.get("presets", {})

    use_sac = bool(chunking_cfg.get("use_sac", True))
    sac_use_llm = bool(chunking_cfg.get("sac_use_llm", False))
    sac_max_chars = int(chunking_cfg.get("sac_summary_max_chars", 150))
    supported_exts = {
        e.lower()
        for e in ingest_cfg.get("supported_extensions", [".pdf", ".txt", ".md"])
    }

    all_chunks: list[Chunk] = []

    # Determine which directories to process
    if categories_cfg:
        dirs_to_process: list[tuple[Path, str]] = [
            (raw_dir / c["path"], c["name"]) for c in categories_cfg
        ]
    else:
        # Flat fallback: treat raw_dir as a single unnamed category
        dirs_to_process = [(raw_dir, "")]

    for doc_dir, category_hint in dirs_to_process:
        if not doc_dir.is_dir():
            logger.debug("directory %s does not exist — skipping", doc_dir)
            continue

        docs = load_documents(doc_dir, extensions=supported_exts)
        if not docs:
            logger.debug("no documents found in %s", doc_dir)

        for doc in docs:
            doc_path = Path(doc.source_path)
            meta = resolve_doc_meta(doc_path, categories_cfg)
            category = meta.category or category_hint
            jurisdiction = meta.jurisdiction

            cat_cfg = _get_category_cfg(categories_cfg, category)

            # Resolve chunking strategy (sidecar > category config > global default)
            strategy = cat_cfg.get("chunking_strategy") or global_strategy
            preset_name: str | None = cat_cfg.get("chunking_preset") or None

            chunker = Chunker(
                strategy=strategy,
                config=chunking_cfg,
                preset_name=preset_name,
                presets=presets,
            )

            chunks = chunker.chunk(
                source_doc=doc.source_doc,
                text=doc.text,
                category=category,
                jurisdiction=jurisdiction,
                source_path=doc.source_path,
            )

            # SAC
            if use_sac and chunks:
                _apply_sac(
                    chunks,
                    doc.text,
                    sac_max_chars=sac_max_chars,
                    sac_use_llm=sac_use_llm,
                    llm_func=llm_func,
                    category=category,
                    source_doc=doc.source_doc,
                )

            # Citation extraction
            if _citation_enabled(cat_cfg):
                enabled_cats = frozenset({category}) if category else frozenset()
                for chunk in chunks:
                    extracted = extract_citations(
                        chunk.original_text,
                        category=category,
                        enabled_categories=enabled_cats,
                    )
                    chunk.cites = [c.to_dict() for c in extracted]

            # Contract metadata and internal references
            for chunk in chunks:
                if category == "contracts":
                    c_meta = {}
                    if meta.parties:
                        c_meta["parties"] = meta.parties
                    if meta.effective_date:
                        c_meta["effective_date"] = meta.effective_date
                    if meta.governing_law:
                        c_meta["governing_law"] = meta.governing_law
                    
                    internal_refs = extract_internal_refs(chunk.original_text)
                    if internal_refs:
                        c_meta["internal_refs"] = internal_refs
                    
                    chunk.contract_meta = c_meta

            all_chunks.extend(chunks)

    # Write JSONL output
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / chunks_filename
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            fh.write("\n")

    logger.info("wrote %d chunks to %s", len(all_chunks), out_path)

    if return_chunks:
        return all_chunks
    return None


def _apply_sac(
    chunks: list[Chunk],
    doc_text: str,
    *,
    sac_max_chars: int,
    sac_use_llm: bool,
    llm_func: Callable[[str], str] | None,
    category: str,
    source_doc: str,
) -> None:
    """Apply SAC to *chunks*, handling the LLM→heuristic fallback."""
    if sac_use_llm and llm_func is None:
        logger.warning(
            "sac_use_llm=True but no llm_func provided for %r; "
            "falling back to heuristic summarizer",
            source_doc,
        )
        sac_use_llm = False

    result = summarize_document(
        doc_text,
        max_chars=sac_max_chars,
        use_llm=sac_use_llm,
        llm_func=llm_func,
        category=category,
    )
    apply_sac_to_document(chunks, result.summary)
