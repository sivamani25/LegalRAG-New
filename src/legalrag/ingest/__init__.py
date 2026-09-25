"""Ingestion sub-package: loader, metadata, chunker, SAC, citation, pipeline. (Milestone 1)

Modules
-------
loader    — Document dataclass; load_document / load_documents (PDF/TXT/MD)
metadata  — DocMeta; category / jurisdiction inference; sidecar parser
chunker   — Chunk dataclass; SentenceChunker; PatternChunker (preset-driven); Chunker facade
sac       — SummaryResult; summarize_document; apply_sac_to_document
citation  — Citation dataclass; extract_citations (regex, offline)
pipeline  — run_ingestion(config) → data/processed/chunks.jsonl
"""
