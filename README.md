# LegalRAG

A working reference implementation of a Retrieval-Augmented Generation (RAG)
system for legal documents, built to the architecture described in:

> Hindi *et al.*, "Enhancing the Precision and Interpretability of RAG in
> Legal Technology: A Survey," IEEE Access, 2025

...plus fixes for known failure modes identified in follow-up 2025/2026
research. The authoritative engineering spec is `AGENTS.md` in the reference
project.

---

## Architecture overview

```
data/raw/ ──► loader ──► chunker (sentence | pattern, structure-aware)
                    │
                    └─► SAC (summary prepended to every chunk, toggleable)
                    │
                    ▼
        data/processed/chunks.jsonl   (hierarchy + offsets + SAC metadata)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   dense (Nomic) index      BM25 index      — namespaced by (model, SAC) under data/index/
        └───────────┬───────────┘
                    ▼
        HybridRetriever: dense ∥ BM25 → RRF fusion → MMR rerank
        → structural parent-section boost
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  premise verification    generation (OpenRouter, inline citations)
        └───────────┬───────────┘
                    ▼
      adaptive loop (support check → query rewrite → re-retrieve, max N rounds)
                    ▼
   FastAPI (/query, /eval) + single-page frontend
```

**Failure modes targeted:**

1. **Document-Level Retrieval Mismatch (DRM)** — fixed via Summary-Augmented Chunking (SAC).
2. **Retrieval ceiling** — fixed via config-driven embedding model + ablation harness.
3. **Hallucination with good retrieval** — fixed via premise-verification step + claim-level faithfulness scoring.
4. **Flattened legal structure** — fixed via structure-aware chunking retaining hierarchy metadata.

---

## Project structure

```
.
├── config.yaml                # single source of truth for all runtime settings
├── pyproject.toml             # deps (pinned), pytest config; src layout
├── .env.example               # environment variable documentation
├── src/legalrag/              # application package
│   ├── ingest/                # M1: loader, chunker, SAC, pipeline
│   ├── retrieval/             # M2–M3, M6: embed, index, retriever, adaptive
│   ├── generation/            # M5: prompt, generate (OpenRouter)
│   ├── eval/                  # M4, M7: metrics, DRM, ablation, claim-check
│   └── app/                   # M8: FastAPI + single-page frontend
├── tests/                     # pytest suite (one file per module)
├── scripts/                   # utility scripts (ingestion runner, index builder)
├── frontend/                  # static frontend assets (M8)
├── docs/                      # documentation
└── data/
    ├── raw/                   # drop your legal documents here (.pdf/.txt/.md)
    ├── processed/             # runtime: chunks.jsonl (gitignored)
    ├── index/                 # runtime: namespaced indexes (gitignored)
    ├── cache/                 # runtime: cache (gitignored)
    └── eval/                  # evaluation corpus and query sets (M4)
```

---

## Milestones

| # | What | Status |
|---|---|---|
| M0 | Project skeleton, config, directory structure | ✅ |
| M1 | Ingestion: loader, sentence + pattern chunker, SAC, pipeline | — |
| M2 | Embedding (Nomic) + FAISS dense index + BM25 index | — |
| M3 | Hybrid retrieval: RRF + MMR rerank + structural boost | — |
| M4 | DRM metric + SAC on/off before/after report | — |
| M5 | Prompt template + generation via OpenRouter + premise verification | — |
| M6 | Adaptive retrieval loop | — |
| M7 | Full eval harness: precision/recall/MRR/MAP, DRM, RAGAs-style, claim-level, failure tagger, embedding ablation | — |
| M8 | FastAPI backend + single-page frontend | — |
| M9 | README, DRM before/after numbers, end-to-end verification | — |

---

## Setup

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in keys if you intend to run real
generation. No API key is needed to run tests or to use offline modes.

```bash
# Generation only — key read at call time, never stored in the repo
export OPENROUTER_API_KEY=sk-or-...

# Optional: Ollama local LLM (set generation.provider: ollama in config.yaml)
export OLLAMA_BASE_URL=http://localhost:11434
```

### Run tests

```bash
pytest -q
```

### Configuration

All runtime settings live in [`config.yaml`](config.yaml). Key values:

- `chunking.strategy` — `pattern` (default, structure-aware) or `sentence`
- `chunking.use_sac` — toggle Summary-Augmented Chunking on/off
- `retrieval.embedding_model` — model name passed to the embedder (never hardcoded)
- `generation.provider` — `openrouter` or `ollama`
- `generation.model` — OpenRouter model identifier (e.g. `anthropic/claude-sonnet-4-5`)

---

## Data

Place your legal documents (PDF, TXT, or MD) under `data/raw/`. Do **not**
use copyrighted case law or contracts you do not have rights to. For public-domain
text, consider sample statutes or the Open Australian Legal Corpus (CC-licensed).

No sample documents are shipped in this repository — the pipeline is designed
for the real Tier-1 corpus you supply.

---

## License and data note

This is a reference implementation. **Do not** scrape or reproduce copyrighted
case law or contracts. For a demo corpus, use public-domain or openly licensed
legal text, or documents you supply yourself under `data/raw/`.
