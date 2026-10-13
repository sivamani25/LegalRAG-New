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
      FastAPI (/chat, /chat/pdf, /eval) + single-page frontend
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
| M1 | Ingestion: loader, sentence + pattern chunker, SAC, pipeline | ✅ |
| M2 | Embedding (Nomic) + FAISS dense index + BM25 index | ✅ |
| M3 | Hybrid retrieval: RRF + MMR rerank + structural boost | ✅ |
| M4 | DRM metric + SAC on/off before/after report | 🔶 metrics + evaluation harness only |
| M5 | Prompt template + generation via OpenRouter + premise verification | ✅ (citation-label verification only) |
| M6 | Adaptive retrieval loop | 🔶 classifier + legal-issue rewriter + routing |
| M7 | Full eval harness (precision/recall/MRR/MAP; DRM, RAGAs-style, claim-level, ablation) | 🔶 precision/recall/MRR/MAP implemented |
| M8 | FastAPI backend (`/chat`, `/chat/pdf`, `/eval`) + single-page frontend | ✅ |
| M9 | README, DRM before/after numbers, end-to-end verification | 🔶 in progress |

---

## Setup

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set the OpenRouter key. Generation uses
**OpenRouter only**; no API key is needed to run tests or offline modes.

Linux / macOS (bash):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENROUTER_API_KEY=sk-or-...
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# Put OPENROUTER_API_KEY=sk-or-... in .env (gitignored), or set it for the session:
$env:OPENROUTER_API_KEY = "sk-or-..."
```
Never commit `.env`. The key is read at call time and is never returned by the API
or embedded in the frontend/PDF.

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

## API, frontend and PDF export

Start the API (Windows PowerShell):
```powershell
.\.venv\Scripts\python.exe -m legalrag.app.main
# or: .\.venv\Scripts\python.exe -m uvicorn legalrag.app.main:app --host 127.0.0.1 --port 8000
```
Then open <http://127.0.0.1:8000/> for the single-page frontend.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + configured categories |
| POST | `/chat` | grounded answer, citations/metadata, related cases, limitations |
| POST | `/chat/pdf` | the same answer as a downloadable PDF |
| GET | `/eval` | live retrieval metrics (precision/recall/MRR/MAP) via the eval harness |

Demo workflow:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$q = '{"question":"What does Article 21 of the Constitution guarantee about life and personal liberty?"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType 'application/json' -Body $q
Invoke-WebRequest -Method Post http://127.0.0.1:8000/chat/pdf -ContentType 'application/json' -Body $q -OutFile answer.pdf
Invoke-RestMethod 'http://127.0.0.1:8000/eval?k=1,3,5,10'
```
The frontend has an **Ask** button and a **Download PDF** button (enabled after a
successful answer). Retrieval is federated across the Acts, Constitution and
Judgments indexes; related cases come from the Judgments index.

**First-request warm-up:** the retrieval bundle (three SentenceTransformer-backed
retrievers + the OpenRouter client) is built lazily on the first `/chat`,
`/chat/pdf`, or `/eval` call — expect ~1 minute on the first request (model load
plus a one-time byte-offset scan of the Judgments corpus). Later requests are fast.
`/eval` runs live over all 30 cases and takes a few minutes.

**Important:** the verifier checks only that citation *labels* (`[S1]`, `[S2]`, …)
exist and map to retrieved passages — it does **not** establish that the legal
content is substantively correct. Output is general legal information and is not a
substitute for advice from a qualified lawyer.

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
