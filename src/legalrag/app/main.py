"""FastAPI application for LegalRAG (M7).

Endpoints:
    GET  /health  -> liveness + which categories are configured
    POST /chat    -> grounded answer, citations/metadata, related cases, limitations
    GET  /eval    -> live retrieval metrics via legalrag.eval.runner (no rebuild)
    GET  /        -> frontend single page
    GET  /static/*-> frontend assets

The retrieval bundle (three SentenceTransformer-backed retrievers + OpenRouter
client) is built lazily on first use so the app imports and starts quickly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from legalrag.app import retrieval as R
from legalrag.app.pdf import build_pdf
from legalrag.app.schemas import (
    CategoryEval,
    ChatRequest,
    ChatResponse,
    EvalResponse,
    HealthResponse,
)
from legalrag.app.service import ChatService
from legalrag.app.text_store import MultiCorpusTextStore
from legalrag.eval.runner import evaluate_retriever
from legalrag.generation.client import OpenRouterClient, OpenRouterError

ROOT = Path(__file__).resolve().parents[3]


def _load_cases(cases_path: Path) -> list[dict]:
    with open(cases_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _macro_overall(categories: dict[str, CategoryEval], k_values: list[int]) -> dict[str, dict[str, float]]:
    overall: dict[str, dict[str, float]] = {}
    for k in k_values:
        key = str(k)
        with_metrics = [c for c in categories.values() if key in c.metrics_by_k]
        if not with_metrics:
            overall[key] = {}
            continue
        overall[key] = {
            metric: sum(c.metrics_by_k[key][metric] for c in with_metrics) / len(with_metrics)
            for metric in ("precision", "recall", "mrr", "map")
        }
    return overall


def create_app(
    config: dict | None = None,
    config_path: str | Path | None = None,
    retrievers: dict | None = None,
    chat_service: ChatService | None = None,
    cases_path: str | Path | None = None,
    frontend_dir: str | Path | None = None,
    root: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI app. Overrides allow tests to inject fakes."""
    root = Path(root) if root else ROOT
    cases_file = Path(cases_path) if cases_path else root / "data/eval/retrieval_cases.jsonl"
    frontend = Path(frontend_dir) if frontend_dir else root / "frontend"

    state: dict = {"config": config, "retrievers": retrievers, "chat_service": chat_service, "service": None}
    lock = threading.Lock()

    def _config() -> dict:
        if state["config"] is None:
            cfg_file = Path(config_path) if config_path else root / "config.yaml"
            state["config"] = R.load_config(cfg_file)
        return state["config"]

    def _retrievers() -> dict:
        if state["retrievers"] is None:
            state["retrievers"] = R.build_retrievers(root)
        return state["retrievers"]

    def _service() -> ChatService:
        if state["chat_service"] is not None:
            return state["chat_service"]
        if state["service"] is None:
            with lock:
                if state["service"] is None:
                    multi = R.MultiCategoryRetriever(_retrievers())
                    store = MultiCorpusTextStore(root)
                    state["service"] = ChatService(multi, store, OpenRouterClient(_config()))
        return state["service"]

    app = FastAPI(title="LegalRAG", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            categories=list(R.CATEGORIES),
            retrievers_loaded=state["retrievers"] is not None or state["chat_service"] is not None,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> dict:
        try:
            service = _service()
        except Exception:
            raise HTTPException(status_code=500, detail="Service initialisation failed.")
        history = [turn.model_dump() for turn in request.conversation_history] or None
        try:
            return service.chat(request.question, history)
        except OpenRouterError:
            raise HTTPException(
                status_code=503,
                detail="Generation is unavailable (OpenRouter). Ensure OPENROUTER_API_KEY is set on the server.",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Answer generation failed due to a retrieval or processing error.",
            )

    @app.post("/chat/pdf")
    def chat_pdf(request: ChatRequest) -> Response:
        try:
            service = _service()
        except Exception:
            raise HTTPException(status_code=500, detail="Service initialisation failed.")
        history = [turn.model_dump() for turn in request.conversation_history] or None
        try:
            data = service.chat(request.question, history)
        except OpenRouterError:
            raise HTTPException(
                status_code=503,
                detail="Generation is unavailable (OpenRouter). Ensure OPENROUTER_API_KEY is set on the server.",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Answer generation failed due to a retrieval or processing error.",
            )
        pdf_bytes = build_pdf(data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="legalrag-answer.pdf"'},
        )

    @app.get("/eval", response_model=EvalResponse)
    def run_eval(k: str = "1,3,5,10") -> EvalResponse:
        try:
            k_values = sorted({int(x) for x in k.split(",") if x.strip()})
        except ValueError:
            raise HTTPException(status_code=400, detail="k must be a comma-separated list of integers.")
        if not k_values or any(value <= 0 for value in k_values):
            raise HTTPException(status_code=400, detail="k values must be positive integers.")

        try:
            cases = _load_cases(cases_file)
            retrievers_map = _retrievers()
        except Exception:
            raise HTTPException(status_code=500, detail="Evaluation setup failed.")

        categories: dict[str, CategoryEval] = {}
        for category, retriever in retrievers_map.items():
            category_cases = [c for c in cases if c.get("category") == category]
            if not category_cases:
                categories[category] = CategoryEval(num_cases=0, metrics_by_k={})
                continue
            metrics_by_k: dict[str, dict[str, float]] = {}
            for kk in k_values:
                result = evaluate_retriever(retriever, category_cases, k=kk)
                metrics = result["metrics"]
                metrics_by_k[str(kk)] = {
                    "precision": metrics[f"precision@{kk}"],
                    "recall": metrics[f"recall@{kk}"],
                    "mrr": metrics["mrr"],
                    "map": metrics["map"],
                }
            categories[category] = CategoryEval(num_cases=len(category_cases), metrics_by_k=metrics_by_k)

        return EvalResponse(
            k_values=k_values,
            cases_file=str(cases_file),
            categories=categories,
            overall_macro=_macro_overall(categories, k_values),
        )

    if frontend.is_dir():
        app.mount("/static", StaticFiles(directory=str(frontend)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        index_file = frontend / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="frontend index.html not found")

    return app


app = create_app()


def run() -> None:  # pragma: no cover - manual entry point
    cfg = R.load_config(ROOT / "config.yaml")
    host = str(cfg.get("app", {}).get("host", "127.0.0.1"))
    port = int(cfg.get("app", {}).get("port", 8000))
    import uvicorn

    uvicorn.run("legalrag.app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run()
