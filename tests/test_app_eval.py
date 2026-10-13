"""Focused tests for the M7 /eval endpoint (fake retrievers, no indexes/models)."""

import json

from fastapi.testclient import TestClient

from legalrag.app.main import create_app


class FakeSearchRetriever:
    """Minimal retriever: search(query) -> fixed list of (id, score, meta)."""

    def __init__(self, table):
        self.table = table

    def search(self, query):
        return list(self.table.get(query, []))


def _write_cases(tmp_path):
    cases = [
        {"query": "q1", "relevant_ids": ["A"], "category": "acts"},
        {"query": "q2", "relevant_ids": ["Z"], "category": "acts"},
        {"query": "q3", "relevant_ids": ["C"], "category": "constitution"},
    ]
    path = tmp_path / "cases.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")
    return path


def make_client(tmp_path):
    retrievers = {
        "acts": FakeSearchRetriever({
            "q1": [("A", 1.0, {})],
            "q2": [("X", 1.0, {})],
        }),
        "constitution": FakeSearchRetriever({"q3": [("C", 1.0, {})]}),
        "judgments": FakeSearchRetriever({}),
    }
    app = create_app(
        retrievers=retrievers,
        cases_path=_write_cases(tmp_path),
        frontend_dir="__no_frontend__",
    )
    return TestClient(app)


def test_eval_returns_per_category_and_overall(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/eval", params={"k": "1,3"})

    assert response.status_code == 200
    body = response.json()
    assert body["k_values"] == [1, 3]
    assert body["categories"]["acts"]["num_cases"] == 2
    # q1 -> hit, q2 -> miss => P@1 = 0.5
    assert body["categories"]["acts"]["metrics_by_k"]["1"]["precision"] == 0.5
    assert body["categories"]["constitution"]["metrics_by_k"]["1"]["precision"] == 1.0
    # categories with no measurement cases report zero cases
    assert body["categories"]["judgments"]["num_cases"] == 0
    # overall macro averages only categories that have metrics
    assert body["overall_macro"]["1"]["precision"] == 0.75


def test_eval_rejects_bad_k(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/eval", params={"k": "abc"}).status_code == 400
    assert client.get("/eval", params={"k": "0"}).status_code == 400


def test_health_reports_categories(tmp_path):
    client = make_client(tmp_path)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrievers_loaded"] is True
    assert set(body["categories"]) == {"acts", "constitution", "judgments"}
