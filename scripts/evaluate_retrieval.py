#!/usr/bin/env python3
"""M6 retrieval evaluation over the Acts, Constitution and Judgments indexes.

Loads the evaluation cases (``data/eval/retrieval_cases.jsonl``), evaluates each
category separately using the existing ``HybridRetriever`` and
``legalrag.eval.runner.evaluate_retriever``, and writes a machine-readable JSON
report (``data/eval/retrieval_report.json``).

This script only reads existing indexes. It never rebuilds indexes, re-ingests
documents, or calls OpenRouter / any generation or external API.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k-values 1,3,5,10 --out data/eval/retrieval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from legalrag.eval.runner import evaluate_retriever

ROOT = Path(__file__).resolve().parent.parent

# category -> (config file, HybridRetriever category / index namespace prefix)
CATEGORY_CONFIG = {
    "acts": ("config_acts_eval.yaml", "acts"),
    "constitution": ("config.yaml", "constitution"),
    "judgments": ("config_judgments.yaml", "judgments"),
}
DEFAULT_K_VALUES = [1, 3, 5, 10]
METRIC_NAMES = ("precision", "recall", "mrr", "map")


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_category(retriever, cases, k_values):
    """Evaluate *retriever* on *cases* at each k.

    Returns per-k macro metrics, per-query metrics at every k, and the list of
    queries with no relevant chunk in the top results.
    """
    if not cases:
        raise ValueError("no cases for category")

    runs = {k: evaluate_retriever(retriever, cases, k=k) for k in sorted(k_values)}
    max_k = max(k_values)

    metrics_by_k = {}
    for k in sorted(k_values):
        m = runs[k]["metrics"]
        metrics_by_k[str(k)] = {
            "precision": m[f"precision@{k}"],
            "recall": m[f"recall@{k}"],
            "mrr": m["mrr"],
            "map": m["map"],
        }

    per_query = []
    zero_hit = []
    for i, case in enumerate(cases):
        relevant = {str(x) for x in case["relevant_ids"]}
        retrieved_max = runs[max_k]["results"][i]["retrieved_ids"]
        row = {
            "query": case["query"],
            "relevant_ids": sorted(relevant),
            "retrieved_ids": retrieved_max,
        }
        for k in sorted(k_values):
            r = runs[k]["results"][i]
            row[f"precision@{k}"] = r[f"precision@{k}"]
            row[f"recall@{k}"] = r[f"recall@{k}"]
            row[f"mrr@{k}"] = r["mrr"]
            row[f"map@{k}"] = r["map"]
        per_query.append(row)
        if not (set(retrieved_max) & relevant):
            zero_hit.append(case["query"])

    return {
        "num_cases": len(cases),
        "metrics_by_k": metrics_by_k,
        "per_query": per_query,
        "zero_hit_queries": zero_hit,
    }


def macro_overall(category_reports, k_values):
    """Macro-average each metric across categories that produced results."""
    overall = {}
    for k in sorted(k_values):
        ks = str(k)
        cats = [
            c for c in category_reports.values()
            if "metrics_by_k" in c and ks in c["metrics_by_k"]
        ]
        if not cats:
            overall[ks] = {}
            continue
        overall[ks] = {
            metric: sum(c["metrics_by_k"][ks][metric] for c in cats) / len(cats)
            for metric in METRIC_NAMES
        }
    return overall


def build_report(cases, k_values, config_map=None, retriever_factory=None):
    config_map = config_map or CATEGORY_CONFIG
    if retriever_factory is None:
        from legalrag.retrieval import HybridRetriever
        retriever_factory = HybridRetriever

    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)

    categories = {}
    for category in ("acts", "constitution", "judgments"):
        cat_cases = by_cat.get(category, [])
        entry = {"num_cases": len(cat_cases)}
        if not cat_cases:
            entry["error"] = "no cases for this category"
            categories[category] = entry
            continue
        if category not in config_map:
            entry["error"] = f"no config mapping for category {category!r}"
            categories[category] = entry
            continue

        cfg_file, cat_name = config_map[category]
        try:
            cfg = load_config(ROOT / cfg_file)
            retriever = retriever_factory(cfg, category=cat_name)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the report
            entry["error"] = f"{type(exc).__name__}: {exc}"
            categories[category] = entry
            continue

        entry["index_dir"] = str(cfg.get("retrieval", {}).get("index_dir"))
        entry.update(evaluate_category(retriever, cat_cases, k_values))
        categories[category] = entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k_values": sorted(k_values),
        "categories": categories,
        "overall_macro": macro_overall(categories, k_values),
    }


def _fmt(metrics):
    return (f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  MAP={metrics['map']:.3f}")


def print_report(report):
    print("=" * 78)
    print("M6 RETRIEVAL EVALUATION")
    print("=" * 78)
    for category, entry in report["categories"].items():
        print(f"\n## {category}  (cases={entry.get('num_cases')})")
        if "error" in entry:
            print(f"   ERROR: {entry['error']}")
            continue
        print(f"   index_dir={entry.get('index_dir')}")
        for k in report["k_values"]:
            print(f"   k={k:<2} {_fmt(entry['metrics_by_k'][str(k)])}")
        if entry["zero_hit_queries"]:
            print("   zero-hit queries:")
            for q in entry["zero_hit_queries"]:
                print(f"     - {q}")
    print("\n" + "-" * 78)
    print("OVERALL (macro across categories)")
    for k in report["k_values"]:
        m = report["overall_macro"].get(str(k), {})
        if m:
            print(f"   k={k:<2} {_fmt(m)}")
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description="M6 retrieval evaluation.")
    ap.add_argument("--cases", default=str(ROOT / "data/eval/retrieval_cases.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/retrieval_report.json"))
    ap.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    args = ap.parse_args(argv)

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    cases = load_cases(args.cases)
    print(f"loaded {len(cases)} cases from {args.cases}")

    report = build_report(cases, k_values)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
