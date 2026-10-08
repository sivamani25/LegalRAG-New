#!/usr/bin/env python3
"""Standalone verifier for the LegalRAG judgment dense index.

Checks the built artifacts match the production contract and the corpus.
Exits non-zero if any check fails.

Usage:
    python kaggle_verify_judgments.py \
        --index-dir /kaggle/working/judgment_index_checkpoint \
        --chunks /kaggle/input/<ds>/chunks.jsonl
"""

import argparse
import json
import os

NAMESPACE = "judgments_nomic-embed-text_sac"
EXPECTED_DIM = 768


def _corpus_bounds(chunks_path):
    first = last = None
    n = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cid = json.loads(line)["chunk_id"]
            if first is None:
                first = cid
            last = cid
            n += 1
    return first, last, n


def verify(index_dir, chunks_path, expected_total=None, expected_dim=EXPECTED_DIM,
           check_corpus=True):
    import faiss

    prefix = os.path.join(index_dir, NAMESPACE + "_dense")
    results = {}

    index = faiss.read_index(prefix + ".faiss")
    with open(prefix + "_ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    with open(prefix + "_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(index_dir, "chunk_store.json"), "r", encoding="utf-8") as f:
        store = json.load(f)

    if expected_total is None:
        if check_corpus:
            _, _, expected_total = _corpus_bounds(chunks_path)
        else:
            expected_total = meta.get("chunk_count")

    results["faiss_dimension_768"] = (index.d == expected_dim)
    results["faiss_ntotal"] = (index.ntotal == expected_total)
    results["ids_count"] = (len(ids) == expected_total)
    results["ids_unique"] = (len(set(ids)) == len(ids))
    results["no_duplicate_ids"] = (len(set(ids)) == len(ids) == expected_total)
    results["chunk_store_count"] = (len(store) == expected_total)
    results["meta_chunk_count"] = (meta.get("chunk_count") == expected_total)
    results["meta_has_fingerprints"] = (
        bool(meta.get("corpus_fingerprint")) and bool(meta.get("config_fingerprint"))
    )
    results["ids_present_in_store"] = all(cid in store for cid in ids)

    if check_corpus:
        first, last, corpus_n = _corpus_bounds(chunks_path)
        results["corpus_count_matches"] = (corpus_n == expected_total)
        results["first_id_matches"] = (bool(ids) and ids[0] == first)
        results["last_id_matches"] = (bool(ids) and ids[-1] == last)

    info = {
        "faiss_dim": index.d,
        "faiss_ntotal": index.ntotal,
        "ids": len(ids),
        "unique_ids": len(set(ids)),
        "chunk_store": len(store),
        "meta_chunk_count": meta.get("chunk_count"),
        "expected_total": expected_total,
    }
    if check_corpus:
        info["corpus_first"] = first
        info["corpus_last"] = last
        info["ids_first"] = ids[0] if ids else None
        info["ids_last"] = ids[-1] if ids else None

    ok = all(results.values())
    return ok, results, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="/kaggle/working/judgment_index_checkpoint")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--expected-total", type=int, default=None)
    ap.add_argument("--no-corpus-check", action="store_true")
    args = ap.parse_args()

    ok, results, info = verify(
        args.index_dir, args.chunks,
        expected_total=args.expected_total,
        check_corpus=not args.no_corpus_check,
    )
    print("=" * 60)
    for k, v in info.items():
        print(f"{k:20}: {v}")
    print("-" * 60)
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
