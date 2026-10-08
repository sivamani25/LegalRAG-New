import sys
import yaml
import logging
import json
import gc
from pathlib import Path

from legalrag.retrieval.embedding import Embedder, NOMIC_DOCUMENT_PREFIX
from legalrag.retrieval.index import FAISSDenseIndex, BM25SparseIndex, ChunkStore, INDEX_VERSION, corpus_fingerprint, config_fingerprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_judgments")

def main(config_path="config.yaml", max_chunks=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    retrieval_cfg = cfg.get("retrieval", {})
    ingest_cfg = cfg.get("ingest", {})
    chunking_cfg = cfg.get("chunking", {})

    model_name = retrieval_cfg.get("embedding_model", "nomic-embed-text")
    normalize = retrieval_cfg.get("embedding_normalize", True)
    batch_size = int(retrieval_cfg.get("embedding_batch_size", 32))
    
    index_dir = Path(retrieval_cfg.get("index_dir", "data/index")) / "judgments"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Use config processed_dir but fallback to pilot judgments if not set correctly
    base_processed = Path(ingest_cfg.get("processed_dir", "data/processed"))
    processed_dir = base_processed
    if not processed_dir.name.endswith("judgments-full"):
        processed_dir = processed_dir / "judgments-full"
    if not processed_dir.exists():
        processed_dir = Path("data/pilot/processed/judgments-full")
        
    chunks_file = ingest_cfg.get("chunks_file", "chunks.jsonl")
    use_sac = bool(chunking_cfg.get("use_sac", True))

    chunks_path = processed_dir / chunks_file
    
    corp_fp = corpus_fingerprint(str(chunks_path))
    doc_prefix = NOMIC_DOCUMENT_PREFIX if "nomic" in model_name else ""
    conf_fp = config_fingerprint(
        embedding_model=model_name,
        sac_enabled=use_sac,
        document_prefix=doc_prefix,
        index_version=INDEX_VERSION,
    )
    
    expected_meta = {
        "corpus_fingerprint": corp_fp,
        "config_fingerprint": conf_fp,
        "embedding_model": model_name,
        "sac_enabled": use_sac,
        "document_prefix": doc_prefix,
        "index_version": INDEX_VERSION,
    }

    embedder = Embedder.from_config(retrieval_cfg)
    
    sac_suffix = "sac" if use_sac else "no_sac"
    namespace = f"judgments_{model_name}_{sac_suffix}"
    prefix = index_dir / namespace
    prefix_str = str(prefix)
    
    dense = FAISSDenseIndex(embedder.dimension)
    
    start_offset = 0
    if Path(f"{prefix_str}_dense_meta.json").exists() and Path(f"{prefix_str}_dense.faiss").exists():
        try:
            with open(f"{prefix_str}_dense_meta.json", "r") as f:
                stored_meta = json.load(f)
            stale = any(k in stored_meta and stored_meta[k] != v for k, v in expected_meta.items())
            if not stale:
                dense.load(prefix_str + "_dense", expected_metadata=expected_meta)
                start_offset = len(dense.doc_ids)
                logger.info(f"Loaded valid checkpoint. Resuming from chunk {start_offset}")
            else:
                logger.info("Existing index is stale. Starting fresh.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            dense = FAISSDenseIndex(embedder.dimension)

    store_path = index_dir / "chunk_store.json"
    store = ChunkStore()
    if store_path.exists():
        try:
            store.load(str(store_path))
            logger.info(f"Loaded existing chunk_store with {len(store)} items.")
        except Exception:
            pass

    # Read chunks incrementally
    batch_texts = []
    batch_ids = []
    texts_for_bm25 = []
    doc_ids_for_bm25 = []
    
    processed_count = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunk = json.loads(line)
            
            # 6. Reuse chunk store
            if chunk["chunk_id"] not in store._store:
                store._store[chunk["chunk_id"]] = {k: v for k, v in chunk.items() if k not in store._EXCLUDE}
            
            # We must hold all texts for BM25
            texts_for_bm25.append(chunk["text"])
            doc_ids_for_bm25.append(chunk["chunk_id"])
            
            # Skip embedding if already processed
            if processed_count < start_offset:
                processed_count += 1
                continue
                
            batch_texts.append(chunk["text"])
            batch_ids.append(chunk["chunk_id"])
            
            if len(batch_texts) == batch_size:
                embeddings = embedder.encode(
                    batch_texts,
                    normalize=normalize,
                    prompt_prefix=doc_prefix,
                    batch_size=batch_size
                )
                dense.add(embeddings, batch_ids)
                processed_count += len(batch_texts)
                batch_texts = []
                batch_ids = []
                
                # Checkpoint every 5000 chunks
                if processed_count % 5000 == 0:
                    logger.info(f"Checkpointing at {processed_count} chunks...")
                    dense.save(prefix_str + "_dense", metadata=expected_meta)
                    store.save(str(store_path))
                    gc.collect()
            
            if max_chunks and len(texts_for_bm25) >= max_chunks:
                break

    if batch_texts:
        embeddings = embedder.encode(
            batch_texts,
            normalize=normalize,
            prompt_prefix=doc_prefix,
            batch_size=batch_size
        )
        dense.add(embeddings, batch_ids)
        processed_count += len(batch_texts)
        batch_texts = []
        batch_ids = []

    # Final save
    dense.save(prefix_str + "_dense", metadata=expected_meta)
    store.save(str(store_path))
    
    logger.info("Building BM25...")
    sparse = BM25SparseIndex()
    sparse.add(texts_for_bm25, doc_ids_for_bm25)
    sparse.save(prefix_str + "_sparse", metadata=expected_meta)

    logger.info(f"Finished building. Total embeddings: {len(dense.doc_ids)}, Dimension: {dense.dimension}, ChunkStore: {len(store)}, BM25: {len(sparse.doc_ids)}")

if __name__ == "__main__":
    c_path = "config.yaml"
    m_chunks = None
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            m_chunks = int(sys.argv[1])
        else:
            c_path = sys.argv[1]
            if len(sys.argv) > 2:
                m_chunks = int(sys.argv[2])
    main(config_path=c_path, max_chunks=m_chunks)
