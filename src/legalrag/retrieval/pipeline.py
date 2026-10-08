"""M3 hybrid retrieval pipeline with RRF, MMR, and structural/citation boosting."""

import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .index import FAISSDenseIndex, BM25SparseIndex, ChunkStore
from .embedding import Embedder, NOMIC_QUERY_PREFIX


def extract_explicit_references(query: str) -> List[Tuple[str, str]]:
    """Extract explicit legal references from the query (e.g., Article 21, Section 302)."""
    pattern = re.compile(
        r'\b((?:(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth)\s+Schedule\s+)?Entry|Article|Section|Part|Chapter|Schedule)\s+([IVXLCDM\d]+[A-Z]?)\b', 
        re.IGNORECASE
    )
    refs = []
    for match in pattern.finditer(query):
        ref_type = " ".join(word.capitalize() for word in match.group(1).split())
        refs.append((ref_type, match.group(2).upper()))
    return refs


def is_structural_match(chunk_meta: dict, refs: List[Tuple[str, str]]) -> bool:
    """True if chunk itself is one of the referenced legal units, conforming to hierarchy rules."""
    section = chunk_meta.get("section")
    if not section:
        return False
        
    path_str = " ".join(chunk_meta.get("hierarchy_path", [])).lower()
    
    for ref_type, ref_val in refs:
        combo = f"{ref_type} {ref_val}"
        if section == ref_val or section == combo:
            rt = ref_type.lower()
            if rt == "article":
                if "schedule" in path_str or "appendix" in path_str:
                    continue
                if "part" not in path_str:
                    continue
                return True
            elif rt.endswith("entry"):
                if "schedule" not in path_str:
                    continue
                if rt != "entry":
                    qualifier = rt.replace(" entry", "").strip()
                    if qualifier not in path_str:
                        continue
                return True
            else:
                return True
    return False


def is_hierarchy_match(chunk_meta: dict, refs: List[Tuple[str, str]]) -> bool:
    """True if chunk's hierarchy (parent, hierarchy_path) contains the referenced unit."""
    path = chunk_meta.get("hierarchy_path", [])
    parent = chunk_meta.get("parent")
    for ref_type, ref_val in refs:
        combo = f"{ref_type} {ref_val}"
        for p in path:
            if ref_val == p or combo == p:
                return True
        if parent and (parent == ref_val or parent == combo):
            return True
    return False


class HybridRetriever:
    def __init__(self, config: dict, category: str = "_merged"):
        self.config = config
        self.retrieval_cfg = config.get("retrieval", {})
        self.category = category
        
        self.index_dir = Path(self.retrieval_cfg.get("index_dir", "data/index"))
        self.top_k = int(self.retrieval_cfg.get("top_k", 10))
        self.rrf_k = int(self.retrieval_cfg.get("rrf_k", 60))
        self.mmr_lambda = float(self.retrieval_cfg.get("mmr_lambda", 0.7))
        self.use_mmr = self.retrieval_cfg.get("use_mmr", True)
        
        self.use_structural_boost = self.retrieval_cfg.get("use_structural_boost", True)
        self.structural_boost = float(self.retrieval_cfg.get("structural_boost_strength", 0.1))
        
        self.use_hierarchy_boost = self.retrieval_cfg.get("use_hierarchy_boost", True)
        self.hierarchy_boost = float(self.retrieval_cfg.get("hierarchy_boost_strength", 0.05))
        
        self.use_citation_boost = self.retrieval_cfg.get("use_citation_boost", True)
        self.citation_boost = float(self.retrieval_cfg.get("citation_boost_strength", 0.05))

        self.model_name = self.retrieval_cfg.get("embedding_model", "nomic-embed-text")
        
        use_sac = config.get("chunking", {}).get("use_sac", True)
        sac_suffix = "sac" if use_sac else "no_sac"
        self.namespace = f"{self.category}_{self.model_name}_{sac_suffix}"
        prefix = str(self.index_dir / self.namespace)
        
        self.embedder = Embedder.from_config(self.retrieval_cfg)
        
        self.dense = FAISSDenseIndex(self.embedder.dimension)
        self.dense.load(prefix + "_dense")
        
        self.sparse = BM25SparseIndex()
        self.sparse.load(prefix + "_sparse")
        
        self.chunk_store = ChunkStore()
        self.chunk_store.load(str(self.index_dir / "chunk_store.json"))
        
        self._doc_id_to_idx = {did: i for i, did in enumerate(self.dense.doc_ids)}

    def search(self, query: str) -> List[Tuple[str, float, dict]]:
        doc_prefix = NOMIC_QUERY_PREFIX if "nomic" in self.model_name else ""
        query_emb = self.embedder.encode([query], prompt_prefix=doc_prefix, normalize=True)[0]
        
        fetch_k = self.top_k * 5
        dense_results = self.dense.search(query_emb, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)
        
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        refs = extract_explicit_references(query)
        
        # --- Metadata Candidate Expansion ---
        if refs:
            # We access the protected _store dict for fast scanning
            for did, meta in self.chunk_store._store.items():
                if self.category and self.category != "_merged" and meta.get("category") != self.category:
                    continue
                if is_structural_match(meta, refs):
                    if did not in rrf_scores:
                        rrf_scores[did] = 0.0
        
        candidates = list(rrf_scores.keys())
        
        cited_by = {did: False for did in candidates}
        if self.use_citation_boost:
            for did in candidates:
                meta = self.chunk_store.get(did) or {}
                cites = meta.get("cites", [])
                for cite in cites:
                    c_doc = cite.get("doc")
                    c_sec = cite.get("section")
                    for other_did in candidates:
                        if other_did == did:
                            continue
                        other_meta = self.chunk_store.get(other_did) or {}
                        if other_meta.get("source_doc") == c_doc:
                            if not c_sec or other_meta.get("section") == c_sec:
                                cited_by[other_did] = True

        boosted_scores = {}
        for did in candidates:
            score = rrf_scores[did]
            meta = self.chunk_store.get(did) or {}
            
            if refs:
                if self.use_structural_boost and is_structural_match(meta, refs):
                    score += self.structural_boost
                elif self.use_hierarchy_boost and is_hierarchy_match(meta, refs):
                    score += self.hierarchy_boost
                    
            if self.use_citation_boost and cited_by[did]:
                score += self.citation_boost
                
            boosted_scores[did] = score
            
        sorted_candidates = sorted(boosted_scores.keys(), key=lambda x: boosted_scores[x], reverse=True)
        
        if self.use_mmr and len(sorted_candidates) > 0:
            final_ids = self._apply_mmr(sorted_candidates, boosted_scores)
        else:
            final_ids = sorted_candidates[:self.top_k]
            
        results = []
        for did in final_ids:
            meta = self.chunk_store.get(did) or {}
            results.append((did, boosted_scores[did], meta))
            
        return results

    def _apply_mmr(self, candidates: List[str], scores: Dict[str, float]) -> List[str]:
        cand_embs = []
        valid_cands = []
        valid_scores = []
        for did in candidates:
            if did in self._doc_id_to_idx:
                idx = self._doc_id_to_idx[did]
                vec = self.dense.index.reconstruct(int(idx))
                cand_embs.append(vec)
                valid_cands.append(did)
                valid_scores.append(scores[did])
                
        if not valid_cands:
            return []
            
        cand_embs = np.array(cand_embs, dtype=np.float32)
        
        # Normalize scores to [0, 1] for lambda weighting
        max_s = max(valid_scores)
        min_s = min(valid_scores)
        if max_s > min_s:
            rel_scores = np.array([(s - min_s)/(max_s - min_s) for s in valid_scores])
        else:
            rel_scores = np.ones(len(valid_scores))
            
        # Cosine similarity matrix (assumes vectors are already L2 normalized)
        sim_matrix = np.dot(cand_embs, cand_embs.T)
        
        selected = []
        unselected = list(range(len(valid_cands)))
        
        while len(selected) < self.top_k and unselected:
            if not selected:
                best_idx = max(unselected, key=lambda i: rel_scores[i])
            else:
                best_score = -float('inf')
                best_idx = -1
                for i in unselected:
                    max_sim = max(sim_matrix[i, j] for j in selected)
                    mmr_score = self.mmr_lambda * rel_scores[i] - (1 - self.mmr_lambda) * max_sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                        
            selected.append(best_idx)
            unselected.remove(best_idx)
            
        return [valid_cands[i] for i in selected]
