"""Retrieval sub-package: embed, index, retriever, adaptive loop. (Milestones 2–3, 6)"""

from .builder import IndexBuilder
from .embedding import Embedder, SentenceTransformerEmbedder
from .index import BM25SparseIndex, FAISSDenseIndex, IndexStaleError, ChunkStore
from .pipeline import HybridRetriever
from .classifier import QueryClass, QueryClassifier, RuleBasedQueryClassifier
from .rewriter import LegalIssueRewriter, RuleBasedLegalIssueRewriter
from .adaptive import AdaptiveRetriever

__all__ = [
    "IndexBuilder",
    "Embedder",
    "SentenceTransformerEmbedder",
    "BM25SparseIndex",
    "FAISSDenseIndex",
    "IndexStaleError",
    "ChunkStore",
    "HybridRetriever",
    "QueryClass",
    "QueryClassifier",
    "RuleBasedQueryClassifier",
    "LegalIssueRewriter",
    "RuleBasedLegalIssueRewriter",
    "AdaptiveRetriever",
]
