from typing import List, Tuple
from .classifier import QueryClass, QueryClassifier, RuleBasedQueryClassifier
from .rewriter import LegalIssueRewriter, RuleBasedLegalIssueRewriter
from .pipeline import HybridRetriever

class AdaptiveRetriever:
    """
    Orchestrates fact-pattern vs normal retrieval by applying classification,
    rewriting, and routing to the appropriate category-specific index.
    """
    def __init__(self, 
                 base_retriever: HybridRetriever, 
                 judgments_retriever: HybridRetriever = None,
                 classifier: QueryClassifier = None,
                 rewriter: LegalIssueRewriter = None):
        
        self.base_retriever = base_retriever
        self.judgments_retriever = judgments_retriever
        
        self.classifier = classifier or RuleBasedQueryClassifier()
        self.rewriter = rewriter or RuleBasedLegalIssueRewriter()

    def search(self, query: str) -> List[Tuple[str, float, dict]]:
        q_class = self.classifier.classify(query)
        
        if q_class == QueryClass.FACT_PATTERN_QUERY:
            if not self.judgments_retriever:
                raise RuntimeError("judgments_retriever must be provided for FACT_PATTERN_QUERY")
            
            rewritten_query = self.rewriter.rewrite(query, q_class)
            
            # M3b.3: Retrieve from the Judgments category first
            return self.judgments_retriever.search(rewritten_query)
            
        else:
            # Preserve the existing M3 Hybrid Retrieval behavior
            return self.base_retriever.search(query)

