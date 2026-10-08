import re

from .classifier import QueryClass

class LegalIssueRewriter:
    """Base class for rewriting fact-pattern queries into concise legal issues."""
    def rewrite(self, query: str, query_class: QueryClass) -> str:
        raise NotImplementedError


class RuleBasedLegalIssueRewriter(LegalIssueRewriter):
    """
    Deterministic rule-based rewriter that cleans narrative noise,
    removes stopwords, and injects relevant legal terms.
    """
    def __init__(self):
        # Stopwords and narrative phrases to strip
        self.stopwords = {
            "a", "an", "the", "my", "our", "person", "company", "man", "woman", 
            "what", "is", "are", "was", "were", "can", "may", "be", "available", "apply",
            "after", "being", "for", "several", "days", "and", "in", "with",
            "without", "about", "it", "this", "do", "does", "did", "to", "of", "on"
        }
        
        # Mappings from fact pattern triggers to legal issues/keywords
        self.concept_mappings = {
            r"\bthreatened with violence\b": "threat violence coercion duress",
            r"\bwithout being produced before a magistrate\b": "failure to produce before magistrate",
            r"\bwithout notice\b": "lack of notice",
            r"\bterminated an employee\b": "termination employee",
            r"\bcomplained about discrimination\b": "discrimination complaint",
            r"\barrested\b": "arrest",
            r"\bsigned a contract\b": "contract validity",
            r"\bkept in custody\b": "custody detention"
        }
        
    def rewrite(self, query: str, query_class: QueryClass) -> str:
        if query_class == QueryClass.NORMAL_LEGAL_QUERY:
            return query
            
        rewritten = query.lower()
        
        # 1. Apply concept mappings first
        for pattern, replacement in self.concept_mappings.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
            
        # 2. Tokenize and remove stopwords/punctuation
        words = re.findall(r"\b\w+\b", rewritten)
        filtered_words = [w for w in words if w not in self.stopwords]
        
        # 3. Reconstruct string
        result = " ".join(filtered_words)
        
        # Deduplicate words while preserving order (to keep it clean)
        seen = set()
        final_words = []
        for w in result.split():
            if w not in seen:
                seen.add(w)
                final_words.append(w)
                
        return " ".join(final_words)

