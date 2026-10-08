import re
from enum import Enum, auto

class QueryClass(Enum):
    NORMAL_LEGAL_QUERY = auto()
    FACT_PATTERN_QUERY = auto()

class QueryClassifier:
    """Base class for query classification."""
    def classify(self, query: str) -> QueryClass:
        raise NotImplementedError

class RuleBasedQueryClassifier(QueryClassifier):
    """
    A lightweight, deterministic rule-based classifier that determines 
    whether a query is a standard legal lookup or a fact pattern scenario.
    """
    def __init__(self):
        # Matches queries starting with a specific scenario actor
        self.actor_pattern = re.compile(
            r"^(?:if |suppose |in a scenario where |assume )?"
            r"(?:a|an|the|my|our|two) "
            r"(?:person|company|employee|employer|man|woman|landlord|tenant|buyer|seller|contractor|police officer|doctor|driver|patient)\b", 
            re.IGNORECASE
        )
        
        # Matches personal narratives
        self.personal_pattern = re.compile(
            r"\b(my landlord|my employer|my boss|my neighbor|my tenant|my company|i was|i signed|i got|we were)\b",
            re.IGNORECASE
        )
        
        # Matches past-tense narrative verbs typical in fact patterns
        self.narrative_verbs = re.compile(
            r"\b(terminated|signed|complained|threatened|refused|failed to|entered into|sued|caused|kept in custody|assaulted|stole|fired|breached|forced|agreed|arrested|harassed)\b",
            re.IGNORECASE
        )
        
        # Matches conclusion questions asking for advice on the facts
        self.question_pattern = re.compile(
            r"(?:what legal remedy|what legal rights|can (?:the|they|he|she|i|we)|is this (?:legal|valid)|what (?:are|is) the (?:liability|consequences|punishment)|what can i do|can i sue)\b",
            re.IGNORECASE
        )

    def classify(self, query: str) -> QueryClass:
        query_clean = query.strip()
        score = 0
        
        if self.actor_pattern.search(query_clean):
            score += 2
            
        if self.personal_pattern.search(query_clean):
            score += 2
            
        if self.narrative_verbs.search(query_clean):
            score += 1
            
        if self.question_pattern.search(query_clean):
            score += 1
            
        # Fact patterns often contain longer descriptive text
        if len(query_clean.split()) > 15:
            score += 1
            
        if score >= 2:
            return QueryClass.FACT_PATTERN_QUERY
            
        return QueryClass.NORMAL_LEGAL_QUERY

