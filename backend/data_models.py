from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Paper(BaseModel):
    """
    Paper data model with basic information, tags, and quality scoring
    """
    # Basic information
    title: str
    authors: List[str]
    year: int
    abstract: str
    url: str
    citation_count: int = Field(default=0, alias="citationCount")
    venue: str
    
    # Tags
    has_code: bool = False
    is_top_venue: bool = False
    is_highly_cited: bool = False
    is_open_access: bool = False
    is_baseline: bool = False  # Must-read foundational paper
    
    # Data source tracking
    source: str = ""  # "arxiv", "openalex", "dblp", "llm_verified", "llm_reranked", "db_discovered"
    status: str = ""  # "Preprint", "Published", "LLM Suggestion"
    
    # LLM recommendation fields
    verified_in_db: bool = True  # Whether verified in database
    llm_confidence: float = 0.0  # LLM confidence score (0.0-1.0)
    llm_reasoning: str = ""  # LLM reasoning for recommendation
    llm_relevance_score: int = 0  # LLM relevance score (0-100)
    
    # Quality scoring (from HybridScorer)
    quality_score: float = 0.0  # Overall quality score (0-100)
    score_breakdown: Optional[Dict[str, float]] = None  # {relevance, citation, venue, recency}
    
    class Config:
        populate_by_name = True  # Allow using alias names
