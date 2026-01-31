from pydantic import BaseModel
from typing import List

class Paper(BaseModel):
    """
    Paper data model with basic information and tags
    """
    # Basic information
    title: str
    authors: List[str]
    year: int
    abstract: str
    url: str
    citation_count: int
    venue: str
    
    # Tags
    has_code: bool = False
    is_top_venue: bool = False
    is_highly_cited: bool = False
    is_open_access: bool = False