from typing import List
from .data_models import Paper
import datetime
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enrich_and_tag(raw_papers: list) -> List[Paper]:
    """
    Enrich raw paper data and add tags
    
    Args:
        raw_papers: List of raw paper dictionaries from various APIs
        
    Returns:
        List of Paper objects with tags
    """
    tagged_papers = []
    
    for raw_paper in raw_papers:
        # Process authors - handle different formats
        raw_authors = raw_paper.get("authors", [])
        processed_authors = []
        
        if isinstance(raw_authors, list):
            for author in raw_authors:
                if isinstance(author, str):
                    processed_authors.append(author)
                elif isinstance(author, dict):
                    # Handle {'@pid': '...', 'text': 'Name'} format from DBLP
                    name = author.get('text', author.get('name', author.get('display_name', '')))
                    if name:
                        processed_authors.append(name)
        elif isinstance(raw_authors, str):
            processed_authors = [raw_authors]
        
        # Create Paper object
        try:
            paper = Paper(
                title=raw_paper.get("title", ""),
                authors=processed_authors,
                year=raw_paper.get("year", 0),
                abstract=raw_paper.get("abstract", ""),
                url=raw_paper.get("url", ""),
                citation_count=raw_paper.get("citationCount", raw_paper.get("citation_count", 0)),
                venue=raw_paper.get("venue", ""),
                source=raw_paper.get("source", ""),
                status=raw_paper.get("status", ""),
                verified_in_db=raw_paper.get("verified_in_db", True),
                llm_confidence=raw_paper.get("llm_confidence", 0.0),
                llm_reasoning=raw_paper.get("llm_reasoning", raw_paper.get("llm_recommendation_reason", "")),
                llm_relevance_score=raw_paper.get("llm_relevance_score", 0),
                quality_score=raw_paper.get("quality_score", 0.0),
                score_breakdown=raw_paper.get("score_breakdown", None),
                is_baseline=raw_paper.get("is_baseline", False)
            )
        except Exception as e:
            logger.error(f"Error creating Paper object: {e}")
            logger.error(f"Raw paper data: {raw_paper}")
            continue
        
        # Add tags
        paper.has_code = _check_has_code(paper)
        paper.is_top_venue = _check_is_top_venue(paper)
        paper.is_highly_cited = _check_is_highly_cited(paper)
        paper.is_open_access = _check_is_open_access(paper)
        
        tagged_papers.append(paper)
    
    return tagged_papers

def _check_has_code(paper: Paper) -> bool:
    """
    Check if paper has code available
    """
    # Check if 'github.com' exists in url or abstract
    text_to_check = f"{paper.url} {paper.abstract}"
    return 'github.com' in text_to_check.lower()

def _check_is_top_venue(paper: Paper) -> bool:
    """
    Check if paper is published in a top venue
    """
    top_venues = [
        "cvpr", "iccv", "eccv", "neurips", "iclr", "aaai", "tpami", "ijcv"
    ]
    venue_lower = paper.venue.lower()
    return any(venue in venue_lower for venue in top_venues)

def _check_is_highly_cited(paper: Paper) -> bool:
    """
    Check if paper is highly cited
    """
    current_year = datetime.datetime.now().year
    if paper.year == 0:
        return False
    threshold = 10 * (current_year - paper.year + 1)
    return paper.citation_count > threshold

def _check_is_open_access(paper: Paper) -> bool:
    """
    Check if paper is open access
    """
    # This is a simple check - in reality, we would need to check the paper's access status
    # For now, we'll return False as a placeholder
    return False

def check_big_tech_affiliation(authors: List[str]) -> bool:
    """
    Check if any author is affiliated with Big Tech companies
    
    Args:
        authors: List of author names
        
    Returns:
        True if any author is affiliated with Big Tech
    """
    # Note: This is a placeholder function
    # In reality, we would need to check author affiliations from the API response
    # Semantic Scholar API doesn't always provide affiliation information in search results
    big_tech_companies = ["google", "meta", "microsoft", "openai"]
    
    # This is a naive check - we're just checking if company names are in author strings
    # This won't work well in practice, but it's a placeholder
    for author in authors:
        author_lower = author.lower()
        if any(company in author_lower for company in big_tech_companies):
            return True
    
    return False

if __name__ == "__main__":
    # Test the tagger
    test_papers = [
        {
            "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            "authors": ["Albert Gu", "Tri Dao"],
            "year": 2024,
            "abstract": "We introduce Mamba, a new architecture that achieves linear-time sequence modeling...",
            "url": "https://github.com/state-spaces/mamba",
            "citationCount": 100,
            "venue": "NeurIPS"
        },
        {
            "title": "Vision Transformer",
            "authors": ["Alexey Dosovitskiy", "Lucas Beyer"],
            "year": 2020,
            "abstract": "We show that a pure transformer applied directly to sequences of image patches...",
            "url": "https://arxiv.org/abs/2010.11929",
            "citationCount": 10000,
            "venue": "ICLR"
        }
    ]
    
    tagged_papers = enrich_and_tag(test_papers)
    for i, paper in enumerate(tagged_papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Has Code: {paper.has_code}")
        print(f"   Top Venue: {paper.is_top_venue}")
        print(f"   Highly Cited: {paper.is_highly_cited}")
        print()