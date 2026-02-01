"""
Hybrid Scorer - Multi-dimensional quality scoring for academic papers
Score = (Relevance × w_r) + (Citation × w_c) + (Venue × w_v) + (Recency × w_t)
"""
import re
import math
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Top venues by field
TOP_VENUES = {
    # Computer Vision
    "cvpr": 1.0,
    "iccv": 1.0,
    "eccv": 0.95,
    # Machine Learning
    "neurips": 1.0,
    "nips": 1.0,
    "icml": 1.0,
    "iclr": 0.95,
    # AI General
    "aaai": 0.9,
    "ijcai": 0.9,
    # Robotics
    "icra": 0.9,
    "iros": 0.85,
    "corl": 0.9,
    "rss": 0.9,
    # NLP
    "acl": 1.0,
    "emnlp": 0.95,
    "naacl": 0.9,
    # Journals
    "tpami": 1.0,
    "ijcv": 0.95,
    "jmlr": 0.95,
    "nature": 1.0,
    "science": 1.0,
}

# Weights for scoring
DEFAULT_WEIGHTS = {
    "relevance": 0.3,
    "citation": 0.25,
    "venue": 0.30,
    "recency": 0.15,
}


class HybridScorer:
    """
    Multi-dimensional quality scorer for academic papers
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.current_year = datetime.now().year
    
    def score_paper(
        self,
        paper: Dict,
        query: str = "",
        target_venue: str = ""
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive quality score for a paper
        
        Args:
            paper: Paper dictionary with title, venue, year, citation_count, etc.
            query: User's search query for relevance calculation
            target_venue: Target venue filter (e.g., "CVPR")
            
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        scores = {}
        
        # 1. Relevance Score (0-1)
        scores["relevance"] = self._calculate_relevance(paper, query)
        
        # 2. Citation Score (0-1, normalized)
        scores["citation"] = self._calculate_citation_score(paper)
        
        # 3. Venue Score (0-1)
        scores["venue"] = self._calculate_venue_score(paper, target_venue)
        
        # 4. Recency Score (0-1)
        scores["recency"] = self._calculate_recency_score(paper)
        
        # Calculate weighted total
        total = sum(
            scores[key] * self.weights[key] 
            for key in scores
        )
        
        # Normalize to 0-100 for display
        total_normalized = min(total * 100, 100)
        
        return total_normalized, scores
    
    def _calculate_relevance(self, paper: Dict, query: str) -> float:
        """Calculate relevance based on keyword matching"""
        if not query:
            return 0.5  # Default if no query
        
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        query_lower = query.lower()
        
        # Extract keywords from query
        keywords = self._extract_keywords(query_lower)
        
        if not keywords:
            return 0.5
        
        # Calculate match ratio
        title_matches = sum(1 for kw in keywords if kw in title)
        abstract_matches = sum(1 for kw in keywords if kw in abstract)
        
        # Title matches are weighted higher
        title_score = title_matches / len(keywords) if keywords else 0
        abstract_score = abstract_matches / len(keywords) if keywords else 0
        
        # Combined score (title weighted 60%, abstract 40%)
        relevance = title_score * 0.6 + abstract_score * 0.4
        
        return min(relevance, 1.0)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common words and operators
        stop_words = {'or', 'and', 'the', 'a', 'an', 'for', 'in', 'on', 'of', 'to', 'with'}
        
        # Split by spaces and operators
        words = re.split(r'[\s\-\+\(\)\"]+', query)
        
        # Filter
        keywords = [
            w.strip() for w in words 
            if w.strip() and w.lower() not in stop_words and len(w) > 2
        ]
        
        return keywords
    
    def _calculate_citation_score(self, paper: Dict) -> float:
        """Calculate citation score using logarithmic scaling"""
        citation_count = paper.get("citation_count", 0) or paper.get("citationCount", 0) or 0
        
        if citation_count <= 0:
            return 0.1  # Minimum score for uncited papers
        
        # Logarithmic scaling: log10(citations + 1) / log10(10000)
        # This gives: 1 citation -> 0.0, 10 -> 0.25, 100 -> 0.5, 1000 -> 0.75, 10000 -> 1.0
        score = math.log10(citation_count + 1) / 4.0  # log10(10000) = 4
        
        return min(score, 1.0)
    
    def _calculate_venue_score(self, paper: Dict, target_venue: str = "") -> float:
        """Calculate venue quality score"""
        venue = paper.get("venue", "").lower()
        
        # If target venue specified, prioritize exact match
        if target_venue:
            target_lower = target_venue.lower()
            
            # Exact match gets full score
            if target_lower in venue:
                return 1.0
            
            # Check if paper is from target venue but field says something else
            # e.g., venue = "arXiv" but paper was presented at CVPR
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            if target_lower in title or target_lower in abstract:
                return 0.7  # Partial match
            
            # Not from target venue, apply penalty
            return self._get_general_venue_score(venue) * 0.5
        
        return self._get_general_venue_score(venue)
    
    def _get_general_venue_score(self, venue: str) -> float:
        """Get general venue quality score"""
        venue_lower = venue.lower()
        
        # Check against top venues
        for venue_name, score in TOP_VENUES.items():
            if venue_name in venue_lower:
                return score
        
        # arXiv gets lower score (preprint)
        if "arxiv" in venue_lower:
            return 0.3
        
        # Unknown venue
        return 0.4
    
    def _calculate_recency_score(self, paper: Dict) -> float:
        """Calculate recency score with time decay"""
        year = paper.get("year", 0)
        
        if not year or year <= 0:
            return 0.3  # Unknown year
        
        years_old = self.current_year - year
        
        if years_old <= 0:
            return 1.0  # Current/future year (in press)
        elif years_old == 1:
            return 0.95
        elif years_old == 2:
            return 0.85
        elif years_old <= 5:
            return 0.7
        elif years_old <= 10:
            return 0.5
        else:
            return 0.3
    
    def score_papers(
        self,
        papers: List[Dict],
        query: str = "",
        target_venue: str = ""
    ) -> List[Dict]:
        """
        Score and sort a list of papers
        
        Args:
            papers: List of paper dictionaries
            query: User's search query
            target_venue: Target venue filter
            
        Returns:
            List of papers with quality_score and score_breakdown added, sorted by score
        """
        scored_papers = []
        
        for paper in papers:
            total_score, breakdown = self.score_paper(paper, query, target_venue)
            
            # Add score information to paper
            paper_copy = dict(paper)
            paper_copy["quality_score"] = round(total_score, 1)
            paper_copy["score_breakdown"] = {
                "relevance": round(breakdown["relevance"] * 100, 1),
                "citation": round(breakdown["citation"] * 100, 1),
                "venue": round(breakdown["venue"] * 100, 1),
                "recency": round(breakdown["recency"] * 100, 1),
            }
            
            scored_papers.append(paper_copy)
        
        # Sort by quality score descending
        scored_papers.sort(key=lambda p: p["quality_score"], reverse=True)
        
        return scored_papers
    
    def filter_by_venue(
        self,
        papers: List[Dict],
        target_venue: str,
        strict: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter papers by venue
        
        Args:
            papers: List of paper dictionaries
            target_venue: Target venue (e.g., "CVPR")
            strict: If True, only return exact matches
            
        Returns:
            Tuple of (matched_papers, other_papers)
        """
        if not target_venue:
            return papers, []
        
        target_lower = target_venue.lower()
        matched = []
        others = []
        
        for paper in papers:
            venue = paper.get("venue", "").lower()
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            # Check venue field
            if target_lower in venue:
                matched.append(paper)
            # Check title/abstract for venue mention
            elif not strict and (target_lower in title or target_lower in abstract):
                paper_copy = dict(paper)
                paper_copy["venue_match"] = "mentioned"  # Mark as mentioned but not primary venue
                matched.append(paper_copy)
            else:
                others.append(paper)
        
        return matched, others


def calculate_quality_score(paper: Dict, query: str = "", target_venue: str = "") -> Dict:
    """
    Convenience function to calculate quality score for a single paper
    
    Returns:
        Dict with quality_score and score_breakdown
    """
    scorer = HybridScorer()
    total, breakdown = scorer.score_paper(paper, query, target_venue)
    
    return {
        "quality_score": round(total, 1),
        "score_breakdown": {
            "relevance": round(breakdown["relevance"] * 100, 1),
            "citation": round(breakdown["citation"] * 100, 1),
            "venue": round(breakdown["venue"] * 100, 1),
            "recency": round(breakdown["recency"] * 100, 1),
        }
    }


if __name__ == "__main__":
    # Test the scorer
    test_papers = [
        {
            "title": "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
            "venue": "arXiv",
            "year": 2023,
            "citation_count": 500,
            "abstract": "We study how vision-language models can be used for robotic control..."
        },
        {
            "title": "PaLM-E: An Embodied Multimodal Language Model",
            "venue": "CVPR 2023",
            "year": 2023,
            "citation_count": 800,
            "abstract": "Large language models excel at reasoning tasks..."
        },
        {
            "title": "Open X-Embodiment: Robotic Learning Datasets and RT-X Models",
            "venue": "ICRA 2024",
            "year": 2024,
            "citation_count": 100,
            "abstract": "We present Open X-Embodiment, the largest robot learning dataset..."
        }
    ]
    
    scorer = HybridScorer()
    scored = scorer.score_papers(test_papers, query="VLA robotics", target_venue="CVPR")
    
    for paper in scored:
        print(f"\n{paper['title']}")
        print(f"  Quality Score: {paper['quality_score']}")
        print(f"  Breakdown: {paper['score_breakdown']}")
        print(f"  Venue: {paper['venue']}")
