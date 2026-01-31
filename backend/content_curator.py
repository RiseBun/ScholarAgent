import json
from typing import List, Dict, Optional
from backend.agent import LLMClient

class ContentCurator:
    """
    Module to curate and rerank papers using LLM
    """
    
    def __init__(self):
        """
        Initialize ContentCurator
        """
        pass
    
    def rerank_papers(
        self,
        papers: List[Dict],
        user_query: str,
        llm_provider: str = "openai",
        api_key: str = "",
        top_n: int = 10
    ) -> Dict:
        """
        Rerank papers using LLM
        
        Args:
            papers: List of paper dictionaries
            user_query: Original user query
            llm_provider: LLM provider to use
            api_key: API key for the LLM provider
            top_n: Number of top papers to return
            
        Returns:
            Dict with reranked papers and explanations
        """
        if not papers:
            return {
                "reranked_papers": [],
                "explanations": [],
                "total_papers": 0
            }
        
        # If no API key, return papers sorted by citation count
        if not api_key:
            sorted_papers = sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)
            return {
                "reranked_papers": sorted_papers[:top_n],
                "explanations": ["Sorted by citation count" for _ in sorted_papers[:top_n]],
                "total_papers": len(papers)
            }
        
        # Prepare papers for LLM reranking
        papers_for_reranking = []
        for i, paper in enumerate(papers[:50]):  # Limit to first 50 papers to stay within token limits
            papers_for_reranking.append({
                "id": i + 1,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "venue": paper.get("venue", ""),
                "year": paper.get("year", 0),
                "citation_count": paper.get("citationCount", 0)
            })
        
        system_prompt = """
        You are an expert research reviewer specializing in scientific literature evaluation.
        Your task is to rerank papers based on their relevance to the user's query and their overall quality.
        
        Evaluation criteria:
        1. Relevance (0-10): How well does this paper address the user's specific research query?
        2. Quality (0-10): Consider venue prestige, citation count, and overall scientific rigor
        3. Novelty (0-10): How innovative is the approach compared to existing work
        4. Clarity (0-10): How well-written and understandable is the paper
        
        User query: "{user_query}"
        
        For each paper, provide:
        - A comprehensive score (0-100) that combines all criteria
        - A brief explanation of why you gave this score
        - The main contribution of the paper
        
        Output format:
        {
            "reranked_papers": [
                {
                    "id": paper_id,
                    "score": total_score,
                    "explanation": "Brief explanation",
                    "main_contribution": "Main contribution of the paper"
                },
                ...
            ]
        }
        
        Papers to evaluate:
        {papers_json}
        """
        
        papers_json = json.dumps(papers_for_reranking, ensure_ascii=False, indent=2)
        system_prompt = system_prompt.format(user_query=user_query, papers_json=papers_json)
        
        try:
            # Create LLM client
            client = LLMClient(api_key=api_key, provider=llm_provider)
            
            # Generate response
            response = client.generate(system_prompt, "Please rerank the papers based on the evaluation criteria.")
            
            if response:
                try:
                    # Parse JSON response
                    reranking_result = json.loads(response)
                    reranked_papers_info = reranking_result.get("reranked_papers", [])
                    
                    # Map back to original papers
                    reranked_original_papers = []
                    explanations = []
                    
                    for info in reranked_papers_info[:top_n]:
                        paper_id = info.get("id", 1) - 1  # Convert to 0-based index
                        if 0 <= paper_id < len(papers):
                            reranked_original_papers.append(papers[paper_id])
                            explanations.append(info.get("explanation", ""))
                    
                    return {
                        "reranked_papers": reranked_original_papers,
                        "explanations": explanations,
                        "total_papers": len(papers)
                    }
                except json.JSONDecodeError:
                    # If JSON parsing fails, return papers sorted by citation count
                    pass
        except Exception as e:
            print(f"Error reranking papers: {e}")
        
        # Fallback: Return papers sorted by citation count
        sorted_papers = sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)
        return {
            "reranked_papers": sorted_papers[:top_n],
            "explanations": ["Sorted by citation count" for _ in sorted_papers[:top_n]],
            "total_papers": len(papers)
        }
    
    def filter_papers(
        self,
        papers: List[Dict],
        filters: Dict
    ) -> List[Dict]:
        """
        Filter papers based on criteria
        
        Args:
            papers: List of paper dictionaries
            filters: Filter criteria
            
        Returns:
            Filtered list of papers
        """
        filtered_papers = []
        
        for paper in papers:
            # Apply code filter
            if filters.get("require_code", False):
                if not paper.get("has_code", False):
                    continue
            
            # Apply top venue filter
            if filters.get("require_top_venue", False):
                if not paper.get("is_top_venue", False):
                    continue
            
            # Apply highly cited filter
            if filters.get("highly_cited_only", False):
                if not paper.get("is_highly_cited", False):
                    continue
            
            # Apply affiliation filter
            affiliation_filter = filters.get("affiliation", "")
            if affiliation_filter:
                institutions = paper.get("institutions", [])
                if not any(affiliation_filter.lower() in inst.lower() for inst in institutions):
                    continue
            
            filtered_papers.append(paper)
        
        return filtered_papers
    
    def evaluate_paper_quality(
        self,
        paper: Dict
    ) -> Dict:
        """
        Evaluate paper quality based on metadata
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Dict with quality metrics
        """
        venue = paper.get("venue", "").lower()
        citations = paper.get("citationCount", 0)
        year = paper.get("year", 0)
        
        # Define top venues
        top_venues = {
            "cv": ["cvpr", "iccv", "eccv", "tpami", "ijcv"],
            "nlp": ["acl", "emnlp", "naacl", "coling", "jmlr"],
            "ml": ["icml", "neurips", "iclr", "jmlr", "jaair"],
            "robotics": ["icra", "iros", "rss", "ijrr"],
            "ai": ["aaai", "ijcai", "icaps"]
        }
        
        # Check if venue is top
        is_top_venue = False
        for field, venues in top_venues.items():
            if any(v in venue for v in venues):
                is_top_venue = True
                break
        
        # Check if highly cited
        is_highly_cited = False
        current_year = 2026  # Update as needed
        years_since_publication = current_year - year
        
        if years_since_publication <= 1 and citations >= 50:
            is_highly_cited = True
        elif years_since_publication <= 2 and citations >= 100:
            is_highly_cited = True
        elif years_since_publication <= 3 and citations >= 150:
            is_highly_cited = True
        elif citations >= 200:
            is_highly_cited = True
        
        # Check if has code (simple heuristic based on abstract)
        abstract = paper.get("abstract", "").lower()
        has_code = any(keyword in abstract for keyword in ["code", "github", "implementation", "repo", "repository"])
        
        # Check if open access
        url = paper.get("url", "")
        is_open_access = any(domain in url for domain in ["arxiv", "openaccess", "publications"])
        
        return {
            "is_top_venue": is_top_venue,
            "is_highly_cited": is_highly_cited,
            "has_code": has_code,
            "is_open_access": is_open_access,
            "quality_score": self._calculate_quality_score(citations, is_top_venue, years_since_publication)
        }
    
    def _calculate_quality_score(
        self,
        citations: int,
        is_top_venue: bool,
        years_since_publication: int
    ) -> int:
        """
        Calculate quality score based on citations, venue, and age
        
        Args:
            citations: Number of citations
            is_top_venue: Whether the paper is from a top venue
            years_since_publication: Years since publication
            
        Returns:
            Quality score (0-100)
        """
        # Base score from citations
        if citations >= 500:
            base_score = 90
        elif citations >= 200:
            base_score = 80
        elif citations >= 100:
            base_score = 70
        elif citations >= 50:
            base_score = 60
        elif citations >= 20:
            base_score = 50
        elif citations >= 10:
            base_score = 40
        else:
            base_score = 30
        
        # Adjust for top venue
        if is_top_venue:
            base_score += 10
        
        # Adjust for recency (papers within last 3 years get bonus)
        if years_since_publication <= 3:
            base_score += 5
        elif years_since_publication <= 5:
            base_score += 2
        
        # Ensure score is within 0-100
        return min(100, max(0, base_score))

if __name__ == "__main__":
    # Test the ContentCurator
    curator = ContentCurator()
    
    # Sample papers
    sample_papers = [
        {
            "title": "Vision-Language-Action Models for Autonomous Driving",
            "abstract": "We present a novel Vision-Language-Action (VLA) model for autonomous driving that integrates visual perception, language understanding, and action prediction.",
            "venue": "CVPR 2024",
            "year": 2024,
            "citationCount": 150,
            "url": "https://arxiv.org/abs/2403.12345"
        },
        {
            "title": "End-to-End Driving with Large Language Models",
            "abstract": "This paper explores the use of large language models for end-to-end autonomous driving systems.",
            "venue": "ICLR 2024",
            "year": 2024,
            "citationCount": 80,
            "url": "https://arxiv.org/abs/2404.67890"
        },
        {
            "title": "Multimodal Fusion for Autonomous Vehicles",
            "abstract": "A multimodal fusion approach for integrating vision, language, and sensor data in autonomous vehicles.",
            "venue": "ICRA 2024",
            "year": 2024,
            "citationCount": 60,
            "url": "https://arxiv.org/abs/2405.23456"
        }
    ]
    
    # Test quality evaluation
    for paper in sample_papers:
        quality = curator.evaluate_paper_quality(paper)
        print(f"Paper: {paper['title']}")
        print(f"Quality: {quality}")
        print()
    
    # Test filtering
    filters = {
        "require_top_venue": True,
        "highly_cited_only": False
    }
    filtered = curator.filter_papers(sample_papers, filters)
    print("Filtered papers:")
    for paper in filtered:
        print(f"- {paper['title']} ({paper['venue']})")
