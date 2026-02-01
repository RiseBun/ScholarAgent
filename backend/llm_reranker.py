"""
LLM Reranker - Two-stage semantic reranking using LLM
Stage 1: API search returns many papers (coarse ranking)
Stage 2: LLM selects best papers with explanations (fine ranking)
"""
import json
import logging
from typing import Dict, List, Optional
from .agent import LLMClient

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    Use LLM to semantically rerank papers based on user intent.
    This mimics how researchers ask ChatGPT to recommend papers.
    """
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        papers: List[Dict],
        user_query: str,
        llm_provider: str,
        api_key: str,
        venue_filter: str = "",
        top_k: int = 10,
        include_baseline: bool = True
    ) -> Dict:
        """
        Rerank papers using LLM semantic understanding
        
        Args:
            papers: List of paper dicts (with title, abstract, venue, year, etc.)
            user_query: User's original research query
            llm_provider: LLM provider name
            api_key: API key for LLM
            venue_filter: Target venue (e.g., "CVPR")
            top_k: Number of top papers to return
            include_baseline: Whether to identify "must-read" baseline papers
            
        Returns:
            Dict with reranked papers and LLM reasoning
        """
        if not api_key:
            logger.warning("No API key provided for LLM reranking")
            return {
                "reranked_papers": papers[:top_k],
                "reasoning": "No API key provided, returning original order",
                "baseline_papers": []
            }
        
        if not papers:
            return {
                "reranked_papers": [],
                "reasoning": "No papers to rerank",
                "baseline_papers": []
            }
        
        # Prepare paper summaries for LLM (limit to 30 papers for faster processing)
        paper_summaries = self._prepare_summaries(papers[:30])
        
        # Build prompts
        system_prompt = self._build_system_prompt(include_baseline)
        user_prompt = self._build_user_prompt(
            paper_summaries, user_query, venue_filter, top_k
        )
        
        try:
            # Call LLM
            client = LLMClient(api_key=api_key, provider=llm_provider)
            response = client.generate(system_prompt, user_prompt)
            
            if not response:
                logger.error("LLM returned empty response for reranking")
                return {
                    "reranked_papers": papers[:top_k],
                    "reasoning": "LLM returned empty response",
                    "baseline_papers": []
                }
            
            # Parse response
            result = self._parse_response(response, papers)
            logger.info(f"LLM reranked {len(result.get('reranked_papers', []))} papers")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {e}")
            return {
                "reranked_papers": papers[:top_k],
                "reasoning": f"Error: {str(e)}",
                "baseline_papers": []
            }
    
    def _prepare_summaries(self, papers: List[Dict]) -> str:
        """Prepare concise paper summaries for LLM"""
        summaries = []
        
        for i, paper in enumerate(papers):
            title = paper.get("title", "Unknown")
            venue = paper.get("venue", "Unknown")
            year = paper.get("year", "Unknown")
            citations = paper.get("citation_count", paper.get("citationCount", 0))
            abstract = paper.get("abstract", "")[:300]  # Truncate abstract
            
            summary = f"""[{i+1}] Title: {title}
Venue: {venue} | Year: {year} | Citations: {citations}
Abstract: {abstract}..."""
            
            summaries.append(summary)
        
        return "\n\n".join(summaries)
    
    def _build_system_prompt(self, include_baseline: bool) -> str:
        """Build system prompt for LLM"""
        baseline_instruction = ""
        if include_baseline:
            baseline_instruction = """
- Identify which papers are "must-read baselines" - foundational works that subsequent papers build upon
- Mark these papers with "is_baseline": true"""
        
        return f"""You are a senior AI researcher helping to select the most relevant and high-quality papers for a research query.

Your task is to:
1. Analyze the user's research intent deeply
2. Select the TOP papers that best match their needs
3. Provide a brief explanation for why each paper is recommended
4. Prioritize papers from the specified venue if one is given{baseline_instruction}

IMPORTANT GUIDELINES:
- Focus on papers that DIRECTLY address the user's research question
- Consider both quality (citations, venue) and relevance
- If venue filter is specified (e.g., CVPR), strongly prefer papers from that venue
- Identify seminal/foundational papers in the field
- Be honest if a paper doesn't quite match the query

OUTPUT FORMAT (JSON):
{{
    "selected_papers": [
        {{
            "index": 1,
            "title": "Paper Title",
            "recommendation_reason": "Why this paper is relevant (1-2 sentences)",
            "relevance_score": 95,
            "is_baseline": false
        }}
    ],
    "overall_reasoning": "Brief summary of the selection strategy",
    "baseline_papers": [1, 5, 8]  // indices of must-read foundational papers
}}"""
    
    def _build_user_prompt(
        self,
        paper_summaries: str,
        user_query: str,
        venue_filter: str,
        top_k: int
    ) -> str:
        """Build user prompt for LLM"""
        venue_instruction = ""
        if venue_filter:
            venue_instruction = f"""
VENUE PREFERENCE: The user specifically wants papers from {venue_filter}.
Please strongly prioritize papers published in {venue_filter}. Papers from other venues should only be included if they are absolutely essential to the topic."""
        
        return f"""RESEARCH QUERY: {user_query}
{venue_instruction}

Please select the TOP {top_k} most relevant and high-quality papers from the following list:

---
{paper_summaries}
---

Remember to:
1. Prioritize papers that directly address "{user_query}"
2. Consider venue quality and citation count
3. Provide a brief explanation for each selection
4. Identify any "must-read baseline" papers

Return your response in the specified JSON format."""
    
    def _parse_response(self, response: str, original_papers: List[Dict]) -> Dict:
        """Parse LLM response and map back to original papers"""
        try:
            # Extract JSON from response
            json_str = response
            
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            json_str = json_str.strip()
            result = json.loads(json_str)
            
            # Map selected papers back to original paper objects
            reranked = []
            selected = result.get("selected_papers", [])
            baseline_indices = set(result.get("baseline_papers", []))
            
            for item in selected:
                idx = item.get("index", 0) - 1  # Convert to 0-based index
                if 0 <= idx < len(original_papers):
                    paper = dict(original_papers[idx])
                    paper["llm_recommendation_reason"] = item.get("recommendation_reason", "")
                    paper["llm_relevance_score"] = item.get("relevance_score", 0)
                    paper["is_baseline"] = item.get("is_baseline", False) or (idx + 1) in baseline_indices
                    paper["source"] = "llm_reranked"
                    reranked.append(paper)
            
            return {
                "reranked_papers": reranked,
                "reasoning": result.get("overall_reasoning", ""),
                "baseline_papers": [
                    original_papers[i-1] for i in baseline_indices 
                    if 0 < i <= len(original_papers)
                ]
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            
            # Fallback: return original papers
            return {
                "reranked_papers": original_papers[:10],
                "reasoning": "Failed to parse LLM response, returning original order",
                "baseline_papers": []
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "reranked_papers": original_papers[:10],
                "reasoning": f"Error: {str(e)}",
                "baseline_papers": []
            }
    
    def explain_paper(
        self,
        paper: Dict,
        user_query: str,
        llm_provider: str,
        api_key: str
    ) -> str:
        """
        Generate a detailed explanation of why a paper is relevant
        
        Args:
            paper: Paper dict
            user_query: User's research query
            llm_provider: LLM provider name
            api_key: API key for LLM
            
        Returns:
            Explanation string
        """
        if not api_key:
            return "No API key provided for explanation"
        
        try:
            client = LLMClient(api_key=api_key, provider=llm_provider)
            
            prompt = f"""Given the research query: "{user_query}"

Please explain in 2-3 sentences why this paper is relevant:

Title: {paper.get('title', 'Unknown')}
Venue: {paper.get('venue', 'Unknown')} | Year: {paper.get('year', 'Unknown')}
Abstract: {paper.get('abstract', 'No abstract')[:500]}

Focus on:
1. How it relates to the query
2. Key contributions
3. Why the user should read it"""
            
            response = client.generate(
                "You are a helpful research assistant.",
                prompt
            )
            
            return response if response else "Unable to generate explanation"
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Test the reranker
    reranker = LLMReranker()
    
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
            "venue": "ICML 2023",
            "year": 2023,
            "citation_count": 800,
            "abstract": "Large language models excel at reasoning tasks..."
        }
    ]
    
    # This will fail without API key, but tests the structure
    result = reranker.rerank(
        papers=test_papers,
        user_query="VLA models for robotics",
        llm_provider="openai",
        api_key="",
        venue_filter="CVPR",
        top_k=5
    )
    
    print("Result:", result)
