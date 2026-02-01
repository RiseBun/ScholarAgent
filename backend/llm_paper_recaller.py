"""
LLM Paper Recaller - Let LLM directly recommend paper titles
"""
import json
import logging
from typing import Dict, List
from .agent import LLMClient

logger = logging.getLogger(__name__)


class LLMPaperRecaller:
    """
    Use LLM to directly recommend paper titles based on user query.
    This mimics how researchers use ChatGPT to find papers.
    """
    
    def __init__(self):
        pass
    
    def recall_papers(
        self,
        user_query: str,
        llm_provider: str,
        api_key: str,
        venue_filter: str = "",
        year_range: str = "2020-2026",
        max_papers: int = 10
    ) -> Dict:
        """
        Let LLM recommend relevant paper titles
        
        Args:
            user_query: User's research query
            llm_provider: LLM provider name
            api_key: API key for LLM
            venue_filter: Optional venue filter (e.g., "CVPR")
            year_range: Year range for papers
            max_papers: Maximum number of papers to recommend
            
        Returns:
            Dict with recommended papers and reasoning
        """
        if not api_key:
            logger.warning("No API key provided for LLM recall")
            return {
                "recommended_papers": [],
                "overall_reasoning": "No API key provided"
            }
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            user_query, venue_filter, year_range, max_papers
        )
        
        try:
            # Call LLM
            client = LLMClient(api_key=api_key, provider=llm_provider)
            response = client.generate(system_prompt, user_prompt)
            
            if not response:
                logger.error("LLM returned empty response")
                return {
                    "recommended_papers": [],
                    "overall_reasoning": "LLM returned empty response"
                }
            
            # Parse response
            result = self._parse_response(response)
            logger.info(f"LLM recalled {len(result.get('recommended_papers', []))} papers")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM recall: {e}")
            return {
                "recommended_papers": [],
                "overall_reasoning": f"Error: {str(e)}"
            }
    
    def extract_titles(self, recall_result: Dict) -> List[str]:
        """
        Extract paper titles from recall result for database search
        
        Args:
            recall_result: Result from recall_papers()
            
        Returns:
            List of paper titles
        """
        papers = recall_result.get("recommended_papers", [])
        return [p.get("title", "") for p in papers if p.get("title")]
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are an expert academic researcher with comprehensive knowledge of scientific papers across all fields.

Your task is to recommend the most relevant and influential papers based on the user's research query.

IMPORTANT GUIDELINES:
1. Only recommend papers that you are confident actually exist
2. Use exact paper titles - do not make up or guess titles
3. Focus on influential, well-cited, and recent papers
4. If a venue filter is specified (e.g., CVPR), only recommend papers from that venue
5. Prioritize papers that directly address the user's query

OUTPUT FORMAT:
You must respond with valid JSON in the following format:
{
    "recommended_papers": [
        {
            "title": "Exact Paper Title Here",
            "reasoning": "Brief explanation of why this paper is relevant",
            "confidence": 0.95,
            "estimated_year": 2024,
            "estimated_venue": "CVPR"
        }
    ],
    "overall_reasoning": "Brief summary of the recommendation strategy"
}

CONFIDENCE SCORING:
- 0.9-1.0: Very confident the paper exists with this exact title
- 0.7-0.9: Fairly confident, title may have minor variations
- 0.5-0.7: Less confident, paper may not exist or title may be different
- Below 0.5: Do not include such papers"""
    
    def _build_user_prompt(
        self,
        user_query: str,
        venue_filter: str,
        year_range: str,
        max_papers: int
    ) -> str:
        """Build user prompt for LLM"""
        prompt = f"""Research Query: {user_query}

"""
        if venue_filter:
            prompt += f"""Venue Filter: {venue_filter}
IMPORTANT: Only recommend papers published in {venue_filter} or closely related venues.

"""
        
        prompt += f"""Year Range: {year_range}
Number of Papers: Please recommend up to {max_papers} papers.

Please provide your recommendations in the specified JSON format."""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            # Handle case where LLM wraps JSON in markdown code blocks
            json_str = response
            
            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            
            # Validate structure
            if "recommended_papers" not in result:
                result["recommended_papers"] = []
            if "overall_reasoning" not in result:
                result["overall_reasoning"] = ""
            
            # Validate each paper
            valid_papers = []
            for paper in result.get("recommended_papers", []):
                if paper.get("title"):
                    valid_papers.append({
                        "title": paper.get("title", ""),
                        "reasoning": paper.get("reasoning", ""),
                        "confidence": float(paper.get("confidence", 0.5)),
                        "estimated_year": int(paper.get("estimated_year", 0)),
                        "estimated_venue": paper.get("estimated_venue", "")
                    })
            
            result["recommended_papers"] = valid_papers
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            
            # Try to extract paper titles from plain text
            return self._extract_from_plain_text(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "recommended_papers": [],
                "overall_reasoning": f"Failed to parse response: {str(e)}"
            }
    
    def _extract_from_plain_text(self, text: str) -> Dict:
        """
        Fallback: Extract paper titles from plain text response
        """
        import re
        
        papers = []
        
        # Look for quoted titles
        quoted_titles = re.findall(r'"([^"]{10,200})"', text)
        for title in quoted_titles:
            # Filter out non-title strings
            if not any(skip in title.lower() for skip in ['reasoning', 'confidence', 'overall']):
                papers.append({
                    "title": title,
                    "reasoning": "Extracted from plain text response",
                    "confidence": 0.5,
                    "estimated_year": 0,
                    "estimated_venue": ""
                })
        
        # Limit to reasonable number
        papers = papers[:15]
        
        return {
            "recommended_papers": papers,
            "overall_reasoning": "Extracted from plain text (JSON parsing failed)"
        }


if __name__ == "__main__":
    # Test the recaller
    recaller = LLMPaperRecaller()
    
    # This will fail without API key, but tests the structure
    result = recaller.recall_papers(
        user_query="Vision-Language-Action models for robotics",
        llm_provider="openai",
        api_key="",
        venue_filter="CVPR",
        max_papers=5
    )
    
    print("Result:", result)
