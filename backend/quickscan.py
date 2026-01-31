from typing import List, Dict
import logging
from .agent import parse_user_intent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickScanner:
    """
    Quick scanner for papers
    Extracts core information and identifies gaps without reading full papers
    """
    
    def __init__(self):
        logger.info("Initialized QuickScanner")
    
    def quick_scan_papers(self, papers: List[Dict], llm_provider: str = "openai", api_key: str = "") -> List[Dict]:
        """
        Quick scan multiple papers
        
        Args:
            papers: List of paper dictionaries
            llm_provider: LLM provider for analysis
            api_key: API key for LLM
            
        Returns:
            List of analyzed papers with extracted information
        """
        logger.info(f"Quick scanning {len(papers)} papers")
        
        analyzed_papers = []
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Scanning paper {i}/{len(papers)}: {paper.get('title', 'Untitled')}")
            
            try:
                analysis = self._scan_single_paper(paper, llm_provider, api_key)
                analyzed_papers.append(analysis)
            except Exception as e:
                logger.error(f"Error scanning paper: {e}")
                # Add basic paper info if analysis fails
                analyzed_papers.append({
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", 0),
                    "url": paper.get("url", ""),
                    "core_contribution": "Analysis failed",
                    "weaknesses": [],
                    "future_work": [],
                    "analysis_failed": True
                })
        
        return analyzed_papers
    
    def _scan_single_paper(self, paper: Dict, llm_provider: str, api_key: str) -> Dict:
        """
        Quick scan a single paper
        
        Args:
            paper: Paper dictionary
            llm_provider: LLM provider for analysis
            api_key: API key for LLM
            
        Returns:
            Dict with extracted information
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert research reviewer. Please analyze the following paper and extract:
        
        Paper Title: {title}
        Abstract:
        {abstract}
        
        Please provide:
        1. Core Contribution: A single sentence summarizing the main contribution
        2. Weaknesses: List 2-3 potential weaknesses or limitations
        3. Future Work: List 2-3 possible future research directions based on this work
        
        Format your response as a JSON object with keys:
        - core_contribution
        - weaknesses (array)
        - future_work (array)
        """
        
        # Use LLM to analyze
        try:
            result = parse_user_intent(prompt, llm_provider, api_key)
            
            # Parse LLM response
            if isinstance(result, dict):
                # Check if result has the expected structure
                if all(key in result for key in ["core_contribution", "weaknesses", "future_work"]):
                    analysis = result
                else:
                    # Try to extract information from keywords
                    analysis = {
                        "core_contribution": result.get("keywords", ""),
                        "weaknesses": result.get("weaknesses", ["Analysis incomplete"]),
                        "future_work": result.get("future_work", ["Analysis incomplete"])
                    }
            else:
                # Fallback if result is not a dict
                analysis = {
                    "core_contribution": str(result),
                    "weaknesses": ["Analysis failed to parse"],
                    "future_work": ["Analysis failed to parse"]
                }
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback analysis
            analysis = self._fallback_analysis(paper)
        
        # Combine with original paper info
        return {
            "title": title,
            "authors": paper.get("authors", []),
            "year": paper.get("year", 0),
            "url": paper.get("url", ""),
            "citation_count": paper.get("citationCount", 0),
            "venue": paper.get("venue", ""),
            "core_contribution": analysis.get("core_contribution", ""),
            "weaknesses": analysis.get("weaknesses", []),
            "future_work": analysis.get("future_work", []),
            "analysis_failed": False
        }
    
    def _fallback_analysis(self, paper: Dict) -> Dict:
        """
        Fallback analysis when LLM fails
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Basic analysis based on keywords
        """
        abstract = paper.get("abstract", "").lower()
        
        # Simple keyword-based analysis
        weaknesses = []
        future_work = []
        
        # Look for weakness indicators
        if "limitation" in abstract or "challenge" in abstract:
            weaknesses.append("Mentioned limitations in abstract")
        if "future work" in abstract or "future research" in abstract:
            future_work.append("Mentioned future work in abstract")
        
        # Add generic items if none found
        if not weaknesses:
            weaknesses.append("No explicit limitations mentioned")
        if not future_work:
            future_work.append("No explicit future work mentioned")
        
        return {
            "core_contribution": paper.get("title", ""),
            "weaknesses": weaknesses,
            "future_work": future_work
        }
    
    def generate_comparison_table(self, analyzed_papers: List[Dict]) -> str:
        """
        Generate comparison table for analyzed papers
        
        Args:
            analyzed_papers: List of analyzed papers
            
        Returns:
            HTML table string
        """
        # Generate HTML table
        table_html = """
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Title</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Year</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Core Contribution</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Weaknesses</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Future Work</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for paper in analyzed_papers:
            title = paper.get("title", "")
            year = paper.get("year", 0)
            core_contribution = paper.get("core_contribution", "")
            weaknesses = paper.get("weaknesses", [])
            future_work = paper.get("future_work", [])
            
            # Format weaknesses and future work
            weaknesses_str = "<br>".join(weaknesses)
            future_work_str = "<br>".join(future_work)
            
            # Add row
            table_html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">{title}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{year}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{core_contribution}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{weaknesses_str}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{future_work_str}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
