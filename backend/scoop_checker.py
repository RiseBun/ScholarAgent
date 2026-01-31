from typing import List, Dict, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .search_engine import PaperFetcher
from .agent import parse_user_intent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScoopChecker:
    """
    Scoop checker for research ideas
    Detects if an idea has been done before and suggests差异化 directions
    """
    
    def __init__(self):
        self.fetcher = PaperFetcher()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        logger.info("Initialized ScoopChecker")
    
    def check_idea(self, idea: str, llm_provider: str = "openai", api_key: str = "", limit: int = 50) -> Dict:
        """
        Check if an idea has been scooped and suggest differentiation
        
        Args:
            idea: Research idea description
            llm_provider: LLM provider for analysis
            api_key: API key for LLM
            limit: Maximum number of papers to check
            
        Returns:
            Dict with scoop analysis and differentiation suggestions
        """
        logger.info(f"Checking idea: '{idea}'")
        
        # 1. Search for similar papers
        similar_papers = self._search_similar_papers(idea, limit)
        logger.info(f"Found {len(similar_papers)} potentially similar papers")
        
        # 2. Calculate similarity scores
        similarities = self._calculate_similarities(idea, similar_papers)
        
        # 3. Identify closest matches
        closest_matches = self._get_closest_matches(similar_papers, similarities, top_k=5)
        
        # 4. Analyze scoop risk
        scoop_analysis = self._analyze_scoop_risk(closest_matches)
        
        # 5. Generate differentiation suggestions
        differentiation = self._generate_differentiation(idea, closest_matches, llm_provider, api_key)
        
        return {
            "idea": idea,
            "scoop_risk": scoop_analysis["risk_level"],
            "risk_score": scoop_analysis["risk_score"],
            "closest_matches": closest_matches,
            "differentiation_suggestions": differentiation["suggestions"],
            "gap_analysis": differentiation["gap_analysis"]
        }
    
    def _search_similar_papers(self, idea: str, limit: int) -> List[Dict]:
        """
        Search for papers that might be similar to the idea
        """
        # Extract keywords from idea
        keywords = self._extract_keywords(idea)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Search papers
        all_papers = []
        for keyword in keywords:
            try:
                papers = self.fetcher.search_papers(keyword, limit=limit//len(keywords))
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers for keyword: '{keyword}'")
            except Exception as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
        
        # Remove duplicates
        unique_papers = self._remove_duplicates(all_papers)
        return unique_papers[:limit]
    
    def _extract_keywords(self, idea: str) -> List[str]:
        """
        Extract keywords from idea
        """
        # Simple keyword extraction - in production, use NLP
        import re
        
        # Split idea into sentences and take first few
        sentences = re.split('[.!?]', idea)
        keywords = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            # Extract meaningful phrases
            words = sentence.split()
            if len(words) > 3:
                # Create keyword phrases
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    keywords.append(phrase)
            elif words:
                keywords.append(' '.join(words))
        
        # Add original idea as fallback
        keywords.append(idea)
        
        return list(set(keywords[:5]))  # Limit to 5 unique keywords
    
    def _calculate_similarities(self, idea: str, papers: List[Dict]) -> List[float]:
        """
        Calculate similarity between idea and papers
        """
        if not papers:
            return []
        
        # Create corpus
        corpus = [idea]
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            corpus.append(text)
        
        # Vectorize
        try:
            vectors = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarities
            idea_vector = vectors[0]
            paper_vectors = vectors[1:]
            
            similarities = cosine_similarity(idea_vector, paper_vectors)[0]
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            # Return default similarities
            return [0.0] * len(papers)
    
    def _get_closest_matches(self, papers: List[Dict], similarities: List[float], top_k: int = 5) -> List[Dict]:
        """
        Get closest matching papers
        """
        if not papers or not similarities:
            return []
        
        # Sort papers by similarity
        sorted_pairs = sorted(zip(papers, similarities), key=lambda x: x[1], reverse=True)
        
        # Format closest matches
        closest_matches = []
        for paper, similarity in sorted_pairs[:top_k]:
            closest_matches.append({
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year", 0),
                "url": paper.get("url", ""),
                "abstract": paper.get("abstract", ""),
                "similarity_score": float(similarity),
                "venue": paper.get("venue", ""),
                "citation_count": paper.get("citationCount", 0)
            })
        
        return closest_matches
    
    def _analyze_scoop_risk(self, closest_matches: List[Dict]) -> Dict:
        """
        Analyze scoop risk based on closest matches
        """
        if not closest_matches:
            return {
                "risk_level": "Low",
                "risk_score": 0.0,
                "explanation": "No similar papers found"
            }
        
        # Calculate average similarity score
        avg_similarity = np.mean([match["similarity_score"] for match in closest_matches])
        max_similarity = max([match["similarity_score"] for match in closest_matches])
        
        # Determine risk level
        if max_similarity > 0.8:
            risk_level = "High"
            explanation = "Very similar work found"
        elif max_similarity > 0.6:
            risk_level = "Medium"
            explanation = "Somewhat similar work found"
        else:
            risk_level = "Low"
            explanation = "No highly similar work found"
        
        return {
            "risk_level": risk_level,
            "risk_score": float(max_similarity),
            "explanation": explanation
        }
    
    def _generate_differentiation(self, idea: str, closest_matches: List[Dict], llm_provider: str, api_key: str) -> Dict:
        """
        Generate differentiation suggestions
        """
        if not closest_matches:
            return {
                "suggestions": ["No similar work found. Your idea appears to be novel."],
                "gap_analysis": "No gaps identified as no similar work found."
            }
        
        # Create prompt for LLM
        prompt = self._create_differentiation_prompt(idea, closest_matches)
        
        try:
            # Use LLM to generate suggestions
            result = parse_user_intent(prompt, llm_provider, api_key)
            
            # Parse response
            if isinstance(result, dict):
                suggestions = result.get("suggestions", ["No suggestions generated"])
                gap_analysis = result.get("gap_analysis", "No gap analysis available")
            else:
                # Fallback if response is not structured
                suggestions = [str(result)]
                gap_analysis = "Basic gap analysis not available"
                
        except Exception as e:
            logger.error(f"LLM differentiation failed: {e}")
            # Fallback suggestions
            suggestions = self._fallback_differentiation(idea, closest_matches)
            gap_analysis = "Gap analysis failed due to LLM error"
        
        return {
            "suggestions": suggestions,
            "gap_analysis": gap_analysis
        }
    
    def _create_differentiation_prompt(self, idea: str, closest_matches: List[Dict]) -> str:
        """
        Create prompt for LLM to generate differentiation suggestions
        """
        prompt = f"""
        You are an expert research advisor. A student has come up with this research idea:
        
        Idea: {idea}
        
        However, I've found these similar papers:
        
        """
        
        for i, match in enumerate(closest_matches, 1):
            prompt += f"Paper {i}:\n"
            prompt += f"Title: {match['title']}\n"
            prompt += f"Year: {match['year']}\n"
            prompt += f"Abstract: {match['abstract'][:200]}...\n"
            prompt += f"Similarity: {match['similarity_score']:.2f}\n\n"
        
        prompt += """
        Please:
        1. Analyze the gaps between the student's idea and these existing papers
        2. Generate 3-5 specific differentiation strategies that would make the student's work novel
        3. For each strategy, explain why it would address a gap in the existing literature
        
        Format your response as a JSON object with:
        - suggestions: array of differentiation strategies
        - gap_analysis: string describing the key gaps
        """
        
        return prompt
    
    def _fallback_differentiation(self, idea: str, closest_matches: List[Dict]) -> List[str]:
        """
        Fallback differentiation suggestions when LLM fails
        """
        suggestions = []
        
        # Suggest different application domain
        suggestions.append("Apply the idea to a different domain or industry not covered in existing work")
        
        # Suggest different methodology
        suggestions.append("Use a different methodology or technique than existing papers")
        
        # Suggest scalability or efficiency improvements
        suggestions.append("Focus on scalability, efficiency, or practical implementation challenges")
        
        # Suggest combination with other approaches
        suggestions.append("Combine your idea with complementary approaches from other fields")
        
        return suggestions
    
    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """
        Remove duplicate papers
        """
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get("title", "").lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
