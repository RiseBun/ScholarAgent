from typing import List, Dict, Tuple
import logging
from .agent import parse_user_intent
from .search_engine import PaperFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IdeaBreeder:
    """
    Idea breeder for research
    Helps users generate and validate new research ideas through cross-breeding
    """
    
    def __init__(self):
        self.fetcher = PaperFetcher()
        logger.info("Initialized IdeaBreeder")
    
    def breed_idea(self, idea: str, llm_provider: str = "openai", api_key: str = "") -> Dict:
        """
        Breed and validate a research idea
        
        Args:
            idea: Initial research idea
            llm_provider: LLM provider for analysis
            api_key: API key for LLM
            
        Returns:
            Dict with idea analysis and validation
        """
        logger.info(f"Breeding idea: '{idea}'")
        
        # 1. Multi-Agent Debate
        debate = self._conduct_debate(idea, llm_provider, api_key)
        
        # 2. Analogy Search
        analogies = self._find_analogies(idea)
        
        # 3. Cross-Breeding Suggestions
        cross_breeding = self._suggest_cross_breeding(idea, debate, llm_provider, api_key)
        
        # 4. Feasibility Analysis
        feasibility = self._analyze_feasibility(idea, debate, llm_provider, api_key)
        
        return {
            "original_idea": idea,
            "debate": debate,
            "analogies": analogies,
            "cross_breeding_suggestions": cross_breeding["suggestions"],
            "refined_idea": cross_breeding["refined_idea"],
            "feasibility_analysis": feasibility
        }
    
    def _conduct_debate(self, idea: str, llm_provider: str, api_key: str) -> Dict:
        """
        Conduct a debate between optimistic and critical agents
        """
        logger.info("Conducting multi-agent debate")
        
        # Optimistic Agent
        optimist_prompt = f"""
        You are an optimistic research advisor. Your job is to:
        1. Highlight the strengths and potential of this research idea
        2. Suggest why this idea could be groundbreaking
        3. Outline possible positive outcomes and impact
        
        Idea: {idea}
        
        Provide your analysis in a structured format with clear points.
        """
        
        # Critical Agent
        critic_prompt = f"""
        You are a critical research reviewer. Your job is to:
        1. Identify potential flaws and limitations in this research idea
        2. Point out possible technical challenges
        3. Suggest what could go wrong
        4. Reference similar work that faced similar challenges
        
        Idea: {idea}
        
        Provide your analysis in a structured format with clear points.
        """
        
        try:
            # Get optimistic analysis
            optimist_response = parse_user_intent(optimist_prompt, llm_provider, api_key)
            
            # Get critical analysis
            critic_response = parse_user_intent(critic_prompt, llm_provider, api_key)
            
            return {
                "optimist": str(optimist_response),
                "critic": str(critic_response),
                "debate_conducted": True
            }
            
        except Exception as e:
            logger.error(f"Debate failed: {e}")
            return {
                "optimist": "Optimist analysis failed",
                "critic": "Critic analysis failed",
                "debate_conducted": False
            }
    
    def _find_analogies(self, idea: str) -> List[Dict]:
        """
        Find analogous research ideas from other fields
        """
        logger.info("Finding analogies for idea")
        
        # Extract key concepts from idea
        key_concepts = self._extract_key_concepts(idea)
        logger.info(f"Extracted key concepts: {key_concepts}")
        
        # Search for analogous papers in different fields
        analogies = []
        for concept in key_concepts:
            try:
                # Search for papers in different contexts
                papers = self.fetcher.search_papers(concept, limit=3)
                for paper in papers:
                    # Check if this paper is from a different field
                    if self._is_different_field(paper, idea):
                        analogies.append({
                            "title": paper.get("title", ""),
                            "authors": paper.get("authors", []),
                            "year": paper.get("year", 0),
                            "url": paper.get("url", ""),
                            "abstract": paper.get("abstract", ""),
                            "relevance": f"Based on concept: {concept}"
                        })
            except Exception as e:
                logger.error(f"Error finding analogies: {e}")
        
        # Limit to top 5 analogies
        return analogies[:5]
    
    def _extract_key_concepts(self, idea: str) -> List[str]:
        """
        Extract key concepts from idea
        """
        # Simple concept extraction
        import re
        
        # Split idea into sentences
        sentences = re.split('[.!?]', idea)
        concepts = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 2:
                # Extract noun phrases (simple approach)
                for i in range(len(words) - 1):
                    phrase = ' '.join(words[i:i+2])
                    if phrase.lower() not in ['the', 'and', 'or', 'but', 'is', 'are', 'was', 'were']:
                        concepts.append(phrase)
        
        # Add original idea keywords
        keywords = idea.split()
        for keyword in keywords[:5]:  # Top 5 keywords
            if len(keyword) > 3:
                concepts.append(keyword)
        
        return list(set(concepts[:8]))  # Limit to 8 unique concepts
    
    def _is_different_field(self, paper: Dict, original_idea: str) -> bool:
        """
        Check if paper is from a different field than original idea
        """
        # Simple field detection based on venue and abstract
        paper_text = f"{paper.get('venue', '')} {paper.get('abstract', '')}".lower()
        idea_text = original_idea.lower()
        
        # If paper text has very little overlap with idea text, it's likely a different field
        common_words = set(paper_text.split()) & set(idea_text.split())
        return len(common_words) < len(set(idea_text.split())) * 0.3
    
    def _suggest_cross_breeding(self, idea: str, debate: Dict, llm_provider: str, api_key: str) -> Dict:
        """
        Suggest cross-breeding with other research areas
        """
        logger.info("Suggesting cross-breeding opportunities")
        
        prompt = f"""
        You are a creative research advisor. Your job is to:
        1. Suggest 3-5 ways to cross-breed this research idea with other fields
        2. For each suggestion, explain why the combination would be powerful
        3. Refine the original idea based on the debate feedback
        4. Provide a concrete, actionable refined idea
        
        Original Idea: {idea}
        
        Debate Feedback:
        Optimist: {debate.get('optimist', 'No feedback')}
        Critic: {debate.get('critic', 'No feedback')}
        
        Format your response as a JSON object with:
        - suggestions: array of cross-breeding suggestions
        - refined_idea: string with the refined idea
        """
        
        try:
            result = parse_user_intent(prompt, llm_provider, api_key)
            
            if isinstance(result, dict):
                return {
                    "suggestions": result.get("suggestions", []),
                    "refined_idea": result.get("refined_idea", idea)
                }
            else:
                # Fallback if response is not structured
                return {
                    "suggestions": ["Cross-breeding suggestions failed to generate"],
                    "refined_idea": idea
                }
                
        except Exception as e:
            logger.error(f"Cross-breeding suggestion failed: {e}")
            # Fallback suggestions
            return {
                "suggestions": [
                    "Combine with machine learning techniques",
                    "Apply to a different domain or industry",
                    "Integrate with emerging technologies",
                    "Scale to larger datasets or problems"
                ],
                "refined_idea": idea
            }
    
    def _analyze_feasibility(self, idea: str, debate: Dict, llm_provider: str, api_key: str) -> Dict:
        """
        Analyze the feasibility of the idea
        """
        logger.info("Analyzing idea feasibility")
        
        prompt = f"""
        You are an expert research project manager. Your job is to:
        1. Analyze the technical feasibility of this research idea
        2. Estimate the resources required (time, compute, expertise)
        3. Identify potential roadblocks and how to overcome them
        4. Provide a feasibility score (1-10)
        
        Idea: {idea}
        
        Debate Feedback:
        Optimist: {debate.get('optimist', 'No feedback')}
        Critic: {debate.get('critic', 'No feedback')}
        
        Format your response as a JSON object with:
        - feasibility_score: number (1-10)
        - technical_analysis: string
        - resource_estimation: string
        - roadblocks: array of potential issues
        - mitigation_strategies: array of solutions
        """
        
        try:
            result = parse_user_intent(prompt, llm_provider, api_key)
            
            if isinstance(result, dict):
                return {
                    "feasibility_score": result.get("feasibility_score", 5),
                    "technical_analysis": result.get("technical_analysis", "No analysis"),
                    "resource_estimation": result.get("resource_estimation", "No estimation"),
                    "roadblocks": result.get("roadblocks", []),
                    "mitigation_strategies": result.get("mitigation_strategies", [])
                }
            else:
                # Fallback if response is not structured
                return {
                    "feasibility_score": 5,
                    "technical_analysis": "Feasibility analysis failed",
                    "resource_estimation": "No estimation",
                    "roadblocks": ["Unknown"],
                    "mitigation_strategies": ["Unknown"]
                }
                
        except Exception as e:
            logger.error(f"Feasibility analysis failed: {e}")
            # Fallback analysis
            return {
                "feasibility_score": 5,
                "technical_analysis": "Feasibility analysis failed due to error",
                "resource_estimation": "Standard research project resources required",
                "roadblocks": ["Technical challenges", "Resource constraints"],
                "mitigation_strategies": ["Start small", "Collaborate with experts"]
            }
