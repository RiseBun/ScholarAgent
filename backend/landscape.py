from typing import List, Dict, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from .search_engine import PaperFetcher
from .agent import parse_user_intent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandscapeGenerator:
    """
    Knowledge landscape generator for research fields
    Helps users understand research trends and identify potential gaps
    """
    
    def __init__(self):
        self.fetcher = PaperFetcher()
        logger.info("Initialized LandscapeGenerator")
    
    def generate_landscape(self, query: str, llm_provider: str = "openai", api_key: str = "", limit: int = 100) -> Dict:
        """
        Generate research landscape for a given query
        
        Args:
            query: Broad research field (e.g., "artificial intelligence")
            llm_provider: LLM provider for keyword expansion
            api_key: API key for LLM
            limit: Maximum number of papers to analyze
            
        Returns:
            Dict with landscape data and visualization
        """
        logger.info(f"Generating landscape for query: '{query}'")
        
        # 1. Keyword expansion
        subtopics = self._expand_keywords(query, llm_provider, api_key)
        logger.info(f"Expanded to {len(subtopics)} subtopics")
        
        # 2. Search papers for each subtopic
        all_papers = []
        for subtopic in subtopics:
            try:
                papers = self.fetcher.search_papers(subtopic, limit=limit//len(subtopics))
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers for subtopic: '{subtopic}'")
            except Exception as e:
                logger.error(f"Error searching for subtopic '{subtopic}': {e}")
        
        # Remove duplicates
        unique_papers = self._remove_duplicates(all_papers)
        logger.info(f"Total unique papers: {len(unique_papers)}")
        
        # 3. Cluster papers
        clusters = self._cluster_papers(unique_papers)
        
        # 4. Generate visualization
        visualization = self._generate_visualization(unique_papers, clusters)
        
        # 5. Analyze trends
        trends = self._analyze_trends(unique_papers, clusters)
        
        return {
            "subtopics": subtopics,
            "total_papers": len(unique_papers),
            "clusters": trends,
            "visualization": visualization
        }
    
    def _expand_keywords(self, query: str, llm_provider: str, api_key: str) -> List[str]:
        """
        Expand query into related subtopics
        """
        try:
            # Use LLM to expand keywords
            intent = parse_user_intent(
                f"List 10 specific subtopics or research directions related to '{query}'",
                llm_provider, api_key
            )
            
            # Extract subtopics from LLM response
            if isinstance(intent, dict) and "keywords" in intent:
                return [intent["keywords"]]  # Fallback if LLM doesn't return proper structure
            
        except Exception as e:
            logger.error(f"Error expanding keywords: {e}")
        
        # Default subtopics if LLM fails
        default_subtopics = [
            f"{query} fundamentals",
            f"{query} applications",
            f"{query} challenges",
            f"{query} recent advances",
            f"{query} future directions"
        ]
        return default_subtopics
    
    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """
        Remove duplicate papers based on title
        """
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get("title", "").lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _cluster_papers(self, papers: List[Dict]) -> List[int]:
        """
        Cluster papers based on abstract similarity
        """
        if len(papers) < 2:
            return [0] * len(papers)
        
        # Extract abstracts
        abstracts = [paper.get("abstract", "") for paper in papers]
        
        # Vectorize abstracts
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            X = vectorizer.fit_transform(abstracts)
        except Exception as e:
            logger.error(f"Error vectorizing abstracts: {e}")
            return [0] * len(papers)
        
        # Determine optimal number of clusters
        n_clusters = min(5, len(papers))
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        try:
            clusters = kmeans.fit_predict(X)
        except Exception as e:
            logger.error(f"Error clustering papers: {e}")
            return [0] * len(papers)
        
        return clusters.tolist()
    
    def _generate_visualization(self, papers: List[Dict], clusters: List[int]) -> str:
        """
        Generate visualization of research landscape
        """
        try:
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            # Use citation count and year as axes
            years = [paper.get("year", 2020) for paper in papers]
            citations = [paper.get("citationCount", 0) for paper in papers]
            
            # Create scatter plot
            scatter = plt.scatter(
                years,
                citations,
                c=clusters,
                s=50,
                alpha=0.7,
                cmap='viridis'
            )
            
            # Add colorbar
            plt.colorbar(scatter, label='Cluster')
            
            # Add labels
            plt.title('Research Landscape', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Citation Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Convert plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""
    
    def _analyze_trends(self, papers: List[Dict], clusters: List[int]) -> List[Dict]:
        """
        Analyze trends in each cluster
        """
        cluster_analysis = []
        
        # Group papers by cluster
        clustered_papers = {}
        for i, cluster in enumerate(clusters):
            if cluster not in clustered_papers:
                clustered_papers[cluster] = []
            clustered_papers[cluster].append(papers[i])
        
        # Analyze each cluster
        for cluster_id, cluster_papers in clustered_papers.items():
            if not cluster_papers:
                continue
            
            # Calculate metrics
            avg_citations = np.mean([p.get("citationCount", 0) for p in cluster_papers])
            avg_year = np.mean([p.get("year", 2020) for p in cluster_papers])
            paper_count = len(cluster_papers)
            
            # Determine cluster type
            if avg_year >= 2024 and avg_citations < 10:
                cluster_type = "Emerging (Blue Ocean)"
            elif avg_citations > 50:
                cluster_type = "Established (Red Ocean)"
            else:
                cluster_type = "Developing"
            
            cluster_analysis.append({
                "cluster_id": cluster_id,
                "paper_count": paper_count,
                "average_citations": float(avg_citations),
                "average_year": float(avg_year),
                "cluster_type": cluster_type,
                "sample_titles": [p.get("title", "") for p in cluster_papers[:3]]
            })
        
        return cluster_analysis
