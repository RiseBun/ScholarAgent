"""
Hybrid Search Engine - Combines LLM recommendations with database search
Includes multi-dimensional quality scoring and two-stage reranking
"""
import asyncio
import logging
import re
from typing import Dict, List, Tuple
from .llm_paper_recaller import LLMPaperRecaller
from .query_expander import QueryExpander
from .search_engine import PaperFetcher
from .tagger import enrich_and_tag
from .data_models import Paper
from .hybrid_scorer import HybridScorer
from .llm_reranker import LLMReranker

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search engine that combines:
    1. LLM paper recommendations (direct title recall)
    2. Traditional query expansion and database search
    3. Multi-dimensional quality scoring
    4. Two-stage semantic reranking with LLM
    """
    
    def __init__(self):
        self.recaller = LLMPaperRecaller()
        self.expander = QueryExpander()
        self.fetcher = PaperFetcher()
        self.scorer = HybridScorer()
        self.reranker = LLMReranker()
    
    async def search(
        self,
        user_query: str,
        llm_provider: str,
        api_key: str,
        limit: int = 20,
        year_range: str = "2024-2026",
        venue_filter: str = "",
        offset: int = 0,
        strict_venue: bool = False,
        enable_rerank: bool = True
    ) -> Dict:
        """
        Perform hybrid search combining LLM recommendations and database search
        
        Args:
            user_query: User's research query
            llm_provider: LLM provider name
            api_key: API key for LLM
            limit: Maximum number of papers to return
            year_range: Year range for search
            venue_filter: Optional venue filter (e.g., "CVPR")
            offset: Pagination offset
            strict_venue: If True, only return exact venue matches
            
        Returns:
            Dict with llm_recommended, db_discovered, merged papers and metadata
        """
        logger.info(f"Starting hybrid search for: {user_query}, venue: {venue_filter}")
        
        # Phase 1: Parallel LLM processing (recall + expand)
        recall_result, expand_result = await self._phase1_llm_processing(
            user_query, llm_provider, api_key, venue_filter, year_range, limit
        )
        
        # Extract results from Phase 1
        llm_titles = self.recaller.extract_titles(recall_result)
        expanded_query = expand_result.get('search_query', user_query)
        
        logger.info(f"LLM recalled {len(llm_titles)} titles")
        logger.info(f"Expanded query: {expanded_query[:100]}...")
        
        # Phase 2: Parallel database search (expand search limit for better coverage)
        search_limit = limit * 5  # Get more papers for filtering
        title_papers, query_papers = await self._phase2_database_search(
            llm_titles, expanded_query, year_range, venue_filter, search_limit, offset
        )
        
        logger.info(f"Title search found {len(title_papers)} papers")
        logger.info(f"Query search found {len(query_papers)} papers")
        
        # Phase 3: Score, filter, and merge results
        result = self._phase3_score_and_merge(
            title_papers, 
            query_papers,
            recall_result,
            user_query,
            venue_filter,
            strict_venue,
            limit * 3  # Get more papers for reranking
        )
        
        # Phase 4: Semantic reranking with LLM (if enabled)
        if enable_rerank and api_key:
            rerank_result = await self._phase4_semantic_rerank(
                result["merged"],
                user_query,
                llm_provider,
                api_key,
                venue_filter,
                limit
            )
            
            # Update result with reranked papers
            if rerank_result.get("reranked_papers"):
                result["merged"] = rerank_result["reranked_papers"]
                result["rerank_reasoning"] = rerank_result.get("reasoning", "")
                result["baseline_papers"] = rerank_result.get("baseline_papers", [])
                result["rerank_enabled"] = True
            else:
                result["rerank_enabled"] = False
                result["rerank_reasoning"] = rerank_result.get("reasoning", "Reranking returned no results")
        else:
            result["rerank_enabled"] = False
            result["rerank_reasoning"] = "Reranking disabled or no API key"
            result["baseline_papers"] = []
            # Limit merged results
            result["merged"] = result["merged"][:limit]
        
        # Add metadata
        result["metadata"] = {
            "llm_recall_count": len(llm_titles),
            "llm_verified_count": result["llm_verified_count"],
            "db_search_count": len(query_papers),
            "venue_matched_count": result["venue_matched_count"],
            "merged_count": len(result["merged"]),
            "venue_filter": venue_filter,
            "expanded_query": expanded_query,
            "rerank_enabled": result.get("rerank_enabled", False)
        }
        result["recall_result"] = recall_result
        result["expand_result"] = expand_result
        
        logger.info(f"Hybrid search complete: {result['metadata']}")
        
        return result
    
    async def _phase4_semantic_rerank(
        self,
        papers: List[Paper],
        user_query: str,
        llm_provider: str,
        api_key: str,
        venue_filter: str,
        limit: int
    ) -> Dict:
        """Phase 4: Semantic reranking using LLM"""
        logger.info(f"Starting semantic reranking of {len(papers)} papers")
        
        # Convert Paper objects to dicts for reranker
        paper_dicts = []
        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "abstract": paper.abstract,
                "venue": paper.venue,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "authors": paper.authors,
                "url": paper.url,
                "source": paper.source,
                "quality_score": getattr(paper, 'quality_score', 0),
                "score_breakdown": getattr(paper, 'score_breakdown', None)
            }
            paper_dicts.append(paper_dict)
        
        try:
            loop = asyncio.get_event_loop()
            rerank_result = await loop.run_in_executor(
                None,
                lambda: self.reranker.rerank(
                    papers=paper_dicts,
                    user_query=user_query,
                    llm_provider=llm_provider,
                    api_key=api_key,
                    venue_filter=venue_filter,
                    top_k=limit
                )
            )
            
            # Convert reranked papers back to Paper objects
            reranked_papers = []
            for paper_dict in rerank_result.get("reranked_papers", []):
                tagged = enrich_and_tag([paper_dict])
                if tagged:
                    paper = tagged[0]
                    # Add reranking info
                    paper.llm_reasoning = paper_dict.get("llm_recommendation_reason", "")
                    paper.source = "llm_reranked"
                    reranked_papers.append(paper)
            
            logger.info(f"Semantic reranking completed: {len(reranked_papers)} papers selected")
            
            return {
                "reranked_papers": reranked_papers,
                "reasoning": rerank_result.get("reasoning", ""),
                "baseline_papers": rerank_result.get("baseline_papers", [])
            }
            
        except Exception as e:
            logger.error(f"Error in semantic reranking: {e}")
            # Fallback: return original papers
            return {
                "reranked_papers": papers[:limit],
                "reasoning": f"Reranking failed: {str(e)}",
                "baseline_papers": []
            }
    
    async def _phase1_llm_processing(
        self, query: str, provider: str, key: str, 
        venue: str, year_range: str, limit: int
    ) -> Tuple[Dict, Dict]:
        """Phase 1: Parallel LLM recall and query expansion"""
        try:
            recall_task = asyncio.create_task(
                self._async_llm_recall(query, provider, key, venue, year_range, limit)
            )
            expand_task = asyncio.create_task(
                self._async_query_expand(query, provider, key)
            )
            
            results = await asyncio.gather(recall_task, expand_task, return_exceptions=True)
            
            recall_result = results[0] if not isinstance(results[0], Exception) else {
                "recommended_papers": [], "overall_reasoning": str(results[0])
            }
            expand_result = results[1] if not isinstance(results[1], Exception) else {
                "search_query": query, "venue_filter": venue
            }
            
            if isinstance(results[0], Exception):
                logger.error(f"LLM recall failed: {results[0]}")
            if isinstance(results[1], Exception):
                logger.error(f"Query expansion failed: {results[1]}")
                
        except Exception as e:
            logger.error(f"Error in Phase 1: {e}")
            recall_result = {"recommended_papers": [], "overall_reasoning": str(e)}
            expand_result = {"search_query": query, "venue_filter": venue}
        
        return recall_result, expand_result
    
    async def _phase2_database_search(
        self, llm_titles: List[str], expanded_query: str,
        year_range: str, venue_filter: str, limit: int, offset: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Phase 2: Optimized database search - skip title search if query returns enough"""
        try:
            # First, run expanded query search (usually fast and returns many results)
            query_papers = await self.fetcher.search_papers_async(
                expanded_query,
                limit=limit,
                year_range=year_range,
                venue_filter="",  # Don't filter by venue in API, do it locally
                offset=offset
            )
            
            # If expanded query returns enough papers (>60), skip title search
            # This saves 10-30 seconds in typical cases
            if len(query_papers) >= 60:
                logger.info(f"Query search returned {len(query_papers)} papers, skipping title search for speed")
                return [], query_papers
            
            # Otherwise, also run title search in parallel
            title_papers = await self._search_by_titles(llm_titles, year_range)
            
            if isinstance(title_papers, Exception):
                logger.error(f"Title search failed: {title_papers}")
                title_papers = []
                
        except Exception as e:
            logger.error(f"Error in Phase 2: {e}")
            title_papers = []
            query_papers = []
        
        return title_papers, query_papers
    
    def _phase3_score_and_merge(
        self,
        title_papers: List[Dict],
        query_papers: List[Dict],
        recall_result: Dict,
        user_query: str,
        venue_filter: str,
        strict_venue: bool,
        limit: int
    ) -> Dict:
        """Phase 3: Score, filter by venue, and merge results"""
        
        # Build LLM title set for matching
        llm_title_set = {
            self._normalize_title(p.get('title', ''))
            for p in recall_result.get('recommended_papers', [])
        }
        
        # Build LLM info mapping
        llm_info_map = {}
        for p in recall_result.get('recommended_papers', []):
            normalized = self._normalize_title(p.get('title', ''))
            if normalized:
                llm_info_map[normalized] = p
        
        # Combine all papers with deduplication
        all_papers = {}
        
        # Process title search results (from LLM recommendations)
        for paper in title_papers:
            normalized = self._normalize_title(paper.get('title', ''))
            if normalized and normalized not in all_papers:
                paper['_from_llm_title_search'] = True
                all_papers[normalized] = paper
        
        # Process query search results
        for paper in query_papers:
            normalized = self._normalize_title(paper.get('title', ''))
            if normalized and normalized not in all_papers:
                paper['_from_llm_title_search'] = False
                all_papers[normalized] = paper
        
        # Score all papers
        papers_list = list(all_papers.values())
        scored_papers = self.scorer.score_papers(papers_list, user_query, venue_filter)
        
        # Filter by venue if specified
        venue_matched = []
        venue_other = []
        
        if venue_filter:
            venue_matched, venue_other = self.scorer.filter_by_venue(
                scored_papers, venue_filter, strict=strict_venue
            )
            logger.info(f"Venue filter '{venue_filter}': {len(venue_matched)} matched, {len(venue_other)} other")
        else:
            venue_other = scored_papers
        
        # Annotate and convert to Paper objects
        llm_recommended_papers = []
        db_discovered_papers = []
        
        # Process venue-matched papers first
        for paper in venue_matched:
            normalized = self._normalize_title(paper.get('title', ''))
            is_llm_rec = normalized in llm_title_set
            llm_info = llm_info_map.get(normalized, {})
            
            # Set source based on whether it was an LLM recommendation
            paper['source'] = 'llm_verified' if is_llm_rec else 'db_discovered'
            paper['verified_in_db'] = True
            paper['llm_confidence'] = llm_info.get('confidence', 0.0) if is_llm_rec else 0.0
            paper['llm_reasoning'] = llm_info.get('reasoning', '') if is_llm_rec else ''
            
            tagged = enrich_and_tag([paper])
            if tagged:
                if is_llm_rec:
                    llm_recommended_papers.extend(tagged)
                else:
                    db_discovered_papers.extend(tagged)
        
        # Process other papers (non-venue-matched)
        for paper in venue_other:
            normalized = self._normalize_title(paper.get('title', ''))
            is_llm_rec = normalized in llm_title_set
            llm_info = llm_info_map.get(normalized, {})
            
            paper['source'] = 'llm_verified' if is_llm_rec else 'db_discovered'
            paper['verified_in_db'] = True
            paper['llm_confidence'] = llm_info.get('confidence', 0.0) if is_llm_rec else 0.0
            paper['llm_reasoning'] = llm_info.get('reasoning', '') if is_llm_rec else ''
            
            tagged = enrich_and_tag([paper])
            if tagged:
                if is_llm_rec:
                    llm_recommended_papers.extend(tagged)
                else:
                    db_discovered_papers.extend(tagged)
        
        # Merge with intelligent sorting
        merged = self._merge_and_sort(
            llm_recommended_papers,
            db_discovered_papers,
            venue_filter,
            limit
        )
        
        return {
            "llm_recommended": llm_recommended_papers,
            "db_discovered": db_discovered_papers,
            "merged": merged,
            "venue_matched": [enrich_and_tag([p])[0] for p in venue_matched if enrich_and_tag([p])],
            "llm_verified_count": len(llm_recommended_papers),
            "venue_matched_count": len(venue_matched)
        }
    
    def _merge_and_sort(
        self,
        llm_papers: List[Paper],
        db_papers: List[Paper],
        venue_filter: str,
        limit: int
    ) -> List[Paper]:
        """Merge and sort papers by quality score"""
        # Combine all unique papers
        merged = {}
        
        for paper in llm_papers + db_papers:
            normalized = self._normalize_title(paper.title)
            if normalized and normalized not in merged:
                merged[normalized] = paper
        
        merged_list = list(merged.values())
        
        # Sort by quality_score (from HybridScorer) if available
        def sort_key(paper: Paper) -> float:
            score = getattr(paper, 'quality_score', 50.0)
            
            # Boost for venue match
            if venue_filter:
                venue_lower = venue_filter.lower()
                if venue_lower in paper.venue.lower():
                    score += 30  # Strong boost for exact venue match
            
            # Boost for verified LLM recommendations
            if paper.source == 'llm_verified' and paper.llm_confidence > 0:
                score += paper.llm_confidence * 10
            
            return score
        
        merged_list.sort(key=sort_key, reverse=True)
        
        return merged_list[:limit]
    
    def search_sync(
        self,
        user_query: str,
        llm_provider: str,
        api_key: str,
        limit: int = 20,
        year_range: str = "2024-2026",
        venue_filter: str = "",
        offset: int = 0,
        strict_venue: bool = False,
        enable_rerank: bool = True
    ) -> Dict:
        """Synchronous wrapper for search()"""
        return asyncio.run(self.search(
            user_query, llm_provider, api_key, limit, year_range, 
            venue_filter, offset, strict_venue, enable_rerank
        ))
    
    async def _async_llm_recall(
        self, query: str, provider: str, key: str, 
        venue: str, year_range: str, limit: int
    ) -> Dict:
        """Async wrapper for LLM recall"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recaller.recall_papers(
                query, provider, key, venue, year_range, min(limit, 10)
            )
        )
    
    async def _async_query_expand(
        self, query: str, provider: str, key: str
    ) -> Dict:
        """Async wrapper for query expansion"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.expander.expand_query(query, provider, key)
        )
    
    async def _search_by_titles(
        self, titles: List[str], year_range: str
    ) -> List[Dict]:
        """Search database by paper titles - PARALLELIZED for speed"""
        if not titles:
            return []
        
        # Limit to only 3 titles to minimize API calls
        titles_to_search = titles[:3]
        
        # Create parallel search tasks for all titles
        async def search_single_title(title: str) -> List[Dict]:
            try:
                papers = await self.fetcher.search_papers_async(
                    title,
                    limit=2,
                    year_range=year_range,
                    venue_filter=""
                )
                return papers
            except Exception as e:
                logger.error(f"Error searching for title '{title[:50]}...': {e}")
                return []
        
        # Execute all title searches in parallel
        tasks = [search_single_title(title) for title in titles_to_search]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all papers from successful searches
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Title search task failed: {result}")
        
        return all_papers
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ""
        normalized = re.sub(r'[\s\-:;,.!?()\[\]{}]+', ' ', title.lower())
        return re.sub(r'\s+', ' ', normalized).strip()


if __name__ == "__main__":
    import asyncio
    
    async def test():
        engine = HybridSearchEngine()
        
        result = await engine.search(
            user_query="Vision-Language-Action models",
            llm_provider="openai",
            api_key="",
            limit=10,
            venue_filter="CVPR"
        )
        
        print(f"LLM Recommended: {len(result['llm_recommended'])}")
        print(f"DB Discovered: {len(result['db_discovered'])}")
        print(f"Venue Matched: {result['venue_matched_count']}")
        print(f"Merged: {len(result['merged'])}")
        print(f"Metadata: {result['metadata']}")
    
    asyncio.run(test())
