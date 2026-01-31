import re
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time
import asyncio
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configurations
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

class PaperFetcher:
    def __init__(self):
        logger.info("Initialized PaperFetcher with arXiv API and OpenAlex API")
    
    async def search_papers_async(self, query: str, limit: int = 20, year_range: str = "2024-2026", venue_filter: str = "", offset: int = 0):
        """
        Search papers using hybrid complementary strategy: arXiv + OpenAlex
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            year_range: Year range in format "YYYY-YYYY"
            venue_filter: Optional venue filter (e.g., "CVPR")
            offset: Offset for pagination
            
        Returns:
            List of dictionaries with paper information
        """
        logger.info(f"Starting hybrid paper search with query: '{query}', limit: {limit}, offset: {offset}, year_range: {year_range}, venue: '{venue_filter}'")
        
        # Parse year range
        try:
            start_year, end_year = map(int, year_range.split("-"))
            logger.info(f"Parsed year range: {start_year}-{end_year}")
        except ValueError:
            start_year, end_year = 2024, 2026
            logger.warning(f"Invalid year range format: {year_range}, using default: 2024-2026")
        
        # First stage: Concurrent retrieval with increased limit for broader coverage
        logger.info("Starting concurrent retrieval from multiple sources")
        
        # Run both searches concurrently with increased limit for better coverage
        retrieval_limit = min(100, (limit + offset) * 5)  # Get more results for pagination
        tasks = [
            self._fetch_arxiv_async(query, limit=retrieval_limit, start_year=start_year, end_year=end_year, venue_filter=venue_filter, offset=offset),
            self._fetch_openalex_async(query, limit=retrieval_limit, start_year=start_year, end_year=end_year, venue_filter=venue_filter, offset=offset)
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        arxiv_papers = []
        openalex_papers = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if i == 0:
                    logger.error(f"Error fetching from arXiv: {result}")
                else:
                    logger.error(f"Error fetching from OpenAlex: {result}")
            else:
                if i == 0:
                    arxiv_papers = result
                    logger.info(f"arXiv returned {len(arxiv_papers)} papers")
                else:
                    openalex_papers = result
                    logger.info(f"OpenAlex returned {len(openalex_papers)} papers")
        
        # Second stage: Intelligent merge and deduplication
        logger.info("Starting intelligent merge and deduplication")
        merged_papers = self._smart_merge(arxiv_papers, openalex_papers)
        
        # Apply venue filter if specified
        if venue_filter:
            filtered_papers = []
            for paper in merged_papers:
                venue = paper.get("venue", "").lower()
                if venue_filter.lower() in venue:
                    filtered_papers.append(paper)
            merged_papers = filtered_papers
            logger.info(f"Applied venue filter '{venue_filter}', remaining papers: {len(merged_papers)}")
        
        # Limit to requested number
        final_papers = merged_papers[:limit]
        
        logger.info(f"Final merged result: {len(final_papers)} unique papers")
        return final_papers
    
    def search_papers(self, query: str, limit: int = 20, year_range: str = "2024-2026", venue_filter: str = "", offset: int = 0):
        """
        Synchronous wrapper for search_papers_async
        """
        return asyncio.run(self.search_papers_async(query, limit, year_range, venue_filter, offset))
    
    async def _fetch_arxiv_async(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0):
        """
        Async wrapper for arXiv search
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_arxiv(query, limit, start_year, end_year, venue_filter, offset)
        )
    
    async def _fetch_openalex_async(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0):
        """
        Async wrapper for OpenAlex search
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_openalex(query, limit, start_year, end_year, venue_filter, offset)
        )
    
    def _search_arxiv(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0):
        """
        Search papers using arXiv API
        """
        logger.info(f"Calling arXiv API with query: '{query}', venue filter: '{venue_filter}', offset: {offset}")
        
        # Build search query
        search_query = f'all:{query}'
        params = {
            "search_query": search_query,
            "start": offset,
            "max_results": limit * 3,  # Get more results to filter
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        logger.info(f"arXiv API URL: {url}")
        
        try:
            # Add user agent to avoid being blocked
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ScholarAgent/1.0 (research@example.com)"}
            )
            
            # Add timeout
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_text = resp.read()
            
            root = ET.fromstring(xml_text)
            papers = []
            
            for entry in root.findall("atom:entry", ATOM_NS):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    # Filter by year
                    if paper.get("year") and start_year <= paper["year"] <= end_year:
                        # Apply venue filter if specified
                        if not venue_filter or venue_filter.lower() in paper.get("venue", "").lower():
                            papers.append(paper)
            
            # Limit to requested number
            return papers[:limit]
            
        except Exception as e:
            logger.error(f"Error in arXiv search: {type(e).__name__}: {str(e)}")
            return []
    
    def _parse_arxiv_entry(self, entry):
        """
        Parse arXiv API entry into paper dictionary
        """
        try:
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
            abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
            
            # Extract year from published date
            year = 0
            if published:
                try:
                    year = int(published.split("-")[0])
                except ValueError:
                    pass
            
            # Extract authors
            authors = []
            for a in entry.findall("atom:author", ATOM_NS):
                name = a.findtext("atom:name", default="", namespaces=ATOM_NS)
                if name:
                    authors.append(name)
            
            # Extract arXiv ID
            arxiv_id = ""
            if abs_url:
                arxiv_id = abs_url.split("/")[-1]
            
            # Create paper dictionary
            paper = {
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": year,
                "citationCount": 0,  # arXiv API doesn't provide citation count
                "venue": "arXiv",
                "url": abs_url,
                "pdf_url": abs_url.replace("abs", "pdf"),
                "paperId": arxiv_id,
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "status": "Preprint"
            }
            
            return paper
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def _search_openalex(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0):
        """
        Search papers using OpenAlex API
        """
        logger.info(f"Calling OpenAlex API with query: '{query}', venue filter: '{venue_filter}', offset: {offset}")
        
        try:
            # Build OpenAlex API query with correct syntax
            # Use correct OpenAlex filter format for range values
            filters = [f"publication_year:{start_year}-{end_year}"]
            
            # Add venue filter if specified
            if venue_filter:
                filters.append(f"venue:{venue_filter}")
            
            params = {
                "search": query,
                "filter": ",".join(filters),
                "per-page": min(limit * 3, 200),  # Get more results but not exceed OpenAlex limit
                "page": offset // limit + 1,  # Calculate page number from offset
                "mailto": "research@example.com"
            }
            
            # Construct URL with proper encoding
            url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
            logger.info(f"OpenAlex API URL: {url}")
            
            # Add user agent header
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "ScholarAgent/1.0"
                }
            )
            
            # Add timeout
            with urllib.request.urlopen(req, timeout=15) as resp:
                # Check response status
                if resp.status != 200:
                    logger.error(f"OpenAlex returned non-200 status: {resp.status}")
                    return []
                
                # Read and parse response
                try:
                    response_text = resp.read().decode('utf-8')
                    data = json.loads(response_text)
                except Exception as e:
                    logger.error(f"Error reading/parsing OpenAlex response: {e}")
                    return []
            
            # Format results
            papers = []
            if data and isinstance(data, dict):
                results = data.get("results", [])
                if isinstance(results, list):
                    for item in results:
                        if item and isinstance(item, dict):
                            # Extract institutions from authors
                            institutions = []
                            for auth in item.get("authorships", []):
                                if auth and isinstance(auth, dict) and auth.get("institutions"):
                                    for inst in auth.get("institutions", []):
                                        if inst and isinstance(inst, dict) and inst.get("display_name"):
                                            institutions.append(inst.get("display_name"))
                            
                            # Handle nested fields with None checks
                            primary_location = item.get("primary_location")
                            venue = ""
                            if primary_location and isinstance(primary_location, dict):
                                source = primary_location.get("source")
                                if source and isinstance(source, dict):
                                    venue = source.get("display_name", "")
                            
                            # Apply venue filter if specified
                            if venue_filter:
                                if venue_filter.lower() not in venue.lower():
                                    continue
                            
                            open_access = item.get("open_access")
                            url = item.get("url", "")
                            if open_access and isinstance(open_access, dict):
                                oa_url = open_access.get("oa_url")
                                if oa_url:
                                    url = oa_url
                            
                            paper_info = {
                                "title": item.get("title", ""),
                                "abstract": item.get("abstract", ""),
                                "authors": [auth.get("author", {}).get("display_name", "") for auth in item.get("authorships", []) if auth and isinstance(auth, dict) and auth.get("author")],
                                "year": item.get("publication_year", 0),
                                "citationCount": item.get("cited_by_count", 0),
                                "venue": venue,
                                "url": url,
                                "paperId": item.get("id", ""),
                                "source": "openalex",
                                "institutions": list(set(institutions)),  # Remove duplicates
                                "status": "Published"
                            }
                            papers.append(paper_info)
            
            return papers[:limit]  # Limit to requested number
            
        except urllib.error.HTTPError as e:
            # Print detailed error information
            error_info = f"HTTP Error {e.code}: {e.reason}"
            try:
                # Read error response body
                error_body = e.read().decode('utf-8')
                error_info += f"\nError body: {error_body}"
            except:
                pass
            logger.error(f"Error in OpenAlex search: {error_info}")
            return []
        except Exception as e:
            logger.error(f"Error in OpenAlex search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _smart_merge(self, arxiv_papers, openalex_papers):
        """
        Intelligent merge and deduplication of papers from different sources
        """
        # Create a dictionary to hold merged papers by normalized title
        merged = {}
        
        # Process arXiv papers first
        for paper in arxiv_papers:
            normalized_title = self._normalize_title(paper.get("title", ""))
            if normalized_title:
                merged[normalized_title] = paper
        
        # Process OpenAlex papers and merge with existing ones
        for paper in openalex_papers:
            normalized_title = self._normalize_title(paper.get("title", ""))
            if normalized_title:
                if normalized_title in merged:
                    # Merge metadata
                    existing_paper = merged[normalized_title]
                    # Use OpenAlex for citation count, venue, institutions
                    existing_paper["citationCount"] = paper.get("citationCount", existing_paper.get("citationCount", 0))
                    existing_paper["venue"] = paper.get("venue", existing_paper.get("venue", ""))
                    existing_paper["institutions"] = paper.get("institutions", existing_paper.get("institutions", []))
                    existing_paper["status"] = "Published"  # If in both, mark as published
                else:
                    # Add new paper from OpenAlex
                    merged[normalized_title] = paper
        
        # Convert to list and sort by citation count (descending)
        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        
        return merged_list
    
    def _normalize_title(self, title):
        """
        Normalize title for deduplication
        """
        if not title:
            return ""
        
        # Remove punctuation
        normalized = re.sub(r'[\s\-:;,.!?()\[\]{}]+', ' ', title.lower())
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

if __name__ == "__main__":
    # Test the PaperFetcher
    fetcher = PaperFetcher()
    papers = fetcher.search_papers("Mamba architecture for vision", limit=5)
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Citations: {paper['citationCount']}")
        print(f"   Venue: {paper['venue']}")
        print(f"   URL: {paper['url']}")
        print(f"   Status: {paper['status']}")
        if 'pdf_url' in paper:
            print(f"   PDF URL: {paper['pdf_url']}")
        if 'institutions' in paper and paper['institutions']:
            print(f"   Institutions: {', '.join(paper['institutions'])}")
        print()