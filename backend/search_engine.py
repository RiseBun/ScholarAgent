import re
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time
import asyncio
import json
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configurations
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
DBLP_API = "https://dblp.org/search/publ/api"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

class PaperFetcher:
    # Class-level arXiv failure tracking
    _arxiv_consecutive_failures = 0
    _arxiv_disabled_until = 0  # Unix timestamp
    
    def __init__(self):
        logger.info("Initialized PaperFetcher with DBLP API, arXiv API, and OpenAlex API")
        # Track if venue filter was dropped due to no results
        self.venue_filter_dropped = False
        self.original_venue_filter = ""
    
    @classmethod
    def _should_skip_arxiv(cls) -> bool:
        """Check if arXiv should be skipped due to rate limiting"""
        if cls._arxiv_disabled_until > time.time():
            logger.info(f"arXiv disabled for {int(cls._arxiv_disabled_until - time.time())}s due to rate limiting")
            return True
        # Skip after just 1 failure (more aggressive for speed)
        if cls._arxiv_consecutive_failures >= 1:
            # Disable for 120 seconds after any failure
            cls._arxiv_disabled_until = time.time() + 120
            logger.warning("arXiv disabled for 120s due to failure - using DBLP/OpenAlex only")
            return True
        return False
    
    @classmethod
    def _record_arxiv_failure(cls):
        """Record an arXiv failure"""
        cls._arxiv_consecutive_failures += 1
    
    @classmethod
    def _record_arxiv_success(cls):
        """Record an arXiv success"""
        cls._arxiv_consecutive_failures = 0
    
    async def search_papers_async(self, query: str, limit: int = 20, year_range: str = "2024-2026", venue_filter: str = "", offset: int = 0, serpapi_key: str = ""):
        """
        Search papers using hybrid complementary strategy: DBLP + arXiv + OpenAlex
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            year_range: Year range in format "YYYY-YYYY"
            venue_filter: Optional venue filter (e.g., "CVPR")
            offset: Offset for pagination
            serpapi_key: Optional SerpAPI API key for Google Scholar
            
        Returns:
            List of dictionaries with paper information
        """
        logger.info(f"Starting hybrid paper search with query: '{query}', limit: {limit}, offset: {offset}, year_range: {year_range}, venue: '{venue_filter}'")
        
        # Reset venue filter tracking
        self.venue_filter_dropped = False
        self.original_venue_filter = venue_filter
        
        # Parse year range
        try:
            start_year, end_year = map(int, year_range.split("-"))
            logger.info(f"Parsed year range: {start_year}-{end_year}")
        except ValueError:
            start_year, end_year = 2024, 2026
            logger.warning(f"Invalid year range format: {year_range}, using default: 2024-2026")
        
        # First stage: Concurrent retrieval with consistent limit for pagination
        logger.info("Starting concurrent retrieval from multiple sources")
        
        # Run searches with分层策略
        retrieval_limit = min(200, limit * 3)  # Get more results but maintain consistency
        tasks = []
        
        # 根据是否指定了会议，采用不同的搜索策略
        if venue_filter:
            # 场景 1：用户指定了会议
            # 1. 首先使用 DBLP 搜索会议论文（Venue 准确度最高）
            tasks.append(
                self._fetch_dblp_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
            # 2. 然后使用 arXiv 搜索最新预印本
            tasks.append(
                self._fetch_arxiv_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
            # 3. 最后使用 OpenAlex 补充引用信息
            tasks.append(
                self._fetch_openalex_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
        else:
            # 场景 2：用户未指定会议
            # 1. 首先使用 arXiv 搜索最新预印本
            tasks.append(
                self._fetch_arxiv_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
            # 2. 然后使用 OpenAlex 搜索已发表论文和引用信息
            tasks.append(
                self._fetch_openalex_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
            # 3. 最后使用 DBLP 补充 venue 信息
            tasks.append(
                self._fetch_dblp_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset
                )
            )
        
        # 添加 Google Scholar 搜索（如果 API key 提供）
        if serpapi_key:
            tasks.append(
                self._fetch_google_scholar_async(
                    query, 
                    limit=retrieval_limit, 
                    start_year=start_year, 
                    end_year=end_year, 
                    venue_filter=venue_filter, 
                    offset=offset, 
                    serpapi_key=serpapi_key
                )
            )
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dblp_papers = []
        arxiv_papers = []
        openalex_papers = []
        google_scholar_papers = []
        
        # Determine which results correspond to which data sources
        has_google_scholar = bool(serpapi_key)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if venue_filter:
                    # 场景 1：用户指定了会议
                    if i == 0:
                        logger.error(f"Error fetching from DBLP: {result}")
                    elif i == 1:
                        logger.error(f"Error fetching from arXiv: {result}")
                    elif i == 2:
                        logger.error(f"Error fetching from OpenAlex: {result}")
                    elif i == 3 and has_google_scholar:
                        logger.error(f"Error fetching from Google Scholar: {result}")
                else:
                    # 场景 2：用户未指定会议
                    if i == 0:
                        logger.error(f"Error fetching from arXiv: {result}")
                    elif i == 1:
                        logger.error(f"Error fetching from OpenAlex: {result}")
                    elif i == 2:
                        logger.error(f"Error fetching from DBLP: {result}")
                    elif i == 3 and has_google_scholar:
                        logger.error(f"Error fetching from Google Scholar: {result}")
            else:
                if venue_filter:
                    # 场景 1：用户指定了会议
                    if i == 0:
                        dblp_papers = result
                        logger.info(f"DBLP returned {len(dblp_papers)} papers")
                    elif i == 1:
                        arxiv_papers = result
                        logger.info(f"arXiv returned {len(arxiv_papers)} papers")
                    elif i == 2:
                        openalex_papers = result
                        logger.info(f"OpenAlex returned {len(openalex_papers)} papers")
                    elif i == 3 and has_google_scholar:
                        google_scholar_papers = result
                        logger.info(f"Google Scholar returned {len(google_scholar_papers)} papers")
                else:
                    # 场景 2：用户未指定会议
                    if i == 0:
                        arxiv_papers = result
                        logger.info(f"arXiv returned {len(arxiv_papers)} papers")
                    elif i == 1:
                        openalex_papers = result
                        logger.info(f"OpenAlex returned {len(openalex_papers)} papers")
                    elif i == 2:
                        dblp_papers = result
                        logger.info(f"DBLP returned {len(dblp_papers)} papers")
                    elif i == 3 and has_google_scholar:
                        google_scholar_papers = result
                        logger.info(f"Google Scholar returned {len(google_scholar_papers)} papers")
        
        # Second stage: Intelligent merge and deduplication
        logger.info("Starting intelligent merge and deduplication")
        merged_papers = self._smart_merge(arxiv_papers, openalex_papers, query, google_scholar_papers)
        
        # 处理 DBLP 论文（最高优先级）
        if dblp_papers:
            # 创建一个字典来保存已有的论文
            existing_papers = {}
            for paper in merged_papers:
                normalized_title = self._normalize_title(paper.get("title", ""))
                if normalized_title:
                    existing_papers[normalized_title] = paper
            
            # 添加或更新 DBLP 论文
            for dblp_paper in dblp_papers:
                normalized_title = self._normalize_title(dblp_paper.get("title", ""))
                if normalized_title:
                    if normalized_title in existing_papers:
                        # 更新现有论文，使用 DBLP 的 venue 信息（最高优先级）
                        existing_paper = existing_papers[normalized_title]
                        existing_paper["venue"] = dblp_paper.get("venue", existing_paper.get("venue", ""))
                        existing_paper["status"] = "Published"
                    else:
                        # 添加新的 DBLP 论文
                        existing_papers[normalized_title] = dblp_paper
            
            # 转换回列表并排序
            merged_papers = list(existing_papers.values())
            
            # 重新计算得分并排序
            scored_papers = []
            for paper in merged_papers:
                score = self.calculate_paper_score(paper, query)
                scored_papers.append((score, paper))
            
            # 按得分降序排序
            scored_papers.sort(key=lambda x: x[0], reverse=True)
            
            # 提取排序后的论文
            merged_papers = [paper for _, paper in scored_papers]
        
        # Apply venue filter if specified (more flexible matching)
        if venue_filter:
            filtered_papers = []
            for paper in merged_papers:
                venue = paper.get("venue", "").lower()
                filter_lower = venue_filter.lower()
                
                # Check if paper matches venue filter
                matches_venue = False
                
                # For all papers, check venue field first
                if (filter_lower in venue or
                   any(abbrev in venue for abbrev in self._get_venue_abbreviations(filter_lower))):
                    matches_venue = True
                
                # For arXiv papers, use more flexible filtering considering preprint nature
                if paper.get("source") == "arxiv" and not matches_venue:
                    # Check title, abstract, and comments for venue information
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()
                    comments = paper.get("comments", "").lower()
                    
                    # Check if any of these fields contain venue information
                    venue_abbreviations = self._get_venue_abbreviations(filter_lower)
                    all_text = f"{title} {abstract} {comments}"
                    
                    # More flexible matching for arXiv papers
                    # Check for venue name or abbreviation in any field
                    if any(abbrev in all_text for abbrev in venue_abbreviations):
                        matches_venue = True
                    # Check for common patterns indicating conference acceptance
                    elif any(pattern in all_text for pattern in [
                        f"to appear in {filter_lower}",
                        f"accepted to {filter_lower}",
                        f"submitted to {filter_lower}",
                        f"presented at {filter_lower}"
                    ]):
                        matches_venue = True
                    # For CVPR specifically, check for common variations
                    elif filter_lower == "cvpr" and any(variation in all_text for variation in [
                        "computer vision and pattern recognition",
                        "cvpr"
                    ]):
                        matches_venue = True
                
                if matches_venue:
                    filtered_papers.append(paper)
            
            merged_papers = filtered_papers
            logger.info(f"Applied venue filter '{venue_filter}', remaining papers: {len(merged_papers)}")
        
        # If no papers after venue filter, try without filter
        if not merged_papers and venue_filter:
            logger.info("No papers passed venue filter, trying without filter...")
            self.venue_filter_dropped = True
            merged_papers = self._smart_merge(arxiv_papers, openalex_papers, query)
            logger.info(f"Found {len(merged_papers)} papers without venue filter")
        
        # Limit to requested number
        final_papers = merged_papers[:limit]
        
        logger.info(f"Final merged result: {len(final_papers)} unique papers")
        return final_papers
    
    def search_papers(self, query: str, limit: int = 20, year_range: str = "2024-2026", venue_filter: str = "", offset: int = 0, serpapi_key: str = ""):
        """
        Synchronous wrapper for search_papers_async
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            year_range: Year range in format "YYYY-YYYY"
            venue_filter: Optional venue filter (e.g., "CVPR")
            offset: Offset for pagination
            serpapi_key: Optional SerpAPI API key for Google Scholar
            
        Returns:
            List of dictionaries with paper information
        """
        return asyncio.run(self.search_papers_async(query, limit, year_range, venue_filter, offset, serpapi_key))
    
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
        Search papers using arXiv API with rate limiting and retry logic
        """
        # Check if arXiv should be skipped due to rate limiting
        if self._should_skip_arxiv():
            logger.info("Skipping arXiv due to rate limiting")
            return []
        
        logger.info(f"Calling arXiv API with query: '{query}', venue filter: '{venue_filter}', offset: {offset}")
        
        # Build search query
        search_query = f'all:{query}'
        params = {
            "search_query": search_query,
            "start": offset,
            "max_results": min(limit * 3, 50),  # Cap at 50 to reduce load
            "sortBy": "relevance",
            "sortOrder": "descending",
            "ts": int(time.time())
        }
        
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        logger.info(f"arXiv API URL: {url}")
        
        # Single attempt only - no retries for speed
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ScholarAgent/1.0 (research@example.com)"}
            )
            
            # Reduced timeout to 10 seconds for speed
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_text = resp.read()
            
            root = ET.fromstring(xml_text)
            papers = []
            
            for entry in root.findall("atom:entry", ATOM_NS):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    if paper.get("year"):
                        try:
                            paper_year = int(paper["year"])
                            if start_year <= paper_year <= end_year:
                                papers.append(paper)
                        except (ValueError, TypeError):
                            pass
            
            # Record success
            self._record_arxiv_success()
            return papers[:limit]
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"arXiv rate limited (429), skipping")
            else:
                logger.error(f"arXiv HTTP error: {e.code}")
            self._record_arxiv_failure()
            return []
        except Exception as e:
            logger.error(f"arXiv error: {type(e).__name__}: {str(e)}")
            self._record_arxiv_failure()
            return []
    
    def _parse_arxiv_entry(self, entry):
        """
        Parse arXiv API entry into paper dictionary
        """
        try:
            import re
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
            abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
            comments = (entry.findtext("arxiv:comment", default="", namespaces={"arxiv": "http://arxiv.org/schemas/atom"}) or "").strip()
            
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
            
            # Extract venue information from comments
            venue = "arXiv"
            if comments:
                # List of common conferences to look for
                conferences = [
                    "iclr", "cvpr", "iccv", "eccv", "neurips", "icml", "aaai", 
                    "ijcai", "acl", "emnlp", "naacl", "icra", "iros", "rss", "corl"
                ]
                
                # Check if any conference name appears in comments
                comments_lower = comments.lower()
                for conf in conferences:
                    if conf in comments_lower:
                        # Extract the full conference information
                        # Look for patterns like "ICLR 2024" or "ICLR'24"
                        # Pattern to match conference name and year
                        pattern = r'(' + conf + r')\s*(?:20)?(\d{2,4})?'
                        match = re.search(pattern, comments_lower, re.IGNORECASE)
                        if match:
                            # Capitalize conference name
                            conf_name = match.group(1).upper()
                            conf_year = match.group(2)
                            if conf_year:
                                if len(conf_year) == 2:
                                    conf_year = "20" + conf_year
                                venue = f"{conf_name} {conf_year}"
                                # Update year if valid
                                try:
                                    year = int(conf_year)
                                except ValueError:
                                    pass
                            else:
                                venue = conf_name.upper()
                        else:
                            # If no year found, just use conference name
                            venue = conf.upper()
                        break
            
            # Also check for CVPR specifically in title or abstract if not found in comments
            # Some papers might mention CVPR in their title or abstract
            if venue == "arXiv":
                # Check title and abstract for CVPR mentions
                title_lower = title.lower()
                summary_lower = summary.lower()
                
                # Look for CVPR patterns in title or abstract
                cvpr_patterns = ["cvpr", "computer vision and pattern recognition"]
                for pattern in cvpr_patterns:
                    if pattern in title_lower or pattern in summary_lower:
                        # Try to extract year if possible
                        year_pattern = r'(?:20|19)\d{2}'
                        year_match = re.search(year_pattern, title_lower + " " + summary_lower)
                        if year_match:
                            cvpr_year = year_match.group()
                            venue = f"CVPR {cvpr_year}"
                            # Update year if valid
                            try:
                                year = int(cvpr_year)
                            except ValueError:
                                pass
                        else:
                            venue = "CVPR"
                        break
            
            # Create paper dictionary
            paper = {
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": year,
                "citationCount": 0,  # arXiv API doesn't provide citation count
                "venue": venue,
                "url": abs_url,
                "pdf_url": abs_url.replace("abs", "pdf"),
                "paperId": arxiv_id,
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "status": "Preprint",
                "comments": comments
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
            
            # Removed venue filter as OpenAlex API doesn't support it
            # Will filter by venue in Python instead
            
            # Simplify query for OpenAlex - it doesn't support complex boolean syntax
            # Extract key terms by removing OR/AND operators and parentheses
            simplified_query = self._simplify_query_for_openalex(query)
            logger.info(f"Simplified query for OpenAlex: '{simplified_query}'")
            
            params = {
                "search": simplified_query,
                "filter": ",".join(filters),
                "per-page": min(limit * 3, 200),  # Get more results but not exceed OpenAlex limit
                "page": offset // limit + 1,  # Calculate page number from offset
                "mailto": "research@example.com"
                # Removed ts parameter as OpenAlex doesn't support it
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
                            
                            # Apply venue filter if specified (more flexible matching)
                            if venue_filter:
                                venue_lower = venue.lower()
                                filter_lower = venue_filter.lower()
                                # Check if filter matches venue name or common abbreviations
                                if not (filter_lower in venue_lower or
                                       any(abbrev in venue_lower for abbrev in self._get_venue_abbreviations(filter_lower))):
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
    
    def _search_google_scholar(self, query, limit=10, start_year=2024, end_year=2026, venue_filter="", offset=0, serpapi_key=""):
        """
        Search papers using Google Scholar via SerpAPI
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            start_year: Start year for filtering
            end_year: End year for filtering
            venue_filter: Optional venue filter (e.g., "ICLR")
            offset: Offset for pagination
            serpapi_key: SerpAPI API key
            
        Returns:
            List of dictionaries with paper information
        """
        logger.info(f"Calling Google Scholar via SerpAPI with query: '{query}', venue filter: '{venue_filter}', offset: {offset}")
        
        try:
            # Build search query with venue filter if specified
            search_query = query
            if venue_filter:
                # Add venue filter to query
                search_query = f"{query} {venue_filter}"
                # For ICLR, add site:openreview.net to get more accurate results
                if venue_filter.lower() == "iclr":
                    search_query += " site:openreview.net"
            
            # Build SerpAPI parameters
            params = {
                "engine": "google_scholar",
                "q": search_query,
                "as_ylo": start_year,
                "as_yhi": end_year,
                "num": min(limit * 2, 20),  # Get more results to filter
                "api_key": serpapi_key
            }
            
            logger.info(f"Google Scholar search query: '{search_query}'")
            
            # Call SerpAPI
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=30
            )
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"SerpAPI returned non-200 status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
            
            # Parse response
            data = response.json()
            
            # Check for errors in response
            if "error" in data:
                logger.error(f"SerpAPI error: {data['error']}")
                return []
            
            # Process results
            papers = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results:
                paper = self._parse_google_scholar_entry(result)
                if paper:
                    # Filter by year
                    if paper.get("year"):
                        try:
                            paper_year = int(paper["year"])
                            if start_year <= paper_year <= end_year:
                                # Filter by venue if specified
                                if venue_filter:
                                    venue = paper.get("venue", "").lower()
                                    filter_lower = venue_filter.lower()
                                    if (filter_lower in venue or
                                       any(abbrev in venue for abbrev in self._get_venue_abbreviations(filter_lower))):
                                        papers.append(paper)
                                else:
                                    papers.append(paper)
                        except (ValueError, TypeError):
                            # Skip papers with invalid year
                            pass
            
            # Limit to requested number
            return papers[:limit]
            
        except Exception as e:
            logger.error(f"Error in Google Scholar search: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _parse_google_scholar_entry(self, entry):
        """
        Parse Google Scholar entry into paper dictionary
        
        Args:
            entry: Google Scholar entry from SerpAPI
            
        Returns:
            Dictionary with paper information or None if parsing fails
        """
        try:
            # Extract basic information
            title = entry.get("title", "")
            link = entry.get("link", "")
            snippet = entry.get("snippet", "")
            
            # Extract authors
            authors = []
            authors_info = entry.get("publication_info", {}).get("authors", [])
            for author in authors_info:
                author_name = author.get("name", "")
                if author_name:
                    authors.append(author_name)
            
            # Extract venue and year
            venue = ""
            year = 0
            publication_info = entry.get("publication_info", {})
            
            # Extract venue from publication_info
            venue_info = publication_info.get("summary", "")
            if venue_info:
                # Try to extract year from venue info
                year_match = re.search(r'(?:20|19)\d{2}', venue_info)
                if year_match:
                    try:
                        year = int(year_match.group())
                    except ValueError:
                        pass
                
                # Extract venue name
                venue = venue_info
                if year_match:
                    venue = venue.replace(year_match.group(), "").strip()
                
                # Clean up venue string
                venue = re.sub(r'\[PDF\]\s*', '', venue)
                venue = re.sub(r'\[HTML\]\s*', '', venue)
                venue = venue.strip()
            
            # Extract citation count
            citation_count = 0
            inline_links = entry.get("inline_links", {})
            citations = inline_links.get("cited_by", {})
            if citations:
                citation_count = citations.get("count", 0)
            
            # Create paper dictionary
            paper = {
                "title": title,
                "abstract": snippet,
                "authors": authors,
                "year": year,
                "citationCount": citation_count,
                "venue": venue,
                "url": link,
                "source": "google_scholar",
                "status": "Published"
            }
            
            return paper
        except Exception as e:
            logger.error(f"Error parsing Google Scholar entry: {e}")
            return None
    
    async def _fetch_google_scholar_async(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0, serpapi_key: str = ""):
        """
        Async wrapper for Google Scholar search
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_google_scholar(query, limit, start_year, end_year, venue_filter, offset, serpapi_key)
        )
    
    def _search_dblp(self, query, limit=10, start_year=2024, end_year=2026, venue_filter="", offset=0):
        """
        Search papers using DBLP API
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            start_year: Start year for filtering
            end_year: End year for filtering
            venue_filter: Optional venue filter (e.g., "CVPR")
            offset: Offset for pagination
            
        Returns:
            List of dictionaries with paper information
        """
        logger.info(f"Calling DBLP API with query: '{query}', venue filter: '{venue_filter}', offset: {offset}")
        
        try:
            # Try different query variations to improve recall
            query_variations = self._generate_query_variations(query, venue_filter)
            
            for i, (simplified_query, search_query) in enumerate(query_variations):
                logger.info(f"DBLP query variation {i+1}: '{search_query}'")
                
                # Build DBLP API parameters
                params = {
                    "q": search_query,
                    "format": "json",
                    "h": min(limit * 2, 100)  # Get more results to filter
                }
                
                try:
                    # Call DBLP API
                    response = requests.get(
                        DBLP_API,
                        params=params,
                        timeout=30
                    )
                    
                    # Check response status
                    if response.status_code != 200:
                        logger.error(f"DBLP API returned non-200 status: {response.status_code}")
                        logger.error(f"Response: {response.text}")
                        continue
                    
                    # Parse response
                    data = response.json()
                    
                    # Check for hits in response
                    if "result" not in data or "hits" not in data["result"]:
                        logger.warning("DBLP API returned no hits")
                        continue
                    
                    # Process results
                    papers = []
                    hits = data["result"]["hits"].get("hit", [])
                    
                    for hit in hits:
                        paper = self._parse_dblp_entry(hit)
                        if paper:
                            # Filter by year
                            if paper.get("year"):
                                try:
                                    paper_year = int(paper["year"])
                                    if start_year <= paper_year <= end_year:
                                        # Filter by venue if specified
                                        if venue_filter:
                                            venue = paper.get("venue", "").lower()
                                            filter_lower = venue_filter.lower()
                                            # Check if filter matches venue name or common abbreviations
                                            if (filter_lower in venue or
                                               any(abbrev in venue for abbrev in self._get_venue_abbreviations(filter_lower))):
                                                papers.append(paper)
                                        else:
                                            papers.append(paper)
                                except (ValueError, TypeError):
                                    # Skip papers with invalid year
                                    pass
                    
                    # If we found papers, return them
                    if papers:
                        logger.info(f"DBLP returned {len(papers)} papers with query variation {i+1}")
                        return papers[:limit]
                        
                except Exception as e:
                    logger.error(f"Error in DBLP search variation {i+1}: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue to next variation
                    continue
            
            # If no papers found with any variation
            logger.warning("DBLP API returned no hits with any query variation")
            return []
            
        except Exception as e:
            logger.error(f"Error in DBLP search: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _generate_query_variations(self, query, venue_filter=""):
        """
        Generate multiple query variations for DBLP API to improve recall
        
        Args:
            query: Original search query
            venue_filter: Optional venue filter
            
        Returns:
            List of tuples (simplified_query, search_query)
        """
        variations = []
        
        # Variation 1: Original simplified query
        simplified_query = self._simplify_query_for_dblp(query)
        search_query = simplified_query
        if venue_filter:
            search_query = f"{simplified_query} {venue_filter}"
        variations.append((simplified_query, search_query))
        
        # Variation 2: Just the venue filter (if specified)
        if venue_filter:
            variations.append((venue_filter, venue_filter))
        
        # Variation 3: Core terms only
        core_terms = self._extract_core_terms(query)
        if core_terms:
            core_query = " ".join(core_terms[:3])
            search_query = core_query
            if venue_filter:
                search_query = f"{core_query} {venue_filter}"
            variations.append((core_query, search_query))
        
        # Variation 4: Common abbreviations
        abbreviated_query = self._abbreviate_query(query)
        if abbreviated_query != simplified_query:
            search_query = abbreviated_query
            if venue_filter:
                search_query = f"{abbreviated_query} {venue_filter}"
            variations.append((abbreviated_query, search_query))
        
        # Limit to 5 variations
        return variations[:5]
    
    def _extract_core_terms(self, query):
        """
        Extract core terms from query
        
        Args:
            query: Original search query
            
        Returns:
            List of core terms
        """
        import re
        
        # Remove parentheses and operators
        simplified = re.sub(r'[()]', ' ', query)
        simplified = re.sub(r'\s+OR\s+', ' ', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s+AND\s+', ' ', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        # Extract terms
        terms = simplified.split()
        
        # Filter out stop words and short terms
        stop_words = {'or', 'and', 'the', 'a', 'an', 'for', 'of', 'in', 'on', 'to', 'with', 'using', 'based'}
        core_terms = [t for t in terms if len(t) > 2 and t.lower() not in stop_words]
        
        return core_terms
    
    def _abbreviate_query(self, query):
        """
        Abbreviate query by replacing common phrases with abbreviations
        
        Args:
            query: Original search query
            
        Returns:
            Abbreviated query
        """
        abbreviations = {
            'Vision-Language-Action': 'VLA',
            'Vision Language Action': 'VLA',
            'Large Language Model': 'LLM',
            'Machine Learning': 'ML',
            'Deep Learning': 'DL',
            'Computer Vision': 'CV',
            'Natural Language Processing': 'NLP',
            'Reinforcement Learning': 'RL',
            'Autonomous Driving': 'AD',
            'Robotics': 'Robotics'
        }
        
        abbreviated = query
        for phrase, abbr in abbreviations.items():
            abbreviated = abbreviated.replace(phrase, abbr)
        
        return abbreviated
    
    def _parse_dblp_entry(self, entry):
        """
        Parse DBLP entry into paper dictionary
        
        Args:
            entry: DBLP entry from API response
            
        Returns:
            Dictionary with paper information or None if parsing fails
        """
        try:
            # Extract info from entry
            info = entry.get("info", {})
            
            # Extract basic information
            title = info.get("title", "")
            raw_authors = info.get("authors", {}).get("author", [])
            
            # Handle different author formats from DBLP
            authors = []
            if isinstance(raw_authors, str):
                # Single author as string
                authors = [raw_authors]
            elif isinstance(raw_authors, list):
                for author in raw_authors:
                    if isinstance(author, str):
                        authors.append(author)
                    elif isinstance(author, dict):
                        # Handle {'@pid': '...', 'text': 'Name'} format
                        name = author.get('text', author.get('name', ''))
                        if name:
                            authors.append(name)
            elif isinstance(raw_authors, dict):
                # Single author as dict
                name = raw_authors.get('text', raw_authors.get('name', ''))
                if name:
                    authors = [name]
            
            # Extract venue and year
            venue = info.get("venue", "")
            year = info.get("year", 0)
            if year:
                try:
                    year = int(year)
                except ValueError:
                    year = 0
            
            # Extract link
            url = info.get("ee", "")
            
            # Create paper dictionary
            paper = {
                "title": title,
                "abstract": "",  # DBLP doesn't provide abstracts
                "authors": authors,
                "year": year,
                "citationCount": 0,  # DBLP doesn't provide citation counts
                "venue": venue,
                "url": url,
                "source": "dblp",
                "status": "Published"
            }
            
            return paper
        except Exception as e:
            logger.error(f"Error parsing DBLP entry: {e}")
            return None
    
    async def _fetch_dblp_async(self, query: str, limit: int = 10, start_year: int = 2024, end_year: int = 2026, venue_filter: str = "", offset: int = 0):
        """
        Async wrapper for DBLP search
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_dblp(query, limit, start_year, end_year, venue_filter, offset)
        )
    
    def _simplify_query_for_openalex(self, query: str) -> str:
        """
        Simplify complex boolean query for OpenAlex API.
        OpenAlex doesn't support complex boolean syntax with parentheses.
        
        Args:
            query: Complex query with OR/AND operators
            
        Returns:
            Simplified query string that preserves key concepts
        """
        import re
        
        # First, try to extract the main concepts from the query
        # Look for terms inside parentheses
        concepts = []
        
        # Extract content inside parentheses
        parenthetical_content = re.findall(r'\(([^)]+)\)', query)
        for content in parenthetical_content:
            # Split by OR to get individual concepts
            or_terms = re.split(r'\s+OR\s+', content, flags=re.IGNORECASE)
            for term in or_terms:
                # Clean up term
                term = term.strip()
                if term and len(term) > 2:
                    concepts.append(term)
        
        # If no parenthetical content, use the whole query
        if not concepts:
            # Remove parentheses
            simplified = re.sub(r'[()]', ' ', query)
            
            # Remove OR and AND operators (case insensitive)
            simplified = re.sub(r'\s+OR\s+', ' ', simplified, flags=re.IGNORECASE)
            simplified = re.sub(r'\s+AND\s+', ' ', simplified, flags=re.IGNORECASE)
            
            # Clean up multiple spaces
            simplified = re.sub(r'\s+', ' ', simplified).strip()
            
            # Extract most important terms
            terms = simplified.split()
            
            # Filter out very short terms and common words
            stop_words = {'or', 'and', 'the', 'a', 'an', 'for', 'of', 'in', 'on', 'to', 'with'}
            meaningful_terms = [t for t in terms if len(t) > 2 and t.lower() not in stop_words]
            
            # Take first 12 terms to keep query manageable but not too short
            key_terms = meaningful_terms[:12]
            
            return ' '.join(key_terms) if key_terms else simplified
        
        # If we have concepts from parentheses, use them
        # Take up to 10 concepts to keep query manageable
        key_concepts = concepts[:10]
        
        return ' '.join(key_concepts)
    
    def _simplify_query_for_dblp(self, query: str) -> str:
        """
        Simplify complex boolean query for DBLP API.
        DBLP only supports simple space-separated keyword search.
        
        Args:
            query: Complex query with OR/AND operators
            
        Returns:
            Simplified query string with core terms only
        """
        import re
        
        # First, extract key terms that should be prioritized
        key_terms = ['VLA', 'Vision-Language-Action', 'robotics', 'autonomous', 'embodied', 'AI', 'LLM']
        
        # Common venue names that should be prioritized
        venue_names = ['CVPR', 'ICCV', 'ECCV', 'ICML', 'NeurIPS', 'NIPS', 'ICLR',
                      'AAAI', 'IJCAI', 'ACL', 'EMNLP', 'NAACL', 'ICRA', 'IROS',
                      'ICCVW', 'CVPRW', 'ICMLW', 'NeurIPSW', 'ICLRW',
                      'TPAMI', 'IJCV', 'JMLR', 'JAIR', 'ACM TOG', 'ACM SIGGRAPH']
        
        # Remove parentheses
        simplified = re.sub(r'[()]', ' ', query)
        
        # Remove OR and AND operators (case insensitive)
        simplified = re.sub(r'\s+OR\s+', ' ', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s+AND\s+', ' ', simplified, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        # Extract all terms
        all_terms = simplified.split()
        
        # Filter out very short terms and common words
        stop_words = {'or', 'and', 'the', 'a', 'an', 'for', 'of', 'in', 'on', 'to', 'with', 'using', 'based'}
        filtered_terms = [t for t in all_terms if len(t) > 2 and t.lower() not in stop_words]
        
        # First, collect key terms that appear in the query
        found_key_terms = []
        for term in filtered_terms:
            if term in key_terms:
                found_key_terms.append(term)
        
        # Collect venue names that appear in the query
        found_venues = []
        for term in filtered_terms:
            if term in venue_names:
                found_venues.append(term)
        
        # Then add remaining terms by relevance
        remaining_terms = [t for t in filtered_terms if t not in found_key_terms and t not in found_venues]
        
        # Create final term list: key terms first, then venues, then remaining terms
        final_terms = found_key_terms + found_venues + remaining_terms
        
        # For DBLP, we need to keep it very simple - max 3 terms
        # Prioritize abbreviations and single words over long phrases
        simple_terms = []
        
        # First, collect all possible simple terms
        all_simple_terms = []
        
        # Process key terms first
        for term in key_terms:
            if term in final_terms:
                if term == 'Vision-Language-Action' and 'VLA' not in all_simple_terms:
                    all_simple_terms.append('VLA')
                elif '-' not in term and ' ' not in term:
                    if term not in all_simple_terms:
                        all_simple_terms.append(term)
        
        # Process venue names next
        for term in venue_names:
            if term in final_terms and term not in all_simple_terms:
                all_simple_terms.append(term)
        
        # Process remaining terms
        for term in final_terms:
            if term not in key_terms and term not in venue_names:
                if '-' in term or ' ' in term:
                    # Split long phrases into individual words
                    for word in re.split(r'[-\s]+', term):
                        if len(word) > 2 and word.lower() not in stop_words and word not in all_simple_terms:
                            all_simple_terms.append(word)
                else:
                    # Add single word terms
                    if term not in all_simple_terms:
                        all_simple_terms.append(term)
        
        # Select the first 3 simple terms
        simple_terms = all_simple_terms[:3]
        
        # Ensure we have at least some terms
        if not simple_terms:
            simple_terms = ['VLA', 'robotics']
        
        # Special case: if query contains a venue, make sure it's included
        for term in filtered_terms:
            if term in venue_names and term not in simple_terms:
                simple_terms = [term] + simple_terms[:2]
                break
        
        # Special case: if query contains 'VLA', make sure it's included
        if 'VLA' in all_terms and 'VLA' not in simple_terms:
            simple_terms = ['VLA'] + simple_terms[:2]
        
        return ' '.join(simple_terms) if simple_terms else 'VLA robotics'
    
    def _get_venue_abbreviations(self, venue_filter):
        """
        Get common abbreviations for a venue filter
        """
        abbreviations = {
            'cvpr': ['computer vision and pattern recognition', 'cvpr', 'ieee/cvf'],
            'iccv': ['international conference on computer vision', 'iccv'],
            'eccv': ['european conference on computer vision', 'eccv'],
            'icml': ['international conference on machine learning', 'icml'],
            'neurips': ['neural information processing systems', 'neurips', 'nips'],
            'iclr': ['international conference on learning representations', 'iclr'],
            'aaai': ['association for the advancement of artificial intelligence', 'aaai'],
            'ijcai': ['international joint conference on artificial intelligence', 'ijcai'],
            'acl': ['association for computational linguistics', 'acl'],
            'emnlp': ['empirical methods in natural language processing', 'emnlp'],
            'naacl': ['north american chapter of the association for computational linguistics', 'naacl'],
            'icra': ['international conference on robotics and automation', 'icra'],
            'iros': ['international conference on intelligent robots and systems', 'iros'],
            'rss': ['robotics: science and systems', 'rss'],
            'corl': ['conference on robot learning', 'corl'],
        }
        return abbreviations.get(venue_filter.lower(), [venue_filter.lower()])
    
    def _smart_merge(self, arxiv_papers, openalex_papers, query="", google_scholar_papers=None):
        """
        Intelligent merge and deduplication of papers from different sources
        
        Args:
            arxiv_papers: List of papers from arXiv
            openalex_papers: List of papers from OpenAlex
            query: Search query for relevance scoring
            google_scholar_papers: List of papers from Google Scholar
        """
        # Create a dictionary to hold merged papers by normalized title
        merged = {}
        
        # If google_scholar_papers is None, initialize as empty list
        if google_scholar_papers is None:
            google_scholar_papers = []
        
        # Process Google Scholar papers first (highest priority)
        for paper in google_scholar_papers:
            normalized_title = self._normalize_title(paper.get("title", ""))
            if normalized_title:
                merged[normalized_title] = paper
        
        # Process arXiv papers and merge with existing ones
        for paper in arxiv_papers:
            normalized_title = self._normalize_title(paper.get("title", ""))
            if normalized_title:
                if normalized_title in merged:
                    # Merge metadata - use existing (Google Scholar) information first
                    existing_paper = merged[normalized_title]
                    # Add arXiv-specific fields if missing
                    if "pdf_url" not in existing_paper and "pdf_url" in paper:
                        existing_paper["pdf_url"] = paper.get("pdf_url")
                    if "arxiv_id" not in existing_paper and "arxiv_id" in paper:
                        existing_paper["arxiv_id"] = paper.get("arxiv_id")
                else:
                    # Add new paper from arXiv
                    merged[normalized_title] = paper
        
        # Process OpenAlex papers and merge with existing ones
        for paper in openalex_papers:
            normalized_title = self._normalize_title(paper.get("title", ""))
            if normalized_title:
                if normalized_title in merged:
                    # Merge metadata
                    existing_paper = merged[normalized_title]
                    # Use OpenAlex for citation count, venue, institutions if not already from Google Scholar
                    if existing_paper.get("source") != "google_scholar":
                        existing_paper["citationCount"] = paper.get("citationCount", existing_paper.get("citationCount", 0))
                        existing_paper["venue"] = paper.get("venue", existing_paper.get("venue", ""))
                        existing_paper["institutions"] = paper.get("institutions", existing_paper.get("institutions", []))
                        existing_paper["status"] = "Published"  # If in both, mark as published
                else:
                    # Add new paper from OpenAlex
                    merged[normalized_title] = paper
        
        # Convert to list and sort by academic weight score
        merged_list = list(merged.values())
        
        # Calculate score for each paper and sort
        scored_papers = []
        for paper in merged_list:
            score = self.calculate_paper_score(paper, query)
            scored_papers.append((score, paper))
        
        # Sort by score descending
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Extract sorted papers
        sorted_papers = [paper for _, paper in scored_papers]
        
        return sorted_papers
    
    def calculate_paper_score(self, paper, query, current_year=2026):
        """
        Calculate academic weight score for a paper
        
        Args:
            paper: Paper dictionary or object
            query: Search query
            current_year: Current year for recency calculation
            
        Returns:
            Float score representing paper quality and relevance
        """
        # Define weights
        weights = {
            'relevance': 0.3,  # Relevance to query
            'citation': 0.3,   # Citation count
            'venue': 0.25,     # Venue prestige
            'recency': 0.15    # Recency
        }
        
        # Get paper attributes
        title = getattr(paper, 'title', paper.get('title', '')).lower()
        abstract = getattr(paper, 'abstract', paper.get('abstract', '')).lower()
        venue = getattr(paper, 'venue', paper.get('venue', '')).lower()
        citation_count = getattr(paper, 'citation_count', paper.get('citationCount', 0))
        year = getattr(paper, 'year', paper.get('year', 0))
        
        # Convert to appropriate types
        try:
            citation_count = int(citation_count)
        except (ValueError, TypeError):
            citation_count = 0
        
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = 0
        
        # Calculate relevance score (simple keyword matching)
        query_terms = query.lower().split()
        text_to_check = f"{title} {abstract}"
        matched_terms = [term for term in query_terms if term in text_to_check]
        relevance_score = len(matched_terms) / len(query_terms) if query_terms else 0.5
        
        # Calculate citation score (logarithmic scale)
        import math
        citation_score = math.log1p(citation_count) / 10  # Normalize to 0-1 range
        
        # Calculate venue score
        top_venues = {
            'cvpr': 1.0, 'iccv': 1.0, 'eccv': 1.0,
            'neurips': 1.0, 'iclr': 1.0, 'icml': 1.0,
            'aaai': 0.9, 'ijcai': 0.9,
            'nature': 1.0, 'science': 1.0,
            'icra': 0.9, 'iros': 0.9, 'rss': 0.9, 'corl': 0.9
        }
        venue_score = 0.0
        for ven, score in top_venues.items():
            if ven in venue:
                venue_score = score
                break
        
        # Calculate recency score (time decay)
        if year > 0:
            years_since_publication = current_year - year
            recency_score = max(0, 1 - years_since_publication * 0.1)  # Decay by 10% per year
        else:
            recency_score = 0.5
        
        # Calculate total score
        total_score = (
            relevance_score * weights['relevance'] +
            citation_score * weights['citation'] +
            venue_score * weights['venue'] +
            recency_score * weights['recency']
        )
        
        return total_score
    
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
    # Test the PaperFetcher with DBLP + arXiv + OpenAlex
    fetcher = PaperFetcher()
    
    # Test 1: Search with venue filter (should use DBLP first)
    print("=== Test 1: Search with CVPR venue filter ===")
    papers = fetcher.search_papers("Vision-Language-Action robotics", limit=5, venue_filter="CVPR", year_range="2024-2026")
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Citations: {paper['citationCount']}")
        print(f"   Venue: {paper['venue']}")
        print(f"   URL: {paper['url']}")
        print(f"   Source: {paper['source']}")
        print(f"   Status: {paper['status']}")
        if 'pdf_url' in paper:
            print(f"   PDF URL: {paper['pdf_url']}")
        if 'institutions' in paper and paper['institutions']:
            print(f"   Institutions: {', '.join(paper['institutions'])}")
        print()
    
    # Test 2: Search without venue filter (should use arXiv first)
    print("=== Test 2: Search without venue filter ===")
    papers = fetcher.search_papers("Mamba architecture for vision", limit=5, year_range="2024-2026")
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Citations: {paper['citationCount']}")
        print(f"   Venue: {paper['venue']}")
        print(f"   URL: {paper['url']}")
        print(f"   Source: {paper['source']}")
        print(f"   Status: {paper['status']}")
        if 'pdf_url' in paper:
            print(f"   PDF URL: {paper['pdf_url']}")
        if 'institutions' in paper and paper['institutions']:
            print(f"   Institutions: {', '.join(paper['institutions'])}")
        print()