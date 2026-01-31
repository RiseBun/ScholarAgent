from backend.search_engine import PaperFetcher
import asyncio

async def test_full_search():
    """Test the full search functionality with both arXiv and OpenAlex"""
    print("Testing full search functionality...")
    
    # Create fetcher instance
    fetcher = PaperFetcher()
    
    # Test search with query 'vla'
    query = "vla"
    limit = 10
    year_range = "2024-2026"
    
    print(f"Searching for '{query}' with limit={limit}, year_range={year_range}")
    
    # Run the search
    papers = await fetcher.search_papers_async(query, limit=limit, year_range=year_range)
    
    print(f"\nSearch completed! Found {len(papers)} papers:")
    
    # Print results
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Citations: {paper.get('citationCount', 0)}")
        print(f"   Venue: {paper.get('venue', 'N/A')}")
        print(f"   Source: {paper.get('source', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_full_search())
