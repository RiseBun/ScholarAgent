import asyncio
from backend.search_engine import PaperFetcher

async def test_cvpr_search():
    # Create PaperFetcher instance
    paper_fetcher = PaperFetcher()
    
    print("Testing CVPR paper search...")
    print("=" * 70)
    
    # Test case 1: Search with CVPR filter
    print("Test 1: Searching for 'VLA' with CVPR filter")
    print("=" * 50)
    
    try:
        # Test the full search pipeline
        papers = await paper_fetcher.search_papers_async(
            query="VLA",
            limit=10,
            year_range="2024-2026",
            venue_filter="CVPR",
            offset=0
        )
        
        print(f"Search returned {len(papers)} papers with CVPR filter")
        
        for i, paper in enumerate(papers[:5]):
            title = paper.get("title", "N/A")
            venue = paper.get("venue", "N/A")
            year = paper.get("year", "N/A")
            source = paper.get("source", "N/A")
            print(f"{i+1}. Title: {title[:100]}...")
            print(f"   Venue: {venue}")
            print(f"   Year: {year}")
            print(f"   Source: {source}")
            print()
            
    except Exception as e:
        print(f"Error in CVPR search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: Search without CVPR filter (baseline)
    print("Test 2: Searching for 'VLA' without CVPR filter")
    print("=" * 50)
    
    try:
        # Test the full search pipeline
        papers = await paper_fetcher.search_papers_async(
            query="VLA",
            limit=10,
            year_range="2024-2026",
            venue_filter="",
            offset=0
        )
        
        print(f"Search returned {len(papers)} papers without CVPR filter")
        
        for i, paper in enumerate(papers[:5]):
            title = paper.get("title", "N/A")
            venue = paper.get("venue", "N/A")
            year = paper.get("year", "N/A")
            source = paper.get("source", "N/A")
            print(f"{i+1}. Title: {title[:100]}...")
            print(f"   Venue: {venue}")
            print(f"   Year: {year}")
            print(f"   Source: {source}")
            print()
            
    except Exception as e:
        print(f"Error in baseline search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cvpr_search())
