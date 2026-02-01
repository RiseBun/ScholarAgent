import asyncio
from backend.search_engine import PaperFetcher

async def test_dblp_search_with_simple_query():
    # Create PaperFetcher instance
    paper_fetcher = PaperFetcher()
    
    print("Testing DBLP search with simplified queries...")
    print("=" * 70)
    
    # Test case 1: Simple query with CVPR
    print("Test 1: Searching for 'VLA CVPR' with CVPR filter")
    print("=" * 50)
    
    try:
        # Test DBLP search directly
        dblp_papers = paper_fetcher._search_dblp(
            query="VLA",
            limit=10,
            start_year=2024,
            end_year=2026,
            venue_filter="CVPR",
            offset=0
        )
        
        print(f"DBLP returned {len(dblp_papers)} papers with CVPR filter")
        
        for i, paper in enumerate(dblp_papers[:5]):
            title = paper.get("title", "N/A")
            venue = paper.get("venue", "N/A")
            year = paper.get("year", "N/A")
            print(f"{i+1}. Title: {title[:100]}...")
            print(f"   Venue: {venue}")
            print(f"   Year: {year}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 2: Simple query without venue filter
    print("Test 2: Searching for 'VLA robotics' without venue filter")
    print("=" * 50)
    
    try:
        # Test DBLP search directly
        dblp_papers = paper_fetcher._search_dblp(
            query="VLA robotics",
            limit=10,
            start_year=2024,
            end_year=2026,
            venue_filter="",
            offset=0
        )
        
        print(f"DBLP returned {len(dblp_papers)} papers without venue filter")
        
        for i, paper in enumerate(dblp_papers[:5]):
            title = paper.get("title", "N/A")
            venue = paper.get("venue", "N/A")
            year = paper.get("year", "N/A")
            print(f"{i+1}. Title: {title[:100]}...")
            print(f"   Venue: {venue}")
            print(f"   Year: {year}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_dblp_search_with_simple_query())
