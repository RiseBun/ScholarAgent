#!/usr/bin/env python3
"""
Test script to verify backend functionality
"""

from backend.search_engine import PaperFetcher
from backend.tagger import enrich_and_tag
from backend.agent import parse_user_intent

def test_search_engine():
    """Test PaperFetcher functionality"""
    print("Testing Search Engine...")
    fetcher = PaperFetcher()
    papers = fetcher.search_papers("Mamba architecture", limit=5)
    print(f"Found {len(papers)} papers")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
    print()

def test_tagger():
    """Test tagger functionality"""
    print("Testing Tagger...")
    fetcher = PaperFetcher()
    raw_papers = fetcher.search_papers("Mamba architecture", limit=5)
    tagged_papers = enrich_and_tag(raw_papers)
    print(f"Tagged {len(tagged_papers)} papers")
    for i, paper in enumerate(tagged_papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Tags: Has Code={paper.has_code}, Top Venue={paper.is_top_venue}, Highly Cited={paper.is_highly_cited}")
    print()

def test_agent():
    """Test LLM intent parsing functionality"""
    print("Testing LLM Intent Parser...")
    try:
        test_queries = [
            "Find me recent papers on Mamba architecture with code",
            "Highly cited papers on transformers from Google",
            "Vision transformer papers from 2024"
        ]
        for query in test_queries:
            try:
                intent = parse_user_intent(query)
                print(f"Query: {query}")
                print(f"Keywords: {intent['keywords']}")
                print(f"Filters: {intent['filters']}")
            except Exception as e:
                print(f"Error parsing intent: {e}")
            print()
    except Exception as e:
        print(f"Skipping intent parser test: {e}")
    print()

if __name__ == "__main__":
    print("=== Testing ScholarAgent Backend ===")
    test_search_engine()
    test_tagger()
    test_agent()
    print("=== Testing Complete ===")
