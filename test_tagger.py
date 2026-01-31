#!/usr/bin/env python3
"""
Test script to verify tagger functionality
"""

from backend.tagger import enrich_and_tag

# Test data
TEST_PAPERS = [
    {
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": ["Albert Gu", "Tri Dao"],
        "year": 2024,
        "abstract": "We introduce Mamba, a new architecture that achieves linear-time sequence modeling...",
        "url": "https://github.com/state-spaces/mamba",
        "citationCount": 100,
        "venue": "NeurIPS"
    },
    {
        "title": "Vision Transformer",
        "authors": ["Alexey Dosovitskiy", "Lucas Beyer"],
        "year": 2020,
        "abstract": "We show that a pure transformer applied directly to sequences of image patches...",
        "url": "https://arxiv.org/abs/2010.11929",
        "citationCount": 10000,
        "venue": "ICLR"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
        "year": 2019,
        "abstract": "We introduce a new language representation model called BERT...",
        "url": "https://arxiv.org/abs/1810.04805",
        "citationCount": 50000,
        "venue": "NAACL"
    }
]

def test_tagger():
    """Test tagger functionality"""
    print("Testing Tagger...")
    print(f"Testing with {len(TEST_PAPERS)} papers")
    print()
    
    # Test tagging
    tagged_papers = enrich_and_tag(TEST_PAPERS)
    
    # Print results
    for i, paper in enumerate(tagged_papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Year: {paper.year}")
        print(f"   Venue: {paper.venue}")
        print(f"   Citation Count: {paper.citation_count}")
        print(f"   Tags:")
        print(f"     Has Code: {paper.has_code}")
        print(f"     Top Venue: {paper.is_top_venue}")
        print(f"     Highly Cited: {paper.is_highly_cited}")
        print(f"     Open Access: {paper.is_open_access}")
        print()

if __name__ == "__main__":
    print("=== Testing ScholarAgent Tagger ===")
    test_tagger()
    print("=== Testing Complete ===")
