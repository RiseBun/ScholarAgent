from backend.query_expander import QueryExpander

# Test the QueryExpander
expander = QueryExpander()

print("Testing venue extraction...")
print("=" * 50)

# Test cases
test_cases = [
    "CVPR VLA 自动驾驶",
    "ICML machine learning",
    "NeurIPS 2024 transformer",
    "VLA robotics",  # No venue mentioned
    "ICCV computer vision",
    "ECCV 2023",
    "AAAI 2024 AI ethics",
]

for query in test_cases:
    venue = expander.extract_venues(query)
    print(f"Query: '{query}'")
    print(f"Extracted venue: '{venue}'")
    print()

print("Testing expand_query with fallback...")
print("=" * 50)

# Test with no API key (should use fallback and extract venue)
for query in test_cases:
    result = expander.expand_query(query, api_key="")
    print(f"Query: '{query}'")
    print(f"Venue filter: '{result['venue_filter']}'")
    print(f"Search query: '{result['search_query']}'")
    print(f"Explanation: '{result['explanation']}'")
    print()
