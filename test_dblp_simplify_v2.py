from backend.search_engine import PaperFetcher

# Create PaperFetcher instance
paper_fetcher = PaperFetcher()

print("Testing updated _simplify_query_for_dblp with venue names...")
print("=" * 70)

# Test cases
test_cases = [
    "CVPR VLA 自动驾驶",
    "VLA robotics CVPR",
    "CVPR 2024 Vision-Language-Action",
    "ICML machine learning",
    "NeurIPS 2024 transformer",
    "VLA robotics",  # No venue mentioned
    "ICCV computer vision",
    "ECCV 2023",
    "AAAI 2024 AI ethics",
    "(Vision-Language-Action OR VLA OR Vision Language Action OR Multimodal LLM for Robotics) AND (robotics OR autonomous systems OR embodied AI OR agent-based systems)",
]

for query in test_cases:
    simplified = paper_fetcher._simplify_query_for_dblp(query)
    print(f"Original query: '{query}'")
    print(f"Simplified query: '{simplified}'")
    print()
