import urllib.parse
import urllib.request
import json

# Test OpenAlex API with corrected filter format
OPENALEX_API = "https://api.openalex.org/works"
query = "vla"
start_year = 2024
end_year = 2026
limit = 5

# Build params with corrected filter format
params = {
    "search": query,
    "filter": f"publication_year:{start_year}-{end_year}",
    "per-page": limit * 2,
    "mailto": "research@example.com"
}

# Construct URL
url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
print(f"Testing OpenAlex API with URL: {url}")

# Make request
try:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ScholarAgent/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    
    print(f"Success! OpenAlex returned {len(data.get('results', []))} papers")
    print(f"Total results available: {data.get('meta', {}).get('count', 0)}")
    
    # Print first few results
    for i, item in enumerate(data.get('results', [])[:3], 1):
        print(f"\n{i}. {item.get('title', 'No title')}")
        print(f"   Year: {item.get('publication_year', 'N/A')}")
        print(f"   Authors: {', '.join([auth.get('author', {}).get('display_name', '') for auth in item.get('authorships', []) if auth.get('author')])}")
        print(f"   Citations: {item.get('cited_by_count', 0)}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
