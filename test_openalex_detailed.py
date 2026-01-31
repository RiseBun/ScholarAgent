import urllib.parse
import urllib.request
import json

# Test OpenAlex API with detailed error handling
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
    
    print("Making request...")
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"Response status: {resp.status} {resp.reason}")
        print(f"Response headers: {dict(resp.headers)}")
        
        # Read raw response
        raw_data = resp.read()
        print(f"Response length: {len(raw_data)} bytes")
        
        # Try to decode
        try:
            decoded_data = raw_data.decode('utf-8')
            print(f"First 500 bytes of response: {decoded_data[:500]}...")
            
            # Try to parse JSON
            try:
                data = json.loads(decoded_data)
                print(f"JSON parsing successful!")
                print(f"Response keys: {list(data.keys())}")
                print(f"Results count: {len(data.get('results', []))}")
                print(f"Meta: {data.get('meta', {})}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {decoded_data}")
        except UnicodeDecodeError as e:
            print(f"Unicode decode error: {e}")
            print(f"Raw response (first 500 bytes): {raw_data[:500]}")
        
except urllib.error.HTTPError as e:
    print(f"HTTP error: {e.code} {e.reason}")
    try:
        error_body = e.read().decode('utf-8')
        print(f"Error body: {error_body}")
    except:
        pass
except Exception as e:
    print(f"Other error: {type(e).__name__}: {str(e)}")
