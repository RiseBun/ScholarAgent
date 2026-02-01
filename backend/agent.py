import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, List

# Provider configurations: base_url and env_var for API key
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "base_url_env_var": "OPENAI_API_BASE_URL",
    },
    "qianwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "QWEN_API_KEY",
        "base_url_env_var": "QWEN_API_BASE_URL",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_var": "DEEPSEEK_API_KEY",
        "base_url_env_var": "DEEPSEEK_API_BASE_URL",
    },
    "gemini": {
        "base_url": None,  # Uses native SDK
        "env_var": "GEMINI_API_KEY",
        "base_url_env_var": "GEMINI_API_BASE_URL",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "base_url_env_var": "OPENROUTER_API_BASE_URL",
    },
}

class LLMClient:
    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        base_url: Optional[str] = None,
        default_model: str = None,
        request_timeout: int = 600,
    ) -> None:
        self.provider = provider.lower()
        # Set default model based on provider
        if default_model:
            self.default_model = default_model
        else:
            # Default models for each provider
            provider_models = {
                "openai": "gpt-3.5-turbo",
                "qianwen": "qwen-turbo",
                "deepseek": "deepseek-chat",
                "gemini": "gemini-1.5-flash",
                "openrouter": "gpt-3.5-turbo",
            }
            self.default_model = provider_models.get(self.provider, "gpt-3.5-turbo")
        self.request_timeout = request_timeout
        
        # Get provider config
        config = PROVIDER_CONFIGS.get(self.provider, PROVIDER_CONFIGS["openai"])
        
        if self.provider == "gemini":
            # Use native Google Generative AI SDK
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._genai = genai
                self._client = None
                self._http_client = None
            except ImportError:
                print("Google Generative AI SDK not installed. Please run 'pip install google-generativeai'")
                self._genai = None
                self._client = None
                self._http_client = None
        else:
            # Use OpenAI-compatible API
            try:
                import httpx
                from openai import OpenAI
                
                self._genai = None
                
                # Set up headers (OpenRouter needs special headers)
                extra_headers = {
                    "Content-Type": "application/json; charset=utf-8",
                }
                if self.provider == "openrouter":
                    extra_headers.update({
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "ScholarAgent",
                    })

                self._http_client = httpx.Client(
                    trust_env=True, 
                    timeout=request_timeout,
                    headers=extra_headers,
                    default_encoding="utf-8"
                )
                
                base_url_env_var = config.get("base_url_env_var")
                env_base_url = os.environ.get(base_url_env_var) if base_url_env_var else None
                effective_base_url = base_url or env_base_url or config["base_url"]
                
                self._client = OpenAI(
                    base_url=effective_base_url,
                    api_key=api_key,
                    http_client=self._http_client,
                )
            except ImportError:
                print("OpenAI SDK not installed. Please run 'pip install openai'")
                self._client = None
                self._http_client = None

    def generate(
        self,
        system_prompt: Optional[str],
        user_query: str,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate response from LLM
        """
        if self.provider == "gemini" and self._genai:
            # Use Gemini SDK
            try:
                model_name = model or "gemini-1.5-flash"
                gemini_model = self._genai.GenerativeModel(model_name)
                response = gemini_model.generate_content(
                    f"{system_prompt}\n\n{user_query}",
                    generation_config={"temperature": 0.0}
                )
                return response.text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return None
        elif self._client:
            # Use OpenAI-compatible API
            try:
                model_name = model or self.default_model
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.0
                )
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling {self.provider} API: {e}")
                return None
        return None

def parse_user_intent(user_query: str, llm_provider: str = "openai", api_key: str = "") -> dict:
    """
    Parse user intent from natural language query
    
    Args:
        user_query: Natural language query from user
        llm_provider: LLM provider to use (openai, qianwen, deepseek, gemini, openrouter)
        api_key: API key for the selected LLM provider
        
    Returns:
        Dict with keywords and filters
    """
    system_prompt = """
    You are an assistant for a research paper search tool called ScholarAgent.
    Your task is to parse user queries into structured format with keywords and filters.
    
    Output format:
    {
        "keywords": "Core search terms",
        "filters": {
            "require_code": boolean,
            "require_top_venue": boolean,
            "highly_cited_only": boolean,
            "affiliation": "Optional company name",
            "year_start": integer,
            "year_end": integer
        }
    }
    
    Examples:
    Input: "Find me recent Google papers on VLA with code."
    Output:
    {
        "keywords": "Vision Language Action",
        "filters": {
            "require_code": true,
            "require_top_venue": false,
            "highly_cited_only": false,
            "affiliation": "Google",
            "year_start": 2024,
            "year_end": 2026
        }
    }
    
    Input: "Mamba architecture for vision"
    Output:
    {
        "keywords": "Mamba architecture for vision",
        "filters": {
            "require_code": false,
            "require_top_venue": false,
            "highly_cited_only": false,
            "affiliation": "",
            "year_start": 2024,
            "year_end": 2026
        }
    }
    """
    
    # Get API key from parameter or environment variable
    if not api_key:
        config = PROVIDER_CONFIGS.get(llm_provider, PROVIDER_CONFIGS["openai"])
        api_key = os.getenv(config.get("env_var"))
    
    if not api_key:
        # Return default structure if no API key
        return {
            "keywords": user_query,
            "filters": {
                "require_code": False,
                "require_top_venue": False,
                "highly_cited_only": False,
                "affiliation": "",
                "year_start": 2024,
                "year_end": 2026
            }
        }
    
    # Call LLM API
    try:
        client = LLMClient(api_key=api_key, provider=llm_provider)
        response = client.generate(system_prompt, user_query)
        
        if response:
            try:
                intent = json.loads(response)
                return intent
            except json.JSONDecodeError:
                # If JSON parsing fails, return default structure
                pass
    except Exception as e:
        print(f"Error calling {llm_provider} API: {e}")
    
    # Return default structure if any error occurs
    return {
        "keywords": user_query,
        "filters": {
            "require_code": False,
            "require_top_venue": False,
            "highly_cited_only": False,
            "affiliation": "",
            "year_start": 2024,
            "year_end": 2026
        }
    }

if __name__ == "__main__":
    # Test the parse_user_intent function
    test_queries = [
        "Find me recent Google papers on VLA with code.",
        "Mamba architecture for vision",
        "Highly cited papers on transformer models from Meta"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        intent = parse_user_intent(query)
        print(f"Intent: {json.dumps(intent, indent=2)}")
        print()