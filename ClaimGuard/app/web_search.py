import httpx
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from app.config import get_config
from app.utils import retry_with_backoff, generate_id


@dataclass
class WebSource:
    source_id: str
    title: str
    url: str
    snippet: str
    source_type: str
    tier: str = "C"


class SerpAPIClient:
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv("SERPAPI_API_KEY", "").strip() or self.config.api.web_search.get("api_key", "")
        self.base_url = self.config.api.web_search.get("base_url", "https://serpapi.com/search")
        self.client = httpx.Client(timeout=30.0)

    @retry_with_backoff(max_retries=3, exceptions=(httpx.RequestError,))
    def search(self, query: str, limit: int = 10) -> List[WebSource]:
        if not self.api_key:
            return []
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": limit,
            "engine": "google",
        }
        
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        sources = []
        for result in data.get("organic_results", [])[:limit]:
            source = WebSource(
                source_id=generate_id("WEB", result.get("link", "")),
                title=result.get("title", ""),
                url=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source_type="web_search",
                tier="C",
            )
            sources.append(source)
        
        return sources


class GoogleCSEClient:
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip() or self.config.api.web_search.get("google_api_key", "")
        self.cse_id = os.getenv("GOOGLE_CSE_ID", "").strip() or self.config.api.web_search.get("google_cse_id", "")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.client = httpx.Client(timeout=30.0)

    @retry_with_backoff(max_retries=3, exceptions=(httpx.RequestError,))
    def search(self, query: str, limit: int = 10) -> List[WebSource]:
        if not self.api_key or not self.cse_id:
            return []
        
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": min(limit, 10),
        }
        
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        sources = []
        for item in data.get("items", [])[:limit]:
            source = WebSource(
                source_id=generate_id("GCS", item.get("link", "")),
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source_type="google_cse",
                tier="C",
            )
            sources.append(source)
        
        return sources


def search_web(query: str, limit: int = 10) -> List[WebSource]:
    config = get_config()
    provider = config.api.web_search.get("provider", "serpapi")
    
    if provider == "serpapi":
        client = SerpAPIClient()
    elif provider == "google_cse":
        client = GoogleCSEClient()
    else:
        return []
    
    return client.search(query, limit)