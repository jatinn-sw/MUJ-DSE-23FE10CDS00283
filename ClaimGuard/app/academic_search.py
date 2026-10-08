import httpx
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config import get_config
from app.utils import retry_with_backoff, generate_id
from app.claim_extractor import Claim


@dataclass
class AcademicSource:
    source_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    abstract: Optional[str]
    source_type: str
    tier: str
    citation_count: int = 0
    relevance_score: float = 0.0


class SemanticScholarClient:
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or self.config.api.semantic_scholar.get("api_key", "")
        self.base_url = self.config.api.semantic_scholar.get("base_url", "https://api.semanticscholar.org/graph/v1")
        self.client = httpx.Client(timeout=30.0)

    @retry_with_backoff(max_retries=3, exceptions=(httpx.RequestError,))
    def search_papers(self, query: str, limit: int = 10) -> List[AcademicSource]:
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,venue,doi,url,abstract,citationCount",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        response = self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        sources = []
        for paper in data.get("data", []):
            source = AcademicSource(
                source_id=generate_id("SS", paper.get("paperId", "")),
                title=paper.get("title", ""),
                authors=[a.get("name", "") for a in paper.get("authors", [])],
                year=paper.get("year"),
                venue=paper.get("venue"),
                doi=paper.get("doi"),
                url=paper.get("url"),
                abstract=paper.get("abstract"),
                source_type="semantic_scholar",
                tier=self._determine_tier(paper.get("venue", ""), paper.get("citationCount", 0)),
                citation_count=paper.get("citationCount", 0),
            )
            sources.append(source)
        
        return sources

    def _determine_tier(self, venue: str, citations: int) -> str:
        if not venue:
            return "C"
        venue_lower = venue.lower()
        top_venues = [
            "nature", "science", "cell", "neurips", "icml", "iclr", "cvpr", "iccv",
            "eccv", "acl", "emnlp", "naacl", "sigir", "www", "kdd", "icdm", "sdm",
            "vldb", "sigmod", "icde", "pods", "eurosys", "osdi", "sosp", "nsdi",
            "usenix security", "ieee sp", "ccs", "ndss", "crypto", "eurocrypt"
        ]
        if any(v in venue_lower for v in top_venues) or citations > 100:
            return "A"
        elif citations > 20:
            return "B"
        return "C"


class CrossrefClient:
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.api.crossref.get("base_url", "https://api.crossref.org")
        self.client = httpx.Client(timeout=30.0)

    @retry_with_backoff(max_retries=3, exceptions=(httpx.RequestError,))
    def search_works(self, query: str, limit: int = 10) -> List[AcademicSource]:
        url = f"{self.base_url}/works"
        params = {
            "query": query,
            "rows": limit,
            "select": "DOI,title,author,issued,container-title,abstract,type,is-referenced-by-count",
        }
        
        response = self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        sources = []
        for item in data.get("message", {}).get("items", []):
            authors = []
            for a in item.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)
            
            year = None
            if "issued" in item and "date-parts" in item["issued"]:
                parts = item["issued"]["date-parts"][0]
                if parts:
                    year = parts[0]
            
            source = AcademicSource(
                source_id=generate_id("CR", item.get("DOI", "")),
                title=item.get("title", [""])[0] if item.get("title") else "",
                authors=authors,
                year=year,
                venue=item.get("container-title", [""])[0] if item.get("container-title") else None,
                doi=item.get("DOI"),
                url=f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                abstract=None,
                source_type="crossref",
                tier="B",
                citation_count=item.get("is-referenced-by-count", 0),
            )
            sources.append(source)
        
        return sources


def search_academic_sources(claim: Claim, queries: List[str], max_results: int = 10) -> List[AcademicSource]:
    all_sources = []
    ss_client = SemanticScholarClient()
    cr_client = CrossrefClient()
    
    for query in queries:
        try:
            ss_results = ss_client.search_papers(query, limit=max_results // 2)
            all_sources.extend(ss_results)
        except Exception as e:
            print(f"Semantic Scholar search failed: {e}")
        
        try:
            cr_results = cr_client.search_works(query, limit=max_results // 2)
            all_sources.extend(cr_results)
        except Exception as e:
            print(f"Crossref search failed: {e}")
    
    seen_dois = set()
    unique_sources = []
    for src in all_sources:
        if src.doi and src.doi in seen_dois:
            continue
        if src.doi:
            seen_dois.add(src.doi)
        unique_sources.append(src)
    
    return rank_academic_sources(claim, unique_sources)[:max_results]


def rank_academic_sources(claim: Claim, sources: List[AcademicSource]) -> List[AcademicSource]:
    """
    Ranks academic sources using Section 11 configurable weights:
    source_score = (w_sem * semantic) + (w_key * keyword) + (w_qual * quality) + (w_rec * recency)
    """
    import re
    config = get_config()
    weights = getattr(config.source_ranking, "weights", {})
    w_sem = weights.get("semantic_similarity", 0.50)
    w_key = weights.get("keyword_similarity", 0.20)
    w_qual = weights.get("source_quality", 0.15)
    w_rec = weights.get("recency", 0.15)

    claim_words = set(re.findall(r"\w+", claim.text.lower()))
    current_year = 2026

    for src in sources:
        src_text = f"{src.title} {src.abstract or ''}".lower()
        src_words = set(re.findall(r"\w+", src_text))
        intersect = claim_words.intersection(src_words)
        union = claim_words.union(src_words)
        keyword_sim = len(intersect) / max(len(union), 1)

        semantic_sim = min(1.0, keyword_sim * 2.2 + (0.25 if intersect else 0.0))

        quality_map = {"A": 1.0, "B": 0.7, "C": 0.4}
        quality = quality_map.get(src.tier, 0.4)

        if src.year:
            age = max(0, current_year - src.year)
            recency = max(0.2, 1.0 - (age * 0.05))
        else:
            recency = 0.5

        src.relevance_score = round(
            w_sem * semantic_sim + w_key * keyword_sim + w_qual * quality + w_rec * recency,
            3
        )

    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return sources