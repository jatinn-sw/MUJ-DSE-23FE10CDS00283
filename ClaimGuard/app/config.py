import os
from pathlib import Path
from typing import Any, Dict, Literal
import yaml
from pydantic import BaseModel, field_validator
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file in ClaimGuard root directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()



class LLMConfig(BaseModel):
    provider: Literal["openai", "gemini", "anthropic"] = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 4000


class EmbeddingConfig(BaseModel):
    model: str = "gemini-embedding-001"
    dimension: int = 3072

    @field_validator("dimension", mode="before")
    @classmethod
    def set_dimension_from_model(cls, v, info):
        if "model" in info.data:
            model = info.data["model"]
            if "gemini-embedding" in model:
                return 3072
            elif "text-embedding-004" in model:
                return 768
            elif "text-embedding-3" in model:
                return 1536 if "small" in model else 3072
        return v


class RetrievalConfig(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.45
    max_search_results: int = 10
    internal_top_k: int = 10


class AnalysisConfig(BaseModel):
    max_claims: int = 30
    min_importance_score: float = 0.5


class SourceRankingConfig(BaseModel):
    weights: Dict[str, float] = {
        "semantic_similarity": 0.50,
        "keyword_similarity": 0.20,
        "source_quality": 0.15,
        "recency": 0.15,
    }


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_hours: int = 24


class FileUploadConfig(BaseModel):
    max_size_mb: int = 50
    allowed_extensions: list = [".pdf", ".docx", ".txt"]


class APIConfig(BaseModel):
    semantic_scholar: Dict[str, Any] = {}
    crossref: Dict[str, Any] = {}
    web_search: Dict[str, Any] = {}


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class EvaluationConfig(BaseModel):
    dataset_path: str = "evaluation/annotations.json"
    metrics: list = ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    source_ranking: SourceRankingConfig = SourceRankingConfig()
    cache: CacheConfig = CacheConfig()
    file_upload: FileUploadConfig = FileUploadConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


@lru_cache(maxsize=1)
def load_config() -> Config:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)
    return Config()


def get_config() -> Config:
    return load_config()