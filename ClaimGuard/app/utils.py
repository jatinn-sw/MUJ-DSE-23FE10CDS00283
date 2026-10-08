import json
import hashlib
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from functools import wraps, lru_cache
import re

from app.config import get_config
from google.genai import Client

logger = logging.getLogger(__name__)


def setup_logging():
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def generate_id(prefix: str, content: str) -> str:
    hash_obj = hashlib.md5(content.encode())
    return f"{prefix}_{hash_obj.hexdigest()[:8].upper()}"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    return nltk.sent_tokenize(text)


def calculate_importance_score(
    sentence: str,
    section: str,
    named_entities: List[str],
    has_numbers: bool,
    has_comparative: bool,
    has_causal: bool,
) -> float:
    score = 0.0
    
    if has_numbers:
        score += 0.25
    if named_entities:
        score += 0.15 * min(len(named_entities), 3)
    if has_comparative:
        score += 0.2
    if has_causal:
        score += 0.2
    if section and section.lower() in ["conclusion", "results", "discussion", "abstract"]:
        score += 0.15
    
    strong_words = [
        "significantly", "proves", "causes", "outperforms", "reduces",
        "increases", "first", "state-of-the-art", "always", "never",
        "guarantees", "demonstrates", "validates", "confirms"
    ]
    for word in strong_words:
        if word in sentence.lower():
            score += 0.05
    
    return min(score, 1.0)


def has_numbers(text: str) -> bool:
    return bool(re.search(r"\d+(\.\d+)?%?", text))


def has_comparative_language(text: str) -> bool:
    comparative = [
        "outperform", "better than", "worse than", "superior", "inferior",
        "compared to", "versus", "vs", "exceeds", "surpass", "overcome"
    ]
    text_lower = text.lower()
    return any(c in text_lower for c in comparative)


def has_causal_language(text: str) -> bool:
    causal = [
        "causes", "caused", "leads to", "results in", "due to",
        "because of", "effect of", "impact of", "influence", "affects"
    ]
    text_lower = text.lower()
    return any(c in text_lower for c in causal)


def extract_named_entities(text: str) -> List[str]:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "EVENT"]]


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            logger.error(f"All retries exhausted. Last error: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


def safe_json_parse(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return default


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def load_prompt(prompt_name: str) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def save_json(data: Any, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@lru_cache(maxsize=1)
def _get_llm_provider() -> str:
    """Get configured LLM provider from config."""
    return get_config().llm.provider


@lru_cache(maxsize=1)
def _get_embedding_provider() -> str:
    """Get configured embedding provider from config."""
    return get_config().embedding.model


def _get_gemini_api_key() -> str:
    """Get Gemini API key from environment."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return key.strip()


def _get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
    return key.strip()


def _get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY") or ""
    return key.strip()


def get_llm_client():
    """
    Factory function to get LLM client based on configured provider.
    Returns a client with chat.completions.create() method compatible with OpenAI API.
    """
    provider = _get_llm_provider()
    config = get_config()
    
    if provider == "gemini":
        from google.genai import Client
        api_key = _get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        client = Client(api_key=api_key)
        return _NewGeminiLLMClient(client, config.llm.model)
    
    elif provider == "openai":
        import openai
        api_key = _get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return openai.OpenAI(api_key=api_key)
    
    elif provider == "anthropic":
        import anthropic
        api_key = _get_anthropic_api_key()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return _AnthropicLLMClient(config.llm.model, api_key)
    
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_embedding_client():
    """
    Factory function to get embedding client based on configured provider.
    Returns a client with embeddings.create() method compatible with OpenAI API.
    """
    provider = _get_llm_provider()
    config = get_config()
    
    if provider == "gemini":
        from google.genai import Client
        api_key = _get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        client = Client(api_key=api_key)
        return _NewGeminiEmbeddingClient(client, config.embedding.model)
    
    elif provider == "openai":
        import openai
        api_key = _get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return openai.OpenAI(api_key=api_key)
    
    raise ValueError(f"Unknown embedding provider: {provider}")


class _NewGeminiLLMClient:
    """Gemini client wrapper compatible with OpenAI chat.completions.create() interface using google.genai."""
    
    def __init__(self, client: Client, model: str):
        self.client = client
        self.model_name = model.replace("models/", "")
        self.chat = type('obj', (object,), {'completions': type('obj', (object,), {'create': self._create})})()
    
    def _create(self, model: str = None, messages: list = None, temperature: float = 0, max_tokens: int = 4000, 
                response_format: dict = None, **kwargs):
        import google.genai as genai
        
        system_prompt = ""
        user_content = ""
        
        for msg in (messages or []):
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")
        
        full_prompt = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
        
        generate_config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if response_format and response_format.get("type") == "json_object":
            generate_config.response_mime_type = "application/json"
        
        target_model = model or self.model_name
        models_to_try = [target_model, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        seen = set()
        candidates = []
        for m in models_to_try:
            if m and m not in seen:
                seen.add(m)
                candidates.append(m)
        
        last_error = None
        for cand in candidates:
            try:
                response = self.client.models.generate_content(
                    model=cand,
                    contents=full_prompt,
                    config=generate_config
                )
                self.model_name = cand
                
                class Choice:
                    def __init__(self, text):
                        self.message = type('obj', (object,), {'content': text})()
                
                class Response:
                    def __init__(self, text):
                        self.choices = [Choice(text)]
                
                return Response(response.text)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(k in err_str for k in ("quota", "resource_exhausted", "429", "not found", "no longer available")):
                    continue
                raise
        if last_error:
            raise last_error
        
        class Choice:
            def __init__(self, text):
                self.message = type('obj', (object,), {'content': text})()
        
        class Response:
            def __init__(self, text):
                self.choices = [Choice(text)]
        
        return Response(response.text)


class _AnthropicLLMClient:
    """Anthropic client wrapper compatible with OpenAI chat.completions.create() interface."""
    
    def __init__(self, model: str, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.chat = type('obj', (object,), {'completions': type('obj', (object,), {'create': self._create})})()
    
    def _create(self, model: str, messages: list, temperature: float = 0, max_tokens: int = 4000,
                response_format: dict = None, **kwargs):
        system_prompt = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)
        
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        class Choice:
            def __init__(self, text):
                self.message = type('obj', (object,), {'content': text})()
        
        class Response:
            def __init__(self, text):
                self.choices = [Choice(text)]
        
        return Response(response.content[0].text)


class _NewGeminiEmbeddingClient:
    """Gemini embedding client wrapper using google.genai."""
    
    def __init__(self, client: Client, model: str):
        self.client = client
        self.model_name = model.replace("models/", "")
        clean_model = self.model_name
        if clean_model in ("text-embedding-004", "text-embedding-3-small", "text-embedding-3-large"):
            clean_model = "gemini-embedding-001"
        self.model_name = clean_model
    
    def embeddings_create(self, input: Any, model: str = None, **kwargs):
        import google.genai as genai
        
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            input = list(input)
        
        target_model = model or self.model_name
        clean_target = target_model.replace("models/", "")
        if clean_target in ("text-embedding-004", "text-embedding-3-small", "text-embedding-3-large"):
            clean_target = "gemini-embedding-001"
        full_model = f"models/{clean_target}"
        
        embeddings = []
        batch_size = 50
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            try:
                result = self.client.embed_content(
                    model=full_model,
                    contents=batch,
                    task_type="retrieval_document"
                )
                emb = result.get('embedding', [])
                if emb and isinstance(emb[0], list):
                    embeddings.extend(emb)
                elif emb:
                    embeddings.append(emb)
            except Exception:
                for text in batch:
                    try:
                        res = self.client.embed_content(
                            model=full_model,
                            contents=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(res['embedding'])
                    except Exception:
                        embeddings.append([0.0] * 3072)
        
        class EmbeddingData:
            def __init__(self, embedding):
                self.embedding = embedding
        
        class Response:
            def __init__(self, embeddings):
                self.data = [EmbeddingData(e) for e in embeddings]
        
        return Response(embeddings)