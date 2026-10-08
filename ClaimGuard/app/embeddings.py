import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import faiss
import pickle
from pathlib import Path

from app.config import get_config
from app.utils import generate_id, cosine_similarity, get_embedding_client
from app.pdf_parser import Sentence
from app.claim_extractor import Claim


@dataclass
class EmbeddingEntry:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class EmbeddingManager:
    def __init__(self):
        self.config = get_config()
        try:
            self.client = get_embedding_client()
        except Exception:
            self.client = None
        self.index: Optional[faiss.Index] = None
        self.entries: List[EmbeddingEntry] = []
        self.cache_dir = Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.client:
            return [[0.0] * self.config.embedding.dimension for _ in texts]
        
        try:
            # Handle different client interfaces
            if hasattr(self.client, 'embeddings_create'):
                # New Gemini client
                response = self.client.embeddings_create(input=texts, model=self.config.embedding.model)
            elif hasattr(self.client, 'embed_content'):
                # Alternative Gemini client path
                responses = []
                for text in texts:
                    result = self.client.embed_content(
                        model=self.config.embedding.model,
                        contents=text,
                        task_type="retrieval_document"
                    )
                    responses.append(result)
                return [r['embedding'] if isinstance(r, dict) else [r['embedding']] for r in responses]
            else:
                return [[0.0] * self.config.embedding.dimension for _ in texts]
            return [item.embedding for item in response.data]
        except Exception:
            return [[0.0] * self.config.embedding.dimension for _ in texts]

    def build_index(self, sentences: List[Sentence]):
        texts = [s.text for s in sentences]
        embeddings = self.embed_texts(texts)
        
        self.entries = []
        for sent, emb in zip(sentences, embeddings):
            entry = EmbeddingEntry(
                id=sent.sentence_id,
                text=sent.text,
                embedding=emb,
                metadata={"page": sent.page, "section": sent.section}
            )
            self.entries.append(entry)
        
        if embeddings:
            dim = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dim)
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.65) -> List[Dict[str, Any]]:
        if not self.index or not self.entries:
            return []
        
        query_emb = self.embed_texts([query])[0]
        query_vec = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.entries) and score >= threshold:
                entry = self.entries[idx]
                results.append({
                    "sentence_id": entry.id,
                    "text": entry.text,
                    "score": float(score),
                    "page": entry.metadata["page"],
                    "section": entry.metadata["section"],
                })
        
        return results

    def save_cache(self, doc_id: str):
        cache_file = self.cache_dir / f"{doc_id}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({
                "entries": self.entries,
                "index": self.index
            }, f)

    def load_cache(self, doc_id: str) -> bool:
        cache_file = self.cache_dir / f"{doc_id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.entries = data["entries"]
            self.index = data["index"]
            return True
        return False


class ClaimEmbedder:
    def __init__(self):
        self.config = get_config()
        self.client = get_embedding_client()

    def embed_claim(self, claim: Claim) -> List[float]:
        text = f"Claim: {claim.text}\nCategory: {claim.category}"
        if hasattr(self.client, 'embeddings_create'):
            response = self.client.embeddings_create(input=[text], model=self.config.embedding.model)
        elif hasattr(self.client, 'embed_content'):
            result = self.client.embed_content(
                model=self.config.embedding.model,
                contents=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        else:
            return [0.0] * self.config.embedding.dimension

    def embed_claims(self, claims: List[Claim]) -> Dict[str, List[float]]:
        texts = [f"Claim: {c.text}\nCategory: {c.category}" for c in claims]
        if hasattr(self.client, 'embeddings_create'):
            response = self.client.embeddings_create(input=texts, model=self.config.embedding.model)
        elif hasattr(self.client, 'embed_content'):
            responses = []
            for text in texts:
                result = self.client.embed_content(
                    model=self.config.embedding.model,
                    contents=text,
                    task_type="retrieval_document"
                )
                responses.append(result['embedding'])
            return {c.claim_id: embeddings[i] for i, c in enumerate(claims)}
        else:
            return {c.claim_id: [0.0] * self.config.embedding.dimension for c in claims}


def create_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


def create_claim_embedder() -> ClaimEmbedder:
    return ClaimEmbedder()