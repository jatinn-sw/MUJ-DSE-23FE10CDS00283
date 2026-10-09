import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import faiss
import pickle
from pathlib import Path

from app.config import get_config
from app.utils import generate_id, cosine_similarity, get_embedding_client, _fallback_embedding
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
        
        dim = self.config.embedding.dimension
        if not self.client:
            return [_fallback_embedding(t, dim) for t in texts]
        
        try:
            raw_embs = []
            if hasattr(self.client, 'embeddings_create'):
                response = self.client.embeddings_create(input=texts, model=self.config.embedding.model)
                raw_embs = [item.embedding for item in response.data]
            elif hasattr(self.client, 'embeddings') and hasattr(self.client.embeddings, 'create'):
                response = self.client.embeddings.create(input=texts, model=self.config.embedding.model)
                raw_embs = [item.embedding for item in response.data]
            elif hasattr(self.client, 'embed_content'):
                responses = []
                for text in texts:
                    result = self.client.embed_content(
                        model=self.config.embedding.model,
                        contents=text,
                        task_type="retrieval_document"
                    )
                    responses.append(result)
                raw_embs = [r['embedding'] if isinstance(r, dict) else [r['embedding']] for r in responses]
            else:
                raw_embs = [_fallback_embedding(t, dim) for t in texts]

            # Ensure every embedding is valid non-zero
            final_embs = []
            for i, text in enumerate(texts):
                if i < len(raw_embs) and raw_embs[i] and any(x != 0.0 for x in raw_embs[i]):
                    final_embs.append(raw_embs[i])
                else:
                    final_embs.append(_fallback_embedding(text, dim))
            return final_embs
        except Exception as e:
            print(f"Embedding generation error, using fallback: {e}")
            return [_fallback_embedding(t, dim) for t in texts]

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

    def search(self, query: str, top_k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.index or not self.entries:
            return []
        
        if threshold is None:
            threshold = getattr(self.config.retrieval, "similarity_threshold", 0.45)
        
        query_embs = self.embed_texts([query])
        query_emb = query_embs[0] if query_embs else _fallback_embedding(query, self.config.embedding.dimension)
        query_vec = np.array([query_emb], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            faiss.normalize_L2(query_vec)
        
        k = min(top_k, len(self.entries))
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.entries) and score >= threshold:
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
        if not self.entries or self.index is None:
            return
        # Ensure at least some vectors are non-zero before saving
        has_nonzero = any(any(x != 0.0 for x in e.embedding) for e in self.entries)
        if not has_nonzero:
            return
        cache_file = self.cache_dir / f"{doc_id}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({
                "entries": self.entries,
                "index": self.index
            }, f)

    def load_cache(self, doc_id: str) -> bool:
        cache_file = self.cache_dir / f"{doc_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                entries = data.get("entries", [])
                index = data.get("index", None)
                if not entries or index is None:
                    cache_file.unlink(missing_ok=True)
                    return False
                # Reject corrupt all-zero cache
                has_nonzero = any(any(x != 0.0 for x in e.embedding) for e in entries)
                if not has_nonzero:
                    cache_file.unlink(missing_ok=True)
                    return False
                self.entries = entries
                self.index = index
                return True
            except Exception:
                cache_file.unlink(missing_ok=True)
                return False
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