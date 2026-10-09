from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.embeddings import EmbeddingManager
from app.claim_extractor import Claim
from app.pdf_parser import Sentence
from app.utils import generate_id


@dataclass
class Evidence:
    evidence_id: str
    claim_id: str
    source_type: str
    source_id: str
    text: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class InternalEvidenceRetriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.config = embedding_manager.config

    def retrieve(self, claim: Claim, top_k: Optional[int] = None) -> List[Evidence]:
        import re
        k = top_k or self.config.retrieval.internal_top_k
        threshold = getattr(self.config.retrieval, "similarity_threshold", 0.45)
        
        # Map entry IDs for quick lookup
        entry_map = {e.id: e for e in self.embedding_manager.entries}
        
        # Results collector: sentence_id -> (score, entry)
        candidate_map = {}
        
        # 1. Direct Source Sentence Linking
        for sid in (claim.sentence_ids or []):
            if sid in entry_map:
                candidate_map[sid] = (0.95, entry_map[sid])
        
        # 2. Vector Semantic Search
        vector_results = self.embedding_manager.search(claim.text, top_k=k, threshold=min(threshold, 0.40))
        for r in vector_results:
            sid = r["sentence_id"]
            score = float(r["score"])
            if sid in entry_map:
                if sid not in candidate_map or score > candidate_map[sid][0]:
                    candidate_map[sid] = (score, entry_map[sid])
        
        # 3. Lexical & Numerical Overlap Search (catch metrics, percentages, and key findings)
        claim_nums = set(re.findall(r"\d+(?:\.\d+)?%?", claim.text))
        claim_words = set(w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim.text))
        
        for e in self.embedding_manager.entries:
            if e.id in candidate_map and candidate_map[e.id][0] >= 0.85:
                continue
            
            e_text_lower = e.text.lower()
            e_words = set(w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", e.text))
            
            # Check numbers match
            has_matching_num = bool(claim_nums and any(num in e.text for num in claim_nums))
            word_overlap = len(claim_words.intersection(e_words)) / max(1, len(claim_words))
            
            if has_matching_num and word_overlap >= 0.20:
                score = round(min(0.92, 0.65 + word_overlap * 0.30), 2)
                if e.id not in candidate_map or score > candidate_map[e.id][0]:
                    candidate_map[e.id] = (score, e)
            elif word_overlap >= 0.45:
                score = round(min(0.85, 0.40 + word_overlap * 0.45), 2)
                if e.id not in candidate_map or score > candidate_map[e.id][0]:
                    candidate_map[e.id] = (score, e)

        # Sort candidates by relevance score descending
        sorted_candidates = sorted(candidate_map.items(), key=lambda item: item[1][0], reverse=True)[:k]
        
        evidence_list = []
        for sid, (score, entry) in sorted_candidates:
            evidence = Evidence(
                evidence_id=generate_id("EVID", f"{claim.claim_id}{entry.id}"),
                claim_id=claim.claim_id,
                source_type="internal",
                source_id=entry.id,
                text=entry.text,
                relevance_score=score,
                metadata={
                    "page": entry.metadata.get("page", 1),
                    "section": entry.metadata.get("section", ""),
                }
            )
            evidence_list.append(evidence)
        
        return evidence_list


class ExternalEvidenceRetriever:
    def __init__(self):
        pass

    def retrieve(self, claim: Claim, queries: List[str]) -> List[Evidence]:
        return []


def retrieve_internal_evidence(
    claim: Claim,
    embedding_manager: EmbeddingManager
) -> List[Evidence]:
    retriever = InternalEvidenceRetriever(embedding_manager)
    return retriever.retrieve(claim)


def retrieve_external_evidence(
    claim: Claim,
    queries: List[str]
) -> List[Evidence]:
    retriever = ExternalEvidenceRetriever()
    return retriever.retrieve(claim, queries)