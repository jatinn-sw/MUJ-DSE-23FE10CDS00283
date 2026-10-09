import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import spacy

from app.config import get_config
from app.utils import (
    load_prompt, safe_json_parse, generate_id,
    calculate_importance_score, has_numbers,
    has_comparative_language, has_causal_language,
    extract_named_entities, get_llm_client
)
from app.pdf_parser import Sentence


@dataclass
class Claim:
    claim_id: str
    text: str
    sentence_ids: List[str]
    category: str
    importance: float


class ClaimExtractor:
    def __init__(self):
        self.config = get_config()
        try:
            self.client = get_llm_client()
        except Exception:
            self.client = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract_claims(self, sentences: List[Sentence]) -> List[Claim]:
        candidate_sentences = self._filter_candidates(sentences)
        
        if not candidate_sentences:
            return []
        
        try:
            claims = self._extract_with_llm(candidate_sentences)
        except Exception:
            claims = self._fallback_extraction(candidate_sentences)
        claims = self._classify_and_score(claims, sentences)
        
        claims.sort(key=lambda c: c.importance, reverse=True)
        return claims[:self.config.analysis.max_claims]

    def _filter_candidates(self, sentences: List[Sentence]) -> List[Sentence]:
        candidates = []
        for sent in sentences:
            text = sent.text.lower()
            
            skip_patterns = [
                r"^figure\s+\d+",
                r"^table\s+\d+",
                r"^references?",
                r"^acknowledgement",
                r"^we\s+(used|employ|utilize|apply|propose)\s",
                r"^this\s+(paper|work|study|section)\s",
                r"^\d+\.\s*$",
            ]
            
            if any(re.search(p, text) for p in skip_patterns):
                continue
            
            if len(text.split()) < 8:
                continue
            
            claim_indicators = [
                r"\d+(\.\d+)?%",
                r"\b(achieve|outperform|surpass|exceed|reduce|increase|improve)\b",
                r"\b(significant|proves?|cause|lead to|result in)\b",
                r"\b(state-of-the-art|first|novel|new)\b",
                r"\b(compared to|versus|vs\.?|better than|worse than)\b",
                r"\b(conclude|demonstrate|show|indicate|suggest)\b",
            ]
            
            if any(re.search(p, text) for p in claim_indicators):
                candidates.append(sent)
        
        return candidates

    def _extract_with_llm(self, sentences: List[Sentence]) -> List[Dict[str, Any]]:
        if not sentences:
            return []
        
        prompt = load_prompt("claim_extraction")
        sentences_text = "\n".join([
            f"{s.sentence_id}: {s.text}" for s in sentences
        ])
        
        try:
            if not self.client or not hasattr(self.client, 'chat'):
                return self._fallback_extraction(sentences)
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Sentences:\n{sentences_text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = safe_json_parse(response.choices[0].message.content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("claims", [])
            return self._fallback_extraction(sentences)
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._fallback_extraction(sentences)

    def _fallback_extraction(self, sentences: List[Sentence]) -> List[Dict[str, Any]]:
        claims = []
        for sent in sentences:
            claims.append({
                "text": sent.text,
                "sentence_ids": [sent.sentence_id],
                "category": "FACTUAL",
                "importance": 0.5
            })
        return claims

    def _classify_and_score(
        self,
        raw_claims: List[Dict[str, Any]],
        all_sentences: List[Sentence]
    ) -> List[Claim]:
        claims = []
        sent_map = {s.sentence_id: s for s in all_sentences}
        
        for raw in raw_claims:
            text = raw.get("text", "").strip()
            if not text:
                continue
            
            sentence_ids = raw.get("sentence_ids", [])
            category = raw.get("category", "FACTUAL")
            importance = raw.get("importance", 0.5)
            
            entities = []
            for sid in sentence_ids:
                if sid in sent_map:
                    if self.nlp:
                        doc = self.nlp(sent_map[sid].text)
                        entities.extend([ent.text for ent in doc.ents])
            
            has_nums = has_numbers(text)
            has_comp = has_comparative_language(text)
            has_caus = has_causal_language(text)
            
            section = sent_map[sentence_ids[0]].section if sentence_ids else "Unknown"
            
            calc_importance = calculate_importance_score(
                text, section, entities, has_nums, has_comp, has_caus
            )
            
            final_importance = max(importance, calc_importance)
            
            claim = Claim(
                claim_id=generate_id("C", text),
                text=text,
                sentence_ids=sentence_ids,
                category=category,
                importance=final_importance,
            )
            claims.append(claim)
        
        return claims


def extract_claims(sentences: List[Sentence]) -> List[Claim]:
    extractor = ClaimExtractor()
    return extractor.extract_claims(sentences)