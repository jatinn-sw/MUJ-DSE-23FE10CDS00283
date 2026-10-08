import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from app.config import get_config
from app.utils import load_prompt, safe_json_parse, generate_id, get_llm_client
from app.claim_extractor import Claim
from app.evidence_retriever import Evidence
from app.academic_search import AcademicSource
from app.web_search import WebSource


@dataclass
class AnalysisResult:
    claim_id: str
    verdict: str
    confidence: float
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    reasoning: str
    source_ids: List[str]
    limitations: str
    suggested_revision: Optional[str] = None


class LLMAnalyzer:
    def __init__(self):
        self.config = get_config()
        try:
            self.client = get_llm_client()
        except Exception:
            self.client = None

    def analyze(
        self,
        claim: Claim,
        internal_evidence: List[Evidence],
        external_sources: List[AcademicSource],
        web_sources: List[WebSource]
    ) -> AnalysisResult:
        prompt = load_prompt("evidence_analysis")
        
        evidence_text = self._format_evidence(internal_evidence, external_sources, web_sources)
        
        full_prompt = f"""{prompt}

Claim: {claim.text}
Category: {claim.category}

Evidence:
{evidence_text}"""
        
        try:
            if not self.client or not hasattr(self.client, 'chat'):
                return AnalysisResult(
                    claim_id=claim.claim_id,
                    verdict="INSUFFICIENT_EVIDENCE",
                    confidence=0.0,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    reasoning="LLM client not available",
                    source_ids=[],
                    limitations="LLM analysis unavailable",
                )
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Claim: {claim.text}\nCategory: {claim.category}\n\nEvidence:\n{evidence_text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = safe_json_parse(response.choices[0].message.content)
            
            return AnalysisResult(
                claim_id=claim.claim_id,
                verdict=result.get("verdict", "INSUFFICIENT_EVIDENCE"),
                confidence=result.get("confidence", 0.5),
                supporting_evidence=result.get("supporting_evidence", []),
                contradicting_evidence=result.get("contradicting_evidence", []),
                reasoning=result.get("reasoning", ""),
                source_ids=result.get("source_ids", []),
                limitations=result.get("limitations", ""),
                suggested_revision=result.get("suggested_revision"),
            )
        except Exception as e:
            # Fallback to deterministic NLP evidence evaluation when LLM quota is exhausted or offline
            return self._heuristic_analyze(claim, internal_evidence, external_sources, web_sources, error_msg=str(e))

    def _heuristic_analyze(
        self,
        claim: Claim,
        internal: List[Evidence],
        external: List[AcademicSource],
        web: List[WebSource],
        error_msg: str = "",
    ) -> AnalysisResult:
        import re
        claim_text_lower = claim.text.lower()
        
        # Check for hyperbolic / overstatement language
        overstated_markers = [
            "state-of-the-art", "outperforms all", "completely", "flawless", 
            "guarantees", "proves definitively", "always", "the first known",
            "drastically surpasses", "unprecedented"
        ]
        is_overstated = any(m in claim_text_lower for m in overstated_markers)

        # Check numerical / metric assertions
        claim_nums = re.findall(r"\d+(?:\.\d+)?%?", claim.text)
        
        # Supporting internal evidence
        strong_internal = [ev for ev in internal if ev.relevance_score >= 0.65]
        moderate_internal = [ev for ev in internal if 0.50 <= ev.relevance_score < 0.65]

        # Supporting external sources
        supporting_ext = [s for s in external if s.relevance_score >= 0.55]
        contradicting_ext = [s for s in external if any(w in s.title.lower() for w in ["limitations", "fails", "challenges", "re-evaluating"])]

        supporting_evidence = []
        for ev in strong_internal[:3]:
            supporting_evidence.append({
                "source_id": ev.source_id,
                "text": ev.text[:200],
                "type": "internal",
                "page": ev.metadata.get("page", 1)
            })
        for s in supporting_ext[:2]:
            supporting_evidence.append({
                "source_id": s.source_id,
                "title": s.title,
                "type": "external_academic",
                "doi": s.doi
            })

        contradicting_evidence = []
        for c in contradicting_ext[:2]:
            contradicting_evidence.append({
                "source_id": c.source_id,
                "title": c.title,
                "type": "external_academic"
            })

        source_ids = [e.get("source_id", "") for e in supporting_evidence + contradicting_evidence if e.get("source_id")]

        if is_overstated and strong_internal:
            verdict = "OVERSTATED"
            confidence = 0.85
            suggested = f"Under the evaluated experimental conditions, results indicate that {claim.text[:120].lower()}."
            reasoning = (
                f"While internal evidence on page {strong_internal[0].metadata.get('page', 1)} supports the general findings, "
                f"the claim uses absolute or superlative language ('{next((m for m in overstated_markers if m in claim_text_lower), 'strong terms')}') "
                f"which exceeds the scope of the empirical evaluation."
            )
        elif len(contradicting_evidence) > 0 and len(strong_internal) == 0:
            verdict = "CONTRADICTED"
            confidence = 0.78
            suggested = None
            reasoning = "Retrieved literature and methodological context challenge the stated claim without sufficient corroborating data."
        elif len(strong_internal) >= 2 or (len(strong_internal) >= 1 and claim_nums and any(n in strong_internal[0].text for n in claim_nums)):
            verdict = "SUPPORTED"
            confidence = 0.89
            suggested = None
            reasoning = (
                f"Direct internal evidence found on page {strong_internal[0].metadata.get('page', 1)} "
                f"with high semantic relevance ({strong_internal[0].relevance_score:.2f}) validates this claim within the study context."
            )
        elif len(strong_internal) == 1 or len(moderate_internal) >= 1 or len(supporting_ext) >= 1:
            verdict = "PARTIALLY_SUPPORTED"
            confidence = 0.72
            suggested = f"Preliminary evidence suggests that {claim.text[:100]} under specific benchmark setups."
            reasoning = "Evidence indicates partial alignment, but further cross-validation across diverse datasets is needed."
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.60
            suggested = None
            reasoning = "Insufficient explicit empirical data or benchmark statistics found in the document to substantiate this statement."

        return AnalysisResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            reasoning=reasoning,
            source_ids=source_ids,
            limitations="Local evidence audit; full LLM API analysis active when quota allows.",
            suggested_revision=suggested,
        )

    def _format_evidence(
        self,
        internal: List[Evidence],
        external: List[AcademicSource],
        web: List[WebSource]
    ) -> str:
        parts = []
        
        if internal:
            parts.append("INTERNAL EVIDENCE (from document):")
            for i, ev in enumerate(internal):
                parts.append(f"  [{ev.source_id}] Page {ev.metadata.get('page', '?')}: {ev.text[:300]} (relevance: {ev.relevance_score:.2f})")
        
        if external:
            parts.append("\nEXTERNAL ACADEMIC SOURCES:")
            for i, src in enumerate(external):
                parts.append(f"  [{src.source_id}] {src.title} ({src.year})")
                parts.append(f"      Authors: {', '.join(src.authors[:3])}")
                parts.append(f"      Venue: {src.venue or 'Unknown'} | Tier: {src.tier} | Citations: {src.citation_count}")
                if src.abstract:
                    parts.append(f"      Abstract: {src.abstract[:300]}")
                parts.append(f"      DOI: {src.doi or 'N/A'}")
        
        if web:
            parts.append("\nWEB SOURCES:")
            for src in web:
                parts.append(f"  [{src.source_id}] {src.title}")
                parts.append(f"      URL: {src.url}")
                parts.append(f"      Snippet: {src.snippet[:200]}")
        
        if not parts:
            parts.append("No evidence retrieved.")
        
        return "\n".join(parts)


def analyze_claim(
    claim: Claim,
    internal_evidence: List[Evidence],
    external_sources: List[AcademicSource],
    web_sources: List[WebSource]
) -> AnalysisResult:
    analyzer = LLMAnalyzer()
    return analyzer.analyze(claim, internal_evidence, external_sources, web_sources)