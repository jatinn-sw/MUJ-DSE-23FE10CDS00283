import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from app.pdf_parser import parse_document, Document
from app.claim_extractor import extract_claims, Claim
from app.embeddings import create_embedding_manager, EmbeddingManager
from app.evidence_retriever import retrieve_internal_evidence, Evidence
from app.query_generator import generate_queries
from app.academic_search import search_academic_sources, AcademicSource
from app.web_search import search_web, WebSource
from app.llm_analyzer import analyze_claim, AnalysisResult
from app.report_generator import generate_report, AuditReport
from app.utils import generate_id
from app.config import get_config


STAGE_WEIGHTS: Dict[str, float] = {
    "document_parsed": 10.0,
    "claims_extracted": 15.0,
    "embeddings_built": 15.0,
    "internal_evidence_retrieved": 10.0,
    "external_search_started": 5.0,
    "academic_search": 15.0,
    "web_search": 5.0,
    "llm_analysis": 20.0,
    "report_generated": 5.0,
}


def stage_progress(stage: str) -> float:
    """Cumulative percentage (0-100) reached once the given stage has started."""
    total = 0.0
    for key, weight in STAGE_WEIGHTS.items():
        total += weight
        if key == stage:
            return total
    return total


def _check_and_clear_embedding_cache():
    """Check if LLM/embedding provider has changed since last run, clear cache if so."""
    cache_file = Path("cache/embeddings/.provider")
    current_provider = f"{get_config().llm.provider}:{get_config().embedding.model}"
    
    if cache_file.exists():
        with open(cache_file, "r") as f:
            last_provider = f.read().strip()
        if last_provider != current_provider:
            # Provider changed - clear embedding cache
            cache_dir = Path("cache/embeddings")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Write current provider
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(current_provider)


@dataclass
class PipelineState:
    document: Optional[Document] = None
    claims: List[Claim] = field(default_factory=list)
    embedding_manager: Optional[EmbeddingManager] = None
    internal_evidence: Dict[str, List[Evidence]] = field(default_factory=dict)
    queries: Dict[str, List[str]] = field(default_factory=dict)
    external_sources: List[AcademicSource] = field(default_factory=list)
    web_sources: List[WebSource] = field(default_factory=list)
    analyses: List[AnalysisResult] = field(default_factory=list)
    report: Optional[AuditReport] = None
    processing_time: float = 0.0
    api_calls: int = 0
    errors: List[str] = field(default_factory=list)


class ClaimGuardPipeline:
    def __init__(self):
        self.config = get_config()
        self.state = PipelineState()

    def run(self, file_path: str, progress_callback=None) -> AuditReport:
        start_time = time.time()
        self.state = PipelineState()
        
        # Check and clear embedding cache if provider changed
        try:
            _check_and_clear_embedding_cache()
        except Exception:
            pass
        
        try:
            self._stage_document_processing(file_path, progress_callback)
            self._stage_claim_extraction(progress_callback)
            self._stage_embedding_building(progress_callback)
            self._stage_internal_evidence(progress_callback)
            try:
                self._stage_external_search(progress_callback)
            except Exception:
                pass
            self._stage_llm_analysis(progress_callback)
            self._stage_report_generation(progress_callback)
        except Exception as e:
            self.state.errors.append(f"Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        self.state.processing_time = time.time() - start_time
        # Ensure report is not None
        if self.state.report is None:
            from app.report_generator import generate_report
            self.state.report = generate_report(
                self.state.document,
                self.state.claims,
                self.state.analyses,
                self.state.external_sources,
                self.state.web_sources,
                self.state.internal_evidence,
                self.state.processing_time,
            ) if self.state.document else None
        else:
            self.state.report.internal_evidence = self.state.internal_evidence
            self.state.report.processing_time = self.state.processing_time
        return self.state.report

    def _stage_document_processing(self, file_path: str, callback):
        if callback:
            callback("document_parsed", "Parsing PDF document...", stage_progress("document_parsed"))
        self.state.document = parse_document(file_path)

    def _stage_claim_extraction(self, callback):
        if callback:
            callback("claims_extracted", "Extracting research claims...", stage_progress("claims_extracted"))
        self.state.claims = extract_claims(self.state.document.sentences)

    def _stage_embedding_building(self, callback):
        if callback:
            callback("embeddings_built", "Generating vector embeddings...", stage_progress("embeddings_built") - 5)
        self.state.embedding_manager = create_embedding_manager()
        doc_id = generate_id("DOC", self.state.document.filename)
        if not self.state.embedding_manager.load_cache(doc_id):
            self.state.embedding_manager.build_index(self.state.document.sentences)
            self.state.embedding_manager.save_cache(doc_id)
        if callback:
            callback("embeddings_built", f"Indexed {len(self.state.document.sentences)} sentences", stage_progress("embeddings_built"))

    def _stage_internal_evidence(self, callback):
        total = len(self.state.claims)
        base = stage_progress("embeddings_built")
        span = STAGE_WEIGHTS.get("internal_evidence_retrieved", 10.0)
        for i, claim in enumerate(self.state.claims):
            if callback:
                pct = base + (i / max(1, total)) * span
                callback("internal_evidence_retrieved", f"Matching evidence for claim {i+1}/{total}...", pct)
            evidence = retrieve_internal_evidence(claim, self.state.embedding_manager)
            self.state.internal_evidence[claim.claim_id] = evidence
        if callback:
            callback("internal_evidence_retrieved", f"Retrieved internal evidence for {total} claims", stage_progress("internal_evidence_retrieved"))

    def _stage_external_search(self, callback):
        if callback:
            callback("external_search_started", "Searching academic & external sources...", stage_progress("external_search_started"))
        
        search_claims = sorted(self.state.claims, key=lambda c: c.importance, reverse=True)[:5]
        
        all_academic = []
        all_web = []
        seen_doi = set()
        seen_web = set()

        for claim in search_claims:
            try:
                queries = generate_queries(claim)
            except Exception:
                queries = [claim.text[:80]]
            self.state.queries[claim.claim_id] = queries

            try:
                acad_sources = search_academic_sources(claim, queries, max_results=self.config.retrieval.max_search_results)
                for s in acad_sources:
                    if s.doi and s.doi in seen_doi:
                        continue
                    if s.doi:
                        seen_doi.add(s.doi)
                    all_academic.append(s)
            except Exception as e:
                self.state.errors.append(f"Academic search error: {e}")

            try:
                for q in queries[:1]:
                    web_res = search_web(q, limit=3)
                    for w in web_res:
                        if w.url and w.url in seen_web:
                            continue
                        if w.url:
                            seen_web.add(w.url)
                        all_web.append(w)
            except Exception as e:
                self.state.errors.append(f"Web search error: {e}")

        self.state.external_sources = all_academic
        self.state.web_sources = all_web

        if callback:
            callback("academic_search", f"Retrieved {len(all_academic)} academic papers", stage_progress("academic_search"))
            callback("web_search", f"Retrieved {len(all_web)} web sources", stage_progress("web_search"))

    def _stage_llm_analysis(self, callback):
        total = len(self.state.claims)
        base = stage_progress("web_search")
        span = STAGE_WEIGHTS.get("llm_analysis", 20.0)
        for i, claim in enumerate(self.state.claims):
            if callback:
                pct = base + (i / max(1, total)) * span
                callback("llm_analysis", f"Auditing claim {i+1}/{total}: \"{claim.text[:35]}...\"", pct)
            internal = self.state.internal_evidence.get(claim.claim_id, [])
            external = [s for s in self.state.external_sources]
            web = [s for s in self.state.web_sources]

            analysis = analyze_claim(claim, internal, external, web)
            self.state.analyses.append(analysis)
        if callback:
            callback("llm_analysis", f"Completed audit for {total} claims", stage_progress("llm_analysis"))

    def _stage_report_generation(self, callback):
        if callback:
            callback("report_generated", "Compiling final audit report...", stage_progress("report_generated"))
        self.state.report = generate_report(
            self.state.document,
            self.state.claims,
            self.state.analyses,
            self.state.external_sources,
            self.state.web_sources,
            self.state.internal_evidence,
            self.state.processing_time,
        )


def run_pipeline(file_path: str, progress_callback=None) -> AuditReport:
    pipeline = ClaimGuardPipeline()
    return pipeline.run(file_path, progress_callback)