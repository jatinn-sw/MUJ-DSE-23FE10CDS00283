import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from app.claim_extractor import Claim
from app.llm_analyzer import AnalysisResult
from app.pdf_parser import Document
from app.academic_search import AcademicSource
from app.web_search import WebSource
from app.report_generator import AuditReport
from app.config import get_config


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin: 0; font-size: 1.5rem;">🛡️ CLAIMGUARD</h1>
            <p style="color: var(--text-secondary); margin: 0.25rem 0;">Evidence Audit Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        pages = [
            ("📤", "Upload Paper", "upload"),
            ("🏠", "Overview", "dashboard"),
            ("🔍", "Claims", "claims"),
            ("📊", "Evidence Graph", "evidence_graph"),
            ("📄", "Paper Viewer", "paper_viewer"),
            ("📋", "Final Report", "report"),
            ("⚙️", "Settings", "settings"),
        ]
        
        for icon, label, key in pages:
            is_active = st.session_state.get("page") == key
            active_class = "active" if is_active else ""
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if st.session_state.get("document"):
            st.caption("DOCUMENT INFO")
            doc = st.session_state.document
            st.caption(f"{doc.filename}")
            st.caption(f"{doc.num_pages} pages • {doc.word_count:,} words")


def render_landing_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">CLAIMGUARD AI</h1>
            <p style="font-size: 1.25rem; color: var(--accent-primary); margin-bottom: 2rem;">
                Verify the claim. Trace the evidence. Challenge the conclusion.
            </p>
            <p style="font-size: 1.1rem; color: var(--text-secondary); max-width: 600px; margin: 0 auto 3rem;">
                AI-powered evidence auditing for research papers. Upload a paper and discover 
                which claims are supported, overstated, contradicted, or lacking sufficient evidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze a Paper", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("📖 Try Sample Paper", use_container_width=True):
            from pathlib import Path
            # Try multiple possible locations
            candidates = [
                Path("sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf"),
                Path("ClaimGuard/sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf"),
                Path("../ClaimGuard/sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf"),
                Path(__file__).parent.parent.parent / "sample_data" / "A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf",
            ]
            sample_path = None
            for c in candidates:
                if c.exists():
                    sample_path = c.resolve()
                    break
            if sample_path and sample_path.exists():
                st.session_state.uploaded_file = str(sample_path)
                st.session_state.page = "processing"
                st.rerun()
            else:
                st.info("Sample paper not found in sample_data folder")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
            <div class="card">
                <h3 style="margin-top: 0;">🔍 Claim Extraction</h3>
                <p style="color: var(--text-secondary);">Automatically identify important factual and research claims using NLP and LLM analysis.</p>
            </div>
            <div class="card">
                <h3 style="margin-top: 0;">📚 Evidence Retrieval</h3>
                <p style="color: var(--text-secondary);">Find relevant evidence from the document and external academic sources via Semantic Scholar and Crossref.</p>
            </div>
            <div class="card">
                <h3 style="margin-top: 0;">🤖 AI Audit</h3>
                <p style="color: var(--text-secondary);">Use an LLM to reason over evidence and identify unsupported, overstated, or contradicted conclusions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_upload_page(run_analysis: Optional[Callable] = None):
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0.5rem;">Upload Research Document</h1>
        <p style="color: var(--text-secondary); max-width: 650px; margin: 0 auto;">
            Submit a research paper, preprint, or article to extract research claims, evaluate internal evidence, search external literature, and run an automated AI evidence audit.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_tab, text_tab, sample_tab = st.tabs(["📄 Upload File (PDF/DOCX/TXT)", "✍️ Direct Text Input", "📑 Sample Research Paper"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a research document (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"✅ Ready: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
            
            if file_size_mb > 50:
                st.error("File exceeds 50MB limit.")
            else:
                if st.button("🔬 Start Evidence Audit", type="primary", use_container_width=True, key="btn_upload_start"):
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    st.session_state.uploaded_file = tmp_path
                    st.session_state.processing_stage = None
                    st.session_state.progress_percentage = 0
                    st.session_state.progress_messages = []
                    st.session_state.page = "processing"
                    st.rerun()

    with text_tab:
        st.markdown("<p style='color: var(--text-secondary); margin-bottom: 0.5rem;'>Paste article text, abstract, or methodology below:</p>", unsafe_allow_html=True)
        text_input = st.text_area("Research Text", height=220, label_visibility="collapsed", placeholder="Paste paper text, key claims, experimental findings, or conclusion here...")
        if st.button("🔬 Audit Pasted Text", type="primary", use_container_width=True, key="btn_text_start"):
            if not text_input.strip() or len(text_input.strip().split()) < 15:
                st.warning("Please provide at least 15 words of text to extract meaningful claims.")
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                    tmp.write(text_input.strip())
                    tmp_path = tmp.name
                st.session_state.uploaded_file = tmp_path
                st.session_state.processing_stage = None
                st.session_state.progress_percentage = 0
                st.session_state.progress_messages = []
                st.session_state.page = "processing"
                st.rerun()

    with sample_tab:
        st.markdown("""
        <div class="card" style="margin-bottom: 1rem;">
            <h4 style="margin-top: 0;">A Unified LLM-Based Framework for Resume-Job Description Semantic Alignment</h4>
            <p style="color: var(--text-secondary); font-size: 0.95rem;">
                A published 12-page Springer computer science paper featuring 11 key research claims, empirical evaluation metrics, and ATS industry statistics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📖 Load Sample Paper & Run Audit", use_container_width=True, type="secondary", key="btn_sample_start"):
            from pathlib import Path
            candidates = [
                Path("sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf"),
                Path("ClaimGuard/sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf"),
                Path(__file__).parent.parent.parent / "sample_data" / "A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf",
            ]
            sample_path = next((c.resolve() for c in candidates if c.exists()), None)
            if sample_path:
                st.session_state.uploaded_file = str(sample_path)
                st.session_state.processing_stage = None
                st.session_state.progress_percentage = 0
                st.session_state.progress_messages = []
                st.session_state.page = "processing"
                st.rerun()
            else:
                st.error("Sample paper file not found in sample_data directory.")


def render_progress_pipeline(run_analysis_fn: Optional[Callable] = None):
    stages = [
        ("document_parsed", "Document Parsed"),
        ("claims_extracted", "Claims Extracted"),
        ("embeddings_built", "Embeddings Built"),
        ("internal_evidence_retrieved", "Internal Evidence Retrieved"),
        ("external_search_started", "External Search Started"),
        ("academic_search", "Academic Search"),
        ("web_search", "Web Search"),
        ("llm_analysis", "LLM Analysis"),
        ("report_generated", "Report Generated"),
    ]
    
    st.markdown("<h1 style='margin-bottom: 0.25rem;'>Evidence Auditing Pipeline</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 1.5rem;'>Autonomous multi-stage NLP & RAG pipeline executing claim extraction, retrieval, and LLM verification.</p>", unsafe_allow_html=True)

    current_stage = st.session_state.get("processing_stage", "")
    
    if current_stage == "error":
        st.error(f"Processing failed: {st.session_state.get('error_message', 'An error occurred during pipeline execution.')}")
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.processing_stage = None
            st.session_state.uploaded_file = None
            st.session_state.page = "upload"
            st.rerun()
        return

    progress_bar = st.progress(min(1.0, max(0.0, st.session_state.get("progress_percentage", 0) / 100.0)))
    status_caption = st.empty()
    steps_container = st.empty()
    
    def render_steps(curr_stg):
        html = '<div style="margin: 1.5rem 0;">'
        for stage_key, stage_name in stages:
            is_completed = _is_stage_completed(stages, stage_key, curr_stg)
            is_active = stage_key == curr_stg
            icon = "✓" if is_completed else ("●" if is_active else "○")
            status_class = "completed" if is_completed else ("active" if is_active else "pending")
            html += f'''
            <div class="progress-step {status_class}">
                <div class="progress-step-icon">{icon}</div>
                <div>{stage_name}</div>
            </div>'''
        html += '</div>'
        return html

    steps_container.markdown(render_steps(current_stage), unsafe_allow_html=True)
    
    logs_expander = st.expander("Processing Logs", expanded=True)
    logs_placeholder = logs_expander.empty()
    
    def update_logs(msgs):
        if msgs:
            log_lines = [f"**[{t.split('T')[-1][:8]}]** `{stg}`: {msg}" for stg, msg, t in msgs[-10:]]
            logs_placeholder.markdown("\n\n".join(log_lines))
        else:
            logs_placeholder.caption("Awaiting pipeline events...")

    update_logs(st.session_state.get("progress_messages", []))
    
    uploaded_file = st.session_state.get("uploaded_file")
    
    if current_stage == "report_generated":
        progress_bar.progress(1.0)
        status_caption.success("Audit complete! Report generated successfully.")
        if st.button("View Audit Results →", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        return

    if not uploaded_file:
        st.warning("No document selected for analysis.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        return

    # Automatically trigger analysis if not already running or completed
    if run_analysis_fn and current_stage not in ("report_generated", "error"):
        def live_update(stage: str, message: str, pct: float):
            progress_bar.progress(min(1.0, max(0.0, pct / 100.0)))
            status_caption.info(f"**Working:** {message}")
            steps_container.markdown(render_steps(stage), unsafe_allow_html=True)
            update_logs(st.session_state.get("progress_messages", []))
            
        with st.spinner("Analyzing document and auditing claims..."):
            run_analysis_fn(uploaded_file, live_callback=live_update)


def _is_stage_completed(stages, stage_key, current_stage):
    if current_stage == "report_generated":
        return True
    if not current_stage:
        return False
    current_idx = next((i for i, (k, _) in enumerate(stages) if k == current_stage), -1)
    target_idx = next((i for i, (k, _) in enumerate(stages) if k == stage_key), -1)
    return target_idx < current_idx


def render_dashboard():
    report = st.session_state.get("report")
    if not report:
        st.warning("No report available")
        return
    
    claims = report.claims
    analyses = report.analyses
    
    verdict_counts = {}
    for a in analyses:
        verdict_counts[a.verdict] = verdict_counts.get(a.verdict, 0) + 1
    
    st.markdown("<h1>Research Evidence Audit</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics = [
        ("Total Claims", len(claims)),
        ("Supported", verdict_counts.get("SUPPORTED", 0)),
        ("Partial", verdict_counts.get("PARTIALLY_SUPPORTED", 0)),
        ("Contradicted", verdict_counts.get("CONTRADICTED", 0)),
        ("Overstated", verdict_counts.get("OVERSTATED", 0)),
        ("Insufficient", verdict_counts.get("INSUFFICIENT_EVIDENCE", 0)),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4, col5, col6], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Evidence Distribution")
        fig = _create_verdict_chart(verdict_counts)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Document Statistics")
        doc = report.document
        st.metric("Pages", doc.num_pages)
        st.metric("Word Count", f"{doc.word_count:,}")
        st.metric("Sections", len(doc.sections))
        st.metric("External Sources", len(report.external_sources) + len(report.web_sources))
        st.metric("Processing Time", f"{report.processing_time:.1f}s" if hasattr(report, 'processing_time') else "N/A")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Recent Claims")
    for claim, analysis in zip(claims[:5], analyses[:5]):
        _render_claim_card(claim, analysis)


def _create_verdict_chart(verdict_counts):
    labels = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"]
    colors = ["#10b981", "#f59e0b", "#ef4444", "#ef4444", "#6b7280"]
    values = [verdict_counts.get(l, 0) for l in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=12,
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        showlegend=True,
        height=300,
        margin=dict(t=0, b=0, l=0, r=0),
    )
    return fig


def _render_claim_card(claim: Claim, analysis: AnalysisResult):
    badge_class = {
        "SUPPORTED": "badge-supported",
        "PARTIALLY_SUPPORTED": "badge-partial",
        "CONTRADICTED": "badge-contradicted",
        "OVERSTATED": "badge-overstated",
        "INSUFFICIENT_EVIDENCE": "badge-insufficient",
    }.get(analysis.verdict, "badge-insufficient")
    
    st.markdown(f"""
    <div class="claim-row">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <p style="margin: 0 0 0.5rem; font-size: 0.95rem;">{claim.text[:150]}...</p>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <span class="badge {badge_class}">{analysis.verdict}</span>
                    <span class="badge" style="background: var(--bg-secondary); border-color: var(--border-color);">{claim.category}</span>
                    <span class="badge" style="background: var(--bg-secondary); border-color: var(--border-color);">Confidence: {analysis.confidence:.0%}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_claims_explorer():
    claims = st.session_state.get("claims", [])
    analyses = st.session_state.get("analyses", [])
    
    if not claims:
        st.warning("No claims available")
        return
    
    st.markdown("<h1>Claims Explorer</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search claims", placeholder="Search by claim text...")
    with col2:
        verdict_filter = st.selectbox("Verdict", ["All"] + ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"])
    with col3:
        category_filter = st.selectbox("Category", ["All"] + sorted(set(c.category for c in claims)))
    
    filtered = []
    for claim, analysis in zip(claims, analyses):
        if search and search.lower() not in claim.text.lower():
            continue
        if verdict_filter != "All" and analysis.verdict != verdict_filter:
            continue
        if category_filter != "All" and claim.category != category_filter:
            continue
        filtered.append((claim, analysis))
    
    st.caption(f"Showing {len(filtered)} of {len(claims)} claims")
    
    for claim, analysis in filtered:
        badge_class = {
            "SUPPORTED": "badge-supported",
            "PARTIALLY_SUPPORTED": "badge-partial",
            "CONTRADICTED": "badge-contradicted",
            "OVERSTATED": "badge-overstated",
            "INSUFFICIENT_EVIDENCE": "badge-insufficient",
        }.get(analysis.verdict, "badge-insufficient")
        
        col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
        with col1:
            st.markdown(f"**{claim.text[:100]}...**")
            st.caption(f"Pages: {', '.join(str(s) for s in _get_pages(claim))} | {claim.category}")
        with col2:
            st.markdown(f'<span class="badge {badge_class}">{analysis.verdict}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f"**{analysis.confidence:.0%}**")
        with col4:
            st.markdown(f"**{len(analysis.supporting_evidence)}** 📚")
        with col5:
            if st.button("View", key=f"view_{claim.claim_id}"):
                st.session_state.selected_claim_id = claim.claim_id
                st.session_state.page = "claim_detail"
                st.rerun()
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def _get_pages(claim: Claim) -> List[int]:
    doc = st.session_state.get("document")
    if not doc or not hasattr(doc, "sentences"):
        return [1]
    pages = sorted(list(set(
        s.page for s in doc.sentences if s.sentence_id in claim.sentence_ids
    )))
    return pages if pages else [1]


def render_claim_detail():
    claim_id = st.session_state.get("selected_claim_id")
    claims = st.session_state.get("claims", [])
    analyses = st.session_state.get("analyses", [])
    internal_evidence = st.session_state.get("internal_evidence", {})
    external_sources = st.session_state.get("external_sources", [])
    web_sources = st.session_state.get("web_sources", [])
    
    claim = next((c for c in claims if c.claim_id == claim_id), None)
    analysis = next((a for a in analyses if a.claim_id == claim_id), None)
    
    if not claim or not analysis:
        st.error("Claim not found")
        return
    
    st.markdown(f"<h1>Claim Detail</h1>", unsafe_allow_html=True)
    
    badge_class = {
        "SUPPORTED": "badge-supported",
        "PARTIALLY_SUPPORTED": "badge-partial",
        "CONTRADICTED": "badge-contradicted",
        "OVERSTATED": "badge-overstated",
        "INSUFFICIENT_EVIDENCE": "badge-insufficient",
    }.get(analysis.verdict, "badge-insufficient")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {claim.text}")
    with col2:
        st.markdown(f'<span class="badge {badge_class}" style="font-size: 1rem;">{analysis.verdict}</span>', unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {analysis.confidence:.0%}")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("📝 Internal Evidence")
    evidence_list = internal_evidence.get(claim_id, [])
    if evidence_list:
        for ev in evidence_list:
            with st.expander(f"Page {ev.metadata.get('page', '?')} • Section: {ev.metadata.get('section', '?')} • Relevance: {ev.relevance_score:.2f}"):
                st.write(ev.text)
    else:
        st.info("No internal evidence found")
    
    st.subheader("📚 External Sources")
    if external_sources:
        for src in external_sources:
            tier_class = f"tier-{src.tier.lower()}"
            st.markdown(f"""
            <div class="source-card {tier_class}">
                <h4 style="margin: 0 0 0.5rem;">{src.title}</h4>
                <p style="margin: 0.25rem 0; color: var(--text-secondary);">
                    {', '.join(src.authors[:3])} | {src.year or 'Unknown'} | {src.venue or 'Unknown'}
                </p>
                <p style="margin: 0.25rem 0; font-size: 0.85rem;">Tier: {src.tier} | Citations: {src.citation_count}</p>
                {f'<p style="margin: 0.25rem 0;"><a href="{src.doi}" target="_blank">DOI: {src.doi}</a></p>' if src.doi else ''}
                {f'<p style="margin: 0.25rem 0;"><a href="{src.url}" target="_blank">View Paper</a></p>' if src.url else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No external academic sources found")
    
    if web_sources:
        st.subheader("🌐 Web Sources")
        for src in web_sources:
            st.markdown(f"""
            <div class="source-card">
                <h4 style="margin: 0 0 0.5rem;"><a href="{src.url}" target="_blank">{src.title}</a></h4>
                <p style="margin: 0.25rem 0; color: var(--text-secondary); font-size: 0.9rem;">{src.snippet}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("🤖 AI Analysis")
    st.markdown(f"**Reasoning:** {analysis.reasoning}")
    
    if analysis.suggested_revision:
        st.success(f"**Suggested Revision:** {analysis.suggested_revision}")
    
    if analysis.limitations:
        st.warning(f"**Limitations:** {analysis.limitations}")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("📊 Evidence Strength")
    _render_evidence_bars(analysis, evidence_list, external_sources)


def _render_evidence_bars(analysis: AnalysisResult, internal_evidence: List, external_sources: List):
    internal_score = min(len(internal_evidence) * 0.2, 1.0) if internal_evidence else 0
    external_score = min(len([a for a in analysis.supporting_evidence]) * 0.15, 1.0) if analysis.supporting_evidence else 0
    quality_score = sum(1 for s in external_sources if s.tier == "A") / max(len(external_sources), 1) if external_sources else 0
    conflict_score = len(analysis.contradicting_evidence) / max(len(analysis.supporting_evidence) + len(analysis.contradicting_evidence), 1) if analysis.supporting_evidence or analysis.contradicting_evidence else 0
    
    bars = [
        ("Internal Evidence", internal_score, "internal"),
        ("External Support", external_score, "external"),
        ("Source Quality", quality_score, "quality"),
        ("Evidence Conflict", conflict_score, "conflict"),
    ]
    
    for label, score, bar_class in bars:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>{label}</span>
                <span>{score:.0%}</span>
            </div>
            <div class="evidence-bar">
                <div class="evidence-bar-fill {bar_class}" style="width: {score*100}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_evidence_graph():
    import math
    import networkx as nx
    
    st.markdown("<h1>Interactive Evidence Graph</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: var(--text-secondary); margin-bottom: 1.5rem;'>"
        "Visualizing relationship networks between claims, internal evidence passages, document pages, and external sources."
        "</p>",
        unsafe_allow_html=True
    )

    claims = st.session_state.get("claims", [])
    analyses = st.session_state.get("analyses", [])
    internal_evidence = st.session_state.get("internal_evidence", {})
    external_sources = st.session_state.get("external_sources", [])

    if not claims:
        st.info("No audit data available. Please upload a research paper and run the analysis to generate the evidence graph.")
        return

    analysis_map = {a.claim_id: a for a in analyses}

    # Filtering controls
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        verdict_filter = st.selectbox(
            "Filter Verdict",
            ["All", "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"],
            key="graph_verdict_filter"
        )
    with col2:
        max_claims_shown = st.slider("Max Claims to Display", min_value=3, max_value=min(25, max(3, len(claims))), value=min(10, len(claims)))
    with col3:
        include_pages = st.checkbox("Show Page Nodes", value=True)

    filtered_claims = []
    for c in claims:
        a = analysis_map.get(c.claim_id)
        if not a:
            continue
        if verdict_filter != "All" and a.verdict != verdict_filter:
            continue
        filtered_claims.append(c)

    filtered_claims = filtered_claims[:max_claims_shown]

    if not filtered_claims:
        st.warning(f"No claims match the filter '{verdict_filter}'.")
        return

    # Build NetworkX graph
    G = nx.Graph()

    verdict_colors = {
        "SUPPORTED": "#10b981",
        "PARTIALLY_SUPPORTED": "#f59e0b",
        "CONTRADICTED": "#ef4444",
        "OVERSTATED": "#f43f5e",
        "INSUFFICIENT_EVIDENCE": "#64748b",
    }

    # Add Claim Nodes
    for c in filtered_claims:
        a = analysis_map.get(c.claim_id)
        verdict = a.verdict if a else "UNKNOWN"
        vcolor = verdict_colors.get(verdict, "#3b82f6")
        
        G.add_node(
            c.claim_id,
            node_type="claim",
            label=f"{c.claim_id}",
            title=f"<b>Claim {c.claim_id}</b><br>{c.text[:120]}...<br><b>Verdict:</b> {verdict}<br><b>Confidence:</b> {a.confidence:.0% if a else 0}%",
            color=vcolor,
            size=22,
        )

        # Internal evidence edges & nodes
        ev_list = internal_evidence.get(c.claim_id, [])
        for ev in ev_list[:2]:
            ev_node_id = f"EV_{ev.source_id}"
            if ev_node_id not in G:
                G.add_node(
                    ev_node_id,
                    node_type="internal_evidence",
                    label=f"Doc S.{ev.source_id}",
                    title=f"<b>Internal Evidence</b><br>Score: {ev.relevance_score:.2f}<br>{ev.text[:150]}...",
                    color="#06b6d4",
                    size=13,
                )
            G.add_edge(c.claim_id, ev_node_id, edge_type="SUPPORTED_BY", color="#10b981")

            # Page node
            if include_pages and ev.metadata.get("page"):
                pg_node_id = f"PAGE_{ev.metadata['page']}"
                if pg_node_id not in G:
                    G.add_node(
                        pg_node_id,
                        node_type="page",
                        label=f"P.{ev.metadata['page']}",
                        title=f"Document Page {ev.metadata['page']}",
                        color="#f59e0b",
                        size=10,
                    )
                G.add_edge(ev_node_id, pg_node_id, edge_type="LOCATED_ON", color="#64748b")

        # External sources
        if a:
            for s in a.supporting_evidence:
                sid = s.get("source_id")
                if sid:
                    ext_node_id = f"EXT_{sid}"
                    if ext_node_id not in G:
                        G.add_node(
                            ext_node_id,
                            node_type="external",
                            label=f"Ext:{sid[:8]}",
                            title=f"<b>External Supporting Source</b><br>{s.get('title', '')[:100]}",
                            color="#8b5cf6",
                            size=14,
                        )
                    G.add_edge(c.claim_id, ext_node_id, edge_type="SUPPORTED_BY", color="#10b981")

            for ce in a.contradicting_evidence:
                sid = ce.get("source_id")
                if sid:
                    ext_node_id = f"EXT_{sid}"
                    if ext_node_id not in G:
                        G.add_node(
                            ext_node_id,
                            node_type="external",
                            label=f"Ext:{sid[:8]}",
                            title=f"<b>Contradicting Source</b><br>{ce.get('title', '')[:100]}",
                            color="#ef4444",
                            size=14,
                        )
                    G.add_edge(c.claim_id, ext_node_id, edge_type="CONTRADICTED_BY", color="#ef4444")

    # Layout
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(148, 163, 184, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        data = G.nodes[node]
        node_text.append(data.get("label", ""))
        node_hover.append(data.get("title", ""))
        node_color.append(data.get("color", "#3b82f6"))
        node_size.append(data.get("size", 14))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color="#0f172a")
        ),
        textfont=dict(size=10, color="#f1f5f9")
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            height=580,
        )
    )

    # Top graph stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Claims Visualized", len(filtered_claims))
    with c2:
        st.metric("Total Graph Nodes", len(G.nodes()))
    with c3:
        st.metric("Evidence Relationships", len(G.edges()))
    with c4:
        st.metric("External Corroborations", sum(1 for n in G.nodes() if str(n).startswith("EXT_")))

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center; padding: 0.75rem; background: #1e293b; border-radius: 8px; border: 1px solid #334155; font-size: 0.85rem;">
        <div><span style="color: #10b981; font-size: 1.1rem;">●</span> Claim (Supported)</div>
        <div><span style="color: #f59e0b; font-size: 1.1rem;">●</span> Claim (Partial)</div>
        <div><span style="color: #f43f5e; font-size: 1.1rem;">●</span> Claim (Overstated)</div>
        <div><span style="color: #ef4444; font-size: 1.1rem;">●</span> Claim (Contradicted)</div>
        <div><span style="color: #64748b; font-size: 1.1rem;">●</span> Claim (Insufficient)</div>
        <div><span style="color: #06b6d4; font-size: 1.1rem;">●</span> Internal Evidence Passage</div>
        <div><span style="color: #8b5cf6; font-size: 1.1rem;">●</span> External Academic Paper</div>
        <div><span style="color: #f59e0b; font-size: 1.1rem;">●</span> Document Page</div>
    </div>
    """, unsafe_allow_html=True)


def render_paper_viewer():
    document = st.session_state.get("document")
    claims = st.session_state.get("claims", [])
    
    if not document:
        st.warning("No document loaded")
        return
    
    st.markdown("<h1>Paper Viewer</h1>", unsafe_allow_html=True)
    
    page_num = st.number_input("Page", min_value=1, max_value=document.num_pages, value=1)
    
    page_sentences = [s for s in document.sentences if s.page == page_num]
    
    for sent in page_sentences:
        is_claim = any(sent.sentence_id in c.sentence_ids for c in claims)
        if is_claim:
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid var(--accent-primary); padding: 0.75rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                <small style="color: var(--accent-primary);">CLAIM • {sent.section}</small><br>
                {sent.text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                <small style="color: var(--text-muted);">Page {sent.page} • {sent.section}</small><br>
                {sent.text}
            </div>
            """, unsafe_allow_html=True)


def render_final_report():
    report = st.session_state.get("report")
    
    if not report:
        st.warning("No report available")
        return
    
    st.markdown("<h1>Final Audit Report</h1>", unsafe_allow_html=True)
    
    markdown_report = report.to_markdown()
    st.markdown(markdown_report)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Download Markdown",
            data=markdown_report,
            file_name=f"audit_report_{report.document.filename}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        try:
            pdf_data = report.to_pdf()
            st.download_button(
                "📥 Download PDF Report",
                data=pdf_data,
                file_name=f"audit_report_{report.document.filename}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")
    with col3:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(report.to_json(), indent=2),
            file_name=f"audit_report_{report.document.filename}.json",
            mime="application/json",
            use_container_width=True
        )


def render_settings():
    config = get_config()
    config_yaml = _load_config_yaml(config)

    st.markdown("<h1>Settings</h1>", unsafe_allow_html=True)
    st.caption("Read-only view of the active configuration")

    _render_copy_config_button(config_yaml)

    sections = [
        ("LLM Configuration", {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        }),
        ("Embedding Configuration", {
            "model": config.embedding.model,
            "dimension": config.embedding.dimension,
        }),
        ("Retrieval Parameters", {
            "top_k": config.retrieval.top_k,
            "similarity_threshold": config.retrieval.similarity_threshold,
            "max_search_results": config.retrieval.max_search_results,
            "internal_top_k": config.retrieval.internal_top_k,
        }),
        ("Analysis Limits", {
            "max_claims": config.analysis.max_claims,
            "min_importance_score": config.analysis.min_importance_score,
        }),
        ("Source Ranking Weights", dict(config.source_ranking.weights)),
        ("Cache Settings", {
            "enabled": config.cache.enabled,
            "ttl_hours": config.cache.ttl_hours,
        }),
        ("File Upload", {
            "max_size_mb": config.file_upload.max_size_mb,
            "allowed_extensions": list(config.file_upload.allowed_extensions),
        }),
        ("API Endpoints", {
            "semantic_scholar": dict(config.api.semantic_scholar),
            "crossref": dict(config.api.crossref),
            "web_search": dict(config.api.web_search),
        }),
    ]

    for title, payload in sections:
        with st.expander(title, expanded=False):
            st.json(payload)

    with st.expander("Raw config/config.yaml", expanded=False):
        st.code(config_yaml, language="yaml")

    st.info("Changes require editing config/config.yaml and restarting the app")


def _load_config_yaml(config) -> str:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")
    return config.model_dump_json(indent=2)


def _render_copy_config_button(config_yaml: str):
    payload = json.dumps(config_yaml)
    content = f"""
<div style="display: flex; justify-content: flex-end; align-items: center; gap: 0.5rem;">
    <button id="cg-copy-btn" style="
        padding: 0.35rem 0.9rem; border-radius: 8px; cursor: pointer;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(148, 163, 184, 0.12); color: inherit;
        font: inherit; font-size: 0.85rem;">
        Copy Config
    </button>
    <span id="cg-copy-status" style="font-size: 0.8rem; opacity: 0.8;"></span>
</div>
<script>
    const btn = document.getElementById("cg-copy-btn");
    const status = document.getElementById("cg-copy-status");
    btn.addEventListener("click", async () => {{
        const text = {payload};
        try {{
            await navigator.clipboard.writeText(text);
            status.textContent = "Copied!";
        }} catch (err) {{
            const area = document.createElement("textarea");
            area.value = text;
            document.body.appendChild(area);
            area.select();
            let copied = false;
            try {{ copied = document.execCommand("copy"); }} catch (e) {{ copied = false; }}
            document.body.removeChild(area);
            status.textContent = copied ? "Copied!" : "Copy failed - copy the YAML below";
        }}
        setTimeout(() => {{ status.textContent = ""; }}, 2500);
    }});
</script>
"""
    if hasattr(st, "iframe"):
        st.iframe(content, height=45)
    else:
        components.html(content, height=45)


# Legacy functions for compatibility
def render_claims_explorer_legacy():
    return render_claims_explorer()


def render_claim_detail_legacy():
    return render_claim_detail()


def render_evidence_graph_legacy():
    return render_evidence_graph()


def render_paper_viewer_legacy():
    return render_paper_viewer()


def render_final_report_legacy():
    return render_final_report()


def render_progress_pipeline_legacy():
    return render_progress_pipeline()