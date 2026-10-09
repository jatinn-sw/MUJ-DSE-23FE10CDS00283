import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from app.frontend.styles import inject_css
from app.frontend.components import (
    render_sidebar, render_landing_page, render_upload_page,
    render_dashboard, render_claims_explorer, render_claim_detail,
    render_evidence_graph, render_paper_viewer, render_final_report,
    render_progress_pipeline, render_settings
)
from app.pipeline import run_pipeline
from app.pdf_parser import parse_document
from app.config import get_config
from app.utils import setup_logging

setup_logging()

st.set_page_config(
    page_title="ClaimGuard AI - Evidence Auditing Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

def init_session_state():
    defaults = {
        "page": "landing",
        "document": None,
        "claims": [],
        "analyses": [],
        "external_sources": [],
        "web_sources": [],
        "internal_evidence": {},
        "report": None,
        "processing_stage": None,
        "progress_percentage": 0,
        "uploaded_file": None,
        "selected_claim_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def progress_callback(stage: str, message: str, percentage: float):
    st.session_state.processing_stage = stage
    st.session_state.progress_percentage = percentage
    if "progress_messages" not in st.session_state:
        st.session_state.progress_messages = []
    st.session_state.progress_messages.append((stage, message, datetime.now().isoformat()))

def run_analysis(file_path: str, live_callback=None):
    st.session_state.processing_stage = "document_parsed"
    st.session_state.progress_percentage = 5
    st.session_state.progress_messages = []
    
    def on_progress(stage: str, message: str, percentage: float):
        progress_callback(stage, message, percentage)
        if live_callback:
            try:
                live_callback(stage, message, percentage)
            except Exception:
                pass
                
    try:
        report = run_pipeline(file_path, on_progress)
        st.session_state.report = report
        st.session_state.document = report.document
        st.session_state.claims = report.claims
        st.session_state.analyses = report.analyses
        st.session_state.external_sources = report.external_sources
        st.session_state.web_sources = report.web_sources
        st.session_state.internal_evidence = getattr(report, "internal_evidence", {})
        st.session_state.processing_stage = "report_generated"
        st.session_state.progress_percentage = 100
        st.session_state.page = "dashboard"
        st.rerun()
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.session_state.processing_stage = "error"
        st.session_state.error_message = str(e)
        st.error(f"Analysis failed: {str(e)}")

def main():
    init_session_state()
    
    render_sidebar()
    
    page = st.session_state.page
    
    if page == "landing":
        render_landing_page()
    elif page == "upload":
        render_upload_page(run_analysis)
    elif page == "processing":
        render_progress_pipeline(run_analysis)
    elif page == "dashboard":
        render_dashboard()
    elif page == "claims":
        render_claims_explorer()
    elif page == "claim_detail":
        render_claim_detail()
    elif page == "evidence_graph":
        render_evidence_graph()
    elif page == "paper_viewer":
        render_paper_viewer()
    elif page == "report":
        render_final_report()
    elif page == "settings":
        render_settings()

if __name__ == "__main__":
    main()