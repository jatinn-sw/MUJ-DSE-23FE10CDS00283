CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1e293b;
    --bg-card-hover: #253047;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-primary: #3b82f6;
    --accent-secondary: #8b5cf6;
    --accent-glow: rgba(59, 130, 246, 0.3);
    --border-color: #334155;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
    --gray: #6b7280;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

header[data-testid="stHeader"] {
    background-color: transparent;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.625rem 1.5rem;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px var(--accent-glow);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px var(--accent-glow);
}

.stButton > button:focus {
    outline: 2px solid var(--accent-primary);
    outline-offset: 2px;
}

.stButton > button[kind="secondary"] {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    box-shadow: none;
}

.stButton > button[kind="secondary"]:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-primary);
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.2s ease;
}

.card:hover {
    border-color: var(--accent-primary);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-supported { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-partial { background: rgba(245, 158, 11, 0.15); color: var(--yellow); border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-contradicted { background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-overstated { background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-insufficient { background: rgba(107, 114, 128, 0.15); color: var(--gray); border: 1px solid rgba(107, 114, 128, 0.3); }

.evidence-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--border-color);
    overflow: hidden;
}

.evidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.evidence-bar-fill.internal { background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); }
.evidence-bar-fill.external { background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary)); }
.evidence-bar-fill.quality { background: var(--green); }
.evidence-bar-fill.conflict { background: var(--red); }

.claim-row {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.claim-row:hover {
    border-color: var(--accent-primary);
    background: var(--bg-card-hover);
}

.claim-row.selected {
    border-color: var(--accent-primary);
    background: rgba(59, 130, 246, 0.05);
}

.progress-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 0;
    color: var(--text-secondary);
}

.progress-step.active {
    color: var(--accent-primary);
}

.progress-step.completed {
    color: var(--green);
}

.progress-step-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
}

.progress-step.completed .progress-step-icon {
    background: var(--green);
    color: white;
}

.progress-step.active .progress-step-icon {
    background: var(--accent-primary);
    color: white;
    box-shadow: 0 0 0 3px var(--accent-glow);
}

.progress-step.pending .progress-step-icon {
    background: var(--border-color);
    color: var(--text-muted);
}

.upload-zone {
    border: 2px dashed var(--border-color);
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    transition: all 0.2s ease;
    background: var(--bg-secondary);
}

.upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent-primary);
    background: rgba(59, 130, 246, 0.05);
}

.source-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.source-card.tier-a { border-left: 4px solid var(--green); }
.source-card.tier-b { border-left: 4px solid var(--yellow); }
.source-card.tier-c { border-left: 4px solid var(--gray); }

.sidebar-nav {
    padding: 1rem 0;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    color: var(--text-secondary);
    text-decoration: none;
    transition: all 0.2s ease;
    margin-bottom: 0.25rem;
}

.nav-item:hover, .nav-item.active {
    background: var(--bg-card);
    color: var(--accent-primary);
}

.divider {
    height: 1px;
    background: var(--border-color);
    margin: 1.5rem 0;
}

.empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.stDataFrame {
    background: var(--bg-card);
    border-radius: 8px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-secondary);
    padding: 0.75rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: white !important;
    border-color: var(--accent-primary) !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border-color) !important;
}

.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

.stFileUploader > div {
    background: transparent !important;
}

.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

.stExpander summary {
    color: var(--text-primary) !important;
}

@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .metric-value {
        font-size: 2rem;
    }
}
</style>
"""

def inject_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)