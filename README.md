# ClaimGuard AI

**Verify the claim. Trace the evidence. Challenge the conclusion.**

ClaimGuard AI is an evidence-auditing platform that automatically identifies factual and research claims from academic documents, retrieves relevant evidence from both the uploaded document and external academic sources, and uses an LLM to assess whether claims are adequately supported.

## 🎯 Problem Statement

Research papers often contain claims that are overstated, contradictory, or lack sufficient evidence. Traditional peer review is slow and inconsistent. ClaimGuard AI provides automated, transparent evidence auditing that:

- **Extracts** verifiable claims from documents using NLP + LLM hybrid approach
- **Retrieves** internal evidence (from the document) and external evidence (from academic databases)
- **Analyzes** evidence using structured LLM reasoning with hallucination protection
- **Produces** audit reports with traceable evidence chains for every verdict

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLAIMGUARD AI PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  UPLOAD          DOCUMENT PROCESSING        CLAIM EXTRACTION
  ──────          ───────────────────        ──────────────────
  ▼               ▼                          ▼
┌──────┐      ┌──────────┐             ┌──────────────┐
│ PDF  │─────▶│ PyMuPDF  │────────────▶│ Candidate    │
│/DOCX/│      │ Extract  │             │ Filtering    │
│ TXT  │      │ Sentences│             │ (Regex + NER)│
└──────┘      └──────────┘             └──────┬───────┘
                                               ▼
                                    ┌──────────────────────┐
                                    │ LLM Claim Extraction │
                                    │ (Structured Output)  │
                                    └──────────┬───────────┘
                                               ▼
                              ┌────────────────────────────┐
                              │ Claim Classification &     │
                              │ Importance Scoring         │
                              └──────────────┬─────────────┘
                                             ▼
  INTERNAL EVIDENCE          EXTERNAL EVIDENCE        LLM ANALYSIS
  ──────────────────         ────────────────         ─────────────
  ▼                          ▼                         ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Embeddings   │        │ Query Gen    │        │ Evidence     │
│ (OpenAI +    │        │ (LLM)        │        │ Analysis     │
│  FAISS)      │        └──────┬───────┘        │ (LLM + RAG)  │
│ Semantic     │               ▼                │ Structured   │
│ Search       │        ┌──────────────┐        │ JSON Output  │
│ Top-K        │        │ Semantic     │        └──────┬───────┘
└──────────────┘        │ Scholar API  │               ▼
                        │ Crossref API │        ┌──────────────┐
                        │ SerpAPI/     │        │ Claim Verdict│
                        │ Google CSE   │        │ + Confidence │
                        └──────────────┘        └──────┬───────┘
                                                       ▼
                                              ┌──────────────┐
                                              │ Conflict     │
                                              │ Detection    │
                                              └──────┬───────┘
                                                     ▼
                                              ┌──────────────┐
                                              │ Final Audit  │
                                              │ Report       │
                                              │ (MD + JSON)  │
                                              └──────────────┘
```

## 🛠️ Technology Stack

### Core Framework
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Streamlit | ≥1.35.0 | Interactive web UI |
| **UI Styling** | Custom CSS + streamlit-option-menu | ≥0.3.6 | Dark navy research-lab aesthetic |
| **Config** | Pydantic + Pydantic-Settings + PyYAML | ≥2.7.0 / ≥2.3.0 / ≥6.0.1 | Type-safe configuration management |
| **Async/HTTP** | httpx | ≥0.27.0 | Async API clients with retry logic |

### NLP & ML Stack
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Core NLP** | spaCy | ≥3.7.2 | NER, sentence segmentation, POS tagging |
| **Tokenization** | NLTK | ≥3.8.1 | Sentence tokenization (punkt) |
| **Transformers** | HuggingFace Transformers | ≥4.40.0 | Model utilities, tokenizers |
| **PyTorch** | PyTorch | ≥2.3.0 | Deep learning backend |
| **Sentence Embeddings** | sentence-transformers | ≥3.0.0 | Local embedding models (optional) |
| **ML Utilities** | scikit-learn | ≥1.4.0 | Metrics, preprocessing, evaluation |

### Document Processing
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **PDF Parsing** | PyMuPDF (fitz) | ≥1.23.0 | Text extraction, page/layout analysis, tables |
| **DOCX Parsing** | python-docx | ≥1.1.0 | Word document text extraction |

### Embeddings & Vector Search
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Vector DB** | FAISS-CPU | ≥1.8.0 | Fast similarity search (IndexFlatIP) |
| **Numerical** | NumPy | ≥1.26.0 | Vector operations, normalization |
| **Embeddings API** | OpenAI Embeddings / Google Gemini | API | text-embedding-3-small (1536-dim) / text-embedding-004 (768-dim) |

### External APIs
| API | Purpose | Authentication |
|-----|---------|----------------|
| **OpenAI API** | LLM (GPT-4o-mini) + Embeddings | `OPENAI_API_KEY` |
| **Anthropic API** | Alternative LLM (Claude) | `ANTHROPIC_API_KEY` |
| **Google Gemini API** | LLM (Gemini 1.5 Flash/Pro) + Embeddings | `GEMINI_API_KEY` |
| **Semantic Scholar** | Academic paper search (titles, abstracts, citations) | `SEMANTIC_SCHOLAR_API_KEY` (optional) |
| **Crossref** | DOI metadata, publication info | `CROSSREF_API_KEY` (optional) |
| **SerpAPI** | Google web search | `SERPAPI_API_KEY` |
| **Google CSE** | Custom Search Engine fallback | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` |

### Visualization & Reporting
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Charts** | Plotly | ≥5.21.0 | Interactive evidence distribution charts |
| **Graphs** | PyVis | ≥0.3.2 | Interactive network graphs (planned) |
| **PDF Generation** | WeasyPrint | ≥61.0 | HTML→PDF report export |
| **ReportLab** | ReportLab | ≥4.2.0 | Programmatic PDF generation |
| **Markdown** | Python-Markdown | ≥3.6.0 | Report formatting |

### Data Processing
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **DataFrames** | Pandas | ≥2.2.0 | Tabular data manipulation |
| **Graphs** | NetworkX | ≥3.2.1 | Evidence graph construction |

### Testing & Quality
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Testing** | pytest | ≥8.2.0 | Unit + integration tests |
| **Async Testing** | pytest-asyncio | ≥0.23.0 | Async test support |
| **Coverage** | pytest-cov | ≥4.1.0 | Code coverage reports |
| **Progress** | tqdm | ≥4.66.0 | Progress bars for batch ops |
| **Rich Console** | rich | ≥13.7.0 | Beautiful terminal output |

### Caching & Utilities
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Cache** | Pickle (local) / Redis (optional) | Document embeddings, search results |
| **Retry Logic** | tenacity | ≥8.2.0 | Exponential backoff for API calls |
| **ID Generation** | hashlib (MD5) | Deterministic unique IDs |
| **Logging** | Python stdlib (JSON format) | Structured logging |

---

## ⚙️ Pipeline Stages (Detailed)

### Stage 1: Document Processing (`app/pdf_parser.py`)
- **Input**: PDF/DOCX/TXT file
- **Operations**:
  - Text extraction with PyMuPDF (preserves page numbers)
  - Section/heading detection via regex patterns
  - Sentence segmentation with NLTK
  - Reference/bibliography detection
  - Table extraction (optional)
  - Boilerplate removal (figure captions, table headers, references)
- **Output**: `Document` object with `Sentence[]` (each with `sentence_id`, `page`, `section`, `text`)

### Stage 2: Claim Extraction (`app/claim_extractor.py`)
- **Input**: `Document.sentences`
- **Operations**:
  1. **Candidate Filtering** (local NLP): Regex patterns to skip boilerplate, find claim indicators (%, comparative, causal, performance language)
  2. **LLM Extraction**: Structured prompt → OpenAI JSON output with claims, categories, importance
  3. **Fallback**: Heuristic extraction if LLM fails
  4. **Classification & Scoring**: spaCy NER + rule-based importance calculation
- **Output**: `Claim[]` (claim_id, text, sentence_ids, category, importance)

### Stage 3: Embedding Building (`app/embeddings.py`)
- **Input**: `Document.sentences`
- **Operations**:
  - Batch embedding via OpenAI `text-embedding-3-small`
  - FAISS IndexFlatIP with L2 normalization
  - Local cache (pickle) keyed by document hash
- **Output**: `EmbeddingManager` with searchable index

### Stage 4: Internal Evidence Retrieval (`app/evidence_retriever.py`)
- **Input**: `Claim`, `EmbeddingManager`
- **Operations**:
  - Embed claim text
  - Cosine similarity search (top-k, threshold from config)
  - Return `Evidence[]` with relevance scores, page, section
- **Output**: Internal evidence per claim

### Stage 5: Query Generation (`app/query_generator.py`)
- **Input**: `Claim`, category
- **Operations**:
  - LLM prompt to generate 3-5 optimized academic search queries
  - Fallback: key term extraction
- **Output**: `List[str]` queries per claim

### Stage 6: External Evidence Search (`app/academic_search.py`, `app/web_search.py`)
- **Input**: Queries per claim
- **Operations**:
  - **Semantic Scholar**: Paper search (title, abstract, authors, year, venue, citations, DOI)
  - **Crossref**: DOI metadata (title, authors, journal, year)
  - **Web Search**: SerpAPI or Google CSE (snippets, URLs)
  - Deduplication by DOI
  - Source tier classification (A/B/C based on venue + citations)
- **Output**: `AcademicSource[]`, `WebSource[]`

### Stage 7: LLM Evidence Analysis (`app/llm_analyzer.py`)
- **Input**: `Claim`, internal evidence, external sources, web sources
- **Operations**:
  - Format evidence with source IDs, metadata
  - Structured prompt with strict hallucination protection rules
  - JSON output: verdict, confidence, supporting/contradicting evidence, reasoning, limitations, suggested revision
- **Output**: `AnalysisResult`

### Stage 8: Report Generation (`app/report_generator.py`)
- **Input**: All pipeline outputs
- **Operations**:
  - Aggregate statistics
  - Verdict distribution
  - Per-claim details grouped by verdict
  - Evidence conflict detection
  - Source quality analysis (tier distribution, recency)
  - Methodological concerns + recommendations
- **Output**: `AuditReport` (Markdown + JSON export)

---

## 🔄 Workflows

### 1. Main User Flow (Web UI)
```
Landing Page → Upload → Processing Pipeline → Dashboard → Claims Explorer → Claim Detail / Evidence Graph / Paper Viewer → Final Report
```

### 2. Batch Processing (CLI/Script)
```python
from app.pipeline import run_pipeline

report = run_pipeline("paper.pdf")
print(report.to_markdown())
# or
report.to_json()  # for programmatic use
```

### 3. Evaluation Workflow
```
evaluation/annotations.json (human labels) 
    → evaluation/evaluate.py 
    → metrics: accuracy, precision, recall, F1, confusion matrix
```

---

## 📁 Project Structure

```
ClaimGuard/
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic config models + YAML loader
│   ├── pdf_parser.py          # Document → Sentences (PyMuPDF)
│   ├── claim_extractor.py     # Sentences → Claims (NLP + LLM)
│   ├── embeddings.py          # Embeddings + FAISS index + cache
│   ├── evidence_retriever.py  # Internal evidence retrieval
│   ├── query_generator.py     # Claim → Search queries (LLM)
│   ├── academic_search.py     # Semantic Scholar + Crossref clients
│   ├── web_search.py          # SerpAPI + Google CSE clients
│   ├── llm_analyzer.py        # Evidence → Verdict (LLM + RAG)
│   ├── report_generator.py    # AuditReport (Markdown + JSON)
│   ├── utils.py               # Helpers: logging, IDs, JSON, retry, NLP utils
│   ├── pipeline.py            # Orchestrates all stages
│   └── frontend/
│       ├── __init__.py
│       ├── main.py            # Streamlit app entry point
│       ├── styles.py          # Custom CSS (dark navy theme)
│       └── components.py      # All UI components
├── prompts/
│   ├── claim_extraction.txt   # LLM prompt: extract claims from sentences
│   ├── claim_classification.txt  # LLM prompt: classify claim category
│   ├── query_generation.txt   # LLM prompt: generate search queries
│   ├── evidence_analysis.txt  # LLM prompt: analyze evidence → verdict
│   └── final_report.txt       # LLM prompt: generate final report (unused, code-based)
├── config/
│   └── config.yaml            # All configurable parameters
├── evaluation/
│   ├── annotations.json       # Human-labeled ground truth
│   ├── evaluate.py            # Evaluation script
│   └── metrics.py             # Metric calculations
├── tests/
│   ├── test_parser.py
│   ├── test_claims.py
│   ├── test_retrieval.py
│   └── test_llm.py
├── sample_data/               # Sample papers for testing
├── cache/                     # Embedding cache (auto-created)
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- **LLM API Key (choose one):**
  - OpenAI API key (`OPENAI_API_KEY`)
  - Anthropic API key (`ANTHROPIC_API_KEY`)  
  - **Google Gemini API key (`GEMINI_API_KEY`)** - *new!*
- Optional: Semantic Scholar API key, SerpAPI key, Google CSE credentials

### Quick Start
```bash
# Clone and enter project
cd ClaimGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app/frontend/main.py
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔧 Configuration (`config/config.yaml`)

```yaml
llm:
  provider: gemini          # openai | anthropic | gemini
  model: gemini-1.5-flash   # gpt-4o, gpt-4o-mini, claude-3-haiku, gemini-1.5-flash, gemini-1.5-pro
  temperature: 0            # 0 for deterministic analysis
  max_tokens: 4000

embedding:
  model: text-embedding-004  # text-embedding-3-small (OpenAI) | text-embedding-004 (Gemini)
  dimension: 768             # 1536 for OpenAI small, 3072 for OpenAI large, 768 for Gemini

retrieval:
  top_k: 5                  # Internal evidence per claim
  similarity_threshold: 0.65
  max_search_results: 10    # External search results per query
  internal_top_k: 10        # Internal candidates before threshold

analysis:
  max_claims: 30            # Max claims to process
  min_importance_score: 0.5 # Minimum importance for external search

source_ranking:
  weights:
    semantic_similarity: 0.50
    keyword_similarity: 0.20
    source_quality: 0.15
    recency: 0.15

cache:
  enabled: true
  ttl_hours: 24

file_upload:
  max_size_mb: 50
  allowed_extensions: [".pdf", ".docx", ".txt"]

api:
  semantic_scholar:
    base_url: https://api.semanticscholar.org/graph/v1
    rate_limit: 100
  crossref:
    base_url: https://api.crossref.org
    rate_limit: 50
  web_search:
    provider: serpapi       # serpapi | google_cse
    base_url: https://serpapi.com/search
    rate_limit: 100

logging:
  level: INFO
  format: json

evaluation:
  dataset_path: evaluation/annotations.json
  metrics: [accuracy, precision, recall, f1_score, confusion_matrix]
```

---

## 🔑 Environment Variables (`.env`)

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Academic Search (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=...
CROSSREF_API_KEY=...  # Optional, increases rate limit

# Web Search (at least one required for web evidence)
SERPAPI_API_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...

# Optional: Redis for distributed caching
REDIS_URL=redis://localhost:6379/0
```

---

## 📖 Usage Examples

### Web Interface
```bash
streamlit run app/frontend/main.py
```
1. Open http://localhost:8501
2. Click "Analyze a Paper"
3. Upload PDF/DOCX/TXT
4. Wait for pipeline to complete
5. Explore: Dashboard → Claims → Evidence Graph → Paper Viewer → Report

### Programmatic API
```python
from app.pipeline import run_pipeline
from app.pdf_parser import parse_document
from app.claim_extractor import extract_claims

# Full pipeline
report = run_pipeline("research_paper.pdf")
print(report.to_markdown())

# Or individual stages
document = parse_document("paper.pdf")
claims = extract_claims(document.sentences)
print(f"Found {len(claims)} claims")
for claim in claims:
    print(f"  [{claim.category}] {claim.text[:80]}... (importance: {claim.importance:.2f})")
```

### Batch Processing Script
```python
# batch_process.py
import json
from pathlib import Path
from app.pipeline import run_pipeline

papers_dir = Path("papers/")
output_dir = Path("reports/")
output_dir.mkdir(exist_ok=True)

for pdf_path in papers_dir.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    report = run_pipeline(str(pdf_path))
    
    # Save markdown
    (output_dir / f"{pdf_path.stem}_report.md").write_text(report.to_markdown())
    # Save JSON
    (output_dir / f"{pdf_path.stem}_report.json").write_text(json.dumps(report.to_json(), indent=2))
    
    print(f"  ✓ {len(report.claims)} claims analyzed")
```

---

## 📊 Evaluation

### Ground Truth Format (`evaluation/annotations.json`)
```json
[
  {
    "claim_id": "C_ABC123",
    "claim_text": "Our model achieves 94.2% accuracy.",
    "human_verdict": "SUPPORTED",
    "notes": "Table 2 shows 94.2% on test set"
  }
]
```

### Running Evaluation
```bash
cd ClaimGuard
python -m evaluation.evaluate
```

### Metrics Computed
| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct verdicts / total |
| **Precision (per class)** | TP / (TP + FP) for each verdict |
| **Recall (per class)** | TP / (TP + FN) for each verdict |
| **F1-Score (per class)** | Harmonic mean of precision/recall |
| **Confusion Matrix** | Verdict prediction vs ground truth |
| **Processing Time** | Avg seconds per document |
| **API Calls** | LLM + embedding + search calls per doc |
| **Tokens/Claim** | Average token usage |
| **Retrieval Latency** | Search API response times |
| **JSON Parse Success** | % of valid LLM JSON outputs |

---

## 🎨 Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| **Landing** | `/` | Hero, features, CTA |
| **Upload** | `/upload` | Drag-drop, file preview, start audit |
| **Processing** | `/processing` | Real-time pipeline progress with logs |
| **Dashboard** | `/dashboard` | Stats cards, verdict pie chart, recent claims |
| **Claims Explorer** | `/claims` | Searchable, filterable table (verdict, category) |
| **Claim Detail** | `/claim_detail` | Full evidence, AI reasoning, suggested revision, evidence bars |
| **Evidence Graph** | `/evidence_graph` | Interactive network (PyVis) - *planned* |
| **Paper Viewer** | `/paper_viewer` | Page-by-page with claim highlighting |
| **Final Report** | `/report` | Full markdown report + PDF/JSON export |
| **Settings** | `/settings` | Configuration UI - *planned* |

---

## 🔬 Key Design Principles

### 1. Evidence Traceability
Every verdict traces back to specific sources:
```
Claim → Internal Evidence (page, section) → External Sources (DOI, URL) → LLM Reasoning → Verdict
```

### 2. No Hallucination Guarantee
LLM prompt enforces:
- Never invent sources/citations
- Never use internal knowledge as evidence
- Only reason over retrieved evidence
- Explicitly state "insufficient evidence" when appropriate

### 3. Scientific Rigor
- **SUPPORTED ≠ PROVEN TRUE**: Only supported by *available* evidence
- **INSUFFICIENT EVIDENCE ≠ FALSE**: Cannot determine, not a negative finding
- **OVERSTATED**: Evidence exists but claim wording exceeds evidence strength

### 4. Efficiency First
- Local NLP filtering before LLM calls
- Embedding cache (24hr TTL)
- Configurable limits: max_claims, top_k, max_search_results
- Batch embedding API calls

### 5. Graceful Degradation
- LLM fallback to heuristic extraction
- Search API failures don't crash pipeline
- Missing API keys → skip that source type with warning

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test module
pytest tests/test_parser.py -v

# Run with async support
pytest tests/ -v --asyncio-mode=auto
```

### Test Coverage Areas
- `test_parser.py`: PDF extraction, sentence splitting, section detection
- `test_claims.py`: Candidate filtering, classification, importance scoring
- `test_retrieval.py`: FAISS search, embedding cache, external search mocking
- `test_llm.py`: JSON parsing, prompt formatting, verdict validation

---

## 🚧 Limitations & Future Work

### Current Limitations
1. **Single-document only**: No cross-document analysis
2. **English only**: spaCy `en_core_web_sm` + OpenAI embeddings
3. **No real-time collaboration**: Single-user session state
4. **Limited graph viz**: PyVis integration incomplete
5. **No auth/multi-tenancy**: Designed for local/research use

### Planned Improvements
- [ ] Multi-document comparison mode
- [ ] PubMed API integration for biomedical domain
- [ ] Local LLM support (Ollama, vLLM) for privacy
- [ ] Redis-backed distributed caching
- [ ] User accounts + project persistence
- [ ] REST API for programmatic access
- [ ] Advanced evidence graph with PyVis/NetworkX
- [ ] Citation network analysis
- [ ] Claim clustering (duplicate/similar detection)
- [ ] Active learning: human feedback → improved prompts

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **Semantic Scholar API** for academic paper metadata
- **Crossref** for DOI resolution
- **OpenAI** for LLM + embedding APIs
- **Streamlit** for rapid UI development
- **FAISS** for efficient vector search
- **spaCy** for industrial-strength NLP

---

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [your-email@domain.com]

---

*ClaimGuard AI — Built for researchers, by researchers. Verify the claim. Trace the evidence. Challenge the conclusion.*