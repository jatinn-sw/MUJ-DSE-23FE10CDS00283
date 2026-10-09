# ClaimGuard AI

**Verify the Claim. Trace the Evidence. Challenge the Conclusion.**

## 👤 Student & Project Details

| **Detail**              | **Information**                                                             |
| ----------------------- | --------------------------------------------------------------------------- |
| **Name**                | Jatin Kumar Sangewar                                                              |
| **Registration Number** | 23FE10CDS00283                                                |
| **Branch**              | B.Tech Computer Science & Engineering – Data Science                        |
| **Batch**               | 2023-27                                                              |
| **Project Title**       | ClaimGuard AI – LLM-Based Research Claim Verification and Evidence Auditing |
| **GitHub Username**     | jatinn-sw                                                                                                       |

---

## 📌 Project Overview

**ClaimGuard AI** is an AI-powered evidence-auditing platform designed to evaluate factual and research claims present in academic documents.

The system automatically extracts claims from uploaded research documents, retrieves relevant supporting or contradicting evidence from the document itself and external academic sources, and uses Large Language Models (LLMs) to evaluate the strength of the available evidence.

The platform generates a transparent audit report in which every verdict can be traced back to the evidence used during analysis.

### Core Capabilities

* 📄 **Document Processing** — Extracts and processes text from PDF, DOCX, and TXT files.
* 🔍 **Claim Extraction** — Identifies potentially verifiable research claims using NLP and LLM-based extraction.
* 🧠 **Semantic Retrieval** — Uses embeddings and FAISS to retrieve relevant evidence from the uploaded document.
* 📚 **Academic Search** — Retrieves external research evidence through Semantic Scholar and Crossref.
* 🌐 **Web Evidence** — Supports web-based evidence retrieval through SerpAPI or Google Custom Search.
* 🤖 **LLM Analysis** — Evaluates claims against retrieved evidence using structured LLM reasoning.
* ⚖️ **Claim Verdicts** — Classifies claims as supported, contradicted, insufficiently supported, or overstated.
* 🔗 **Evidence Traceability** — Links every verdict to its underlying evidence.
* 📊 **Audit Reports** — Generates structured Markdown and JSON reports.

---

## 🎯 Problem Statement

Research papers frequently contain claims that may be:

* Unsupported by sufficient evidence
* Overstated relative to the available evidence
* Contradicted by other findings
* Difficult to verify manually
* Supported by weak or irrelevant sources

Traditional verification and peer-review processes require significant human effort and can be time-consuming.

**ClaimGuard AI addresses this problem by providing an automated evidence-auditing pipeline that combines NLP, semantic retrieval, academic search, and LLM-based reasoning.**

---

## 💡 Proposed Solution

ClaimGuard AI follows a multi-stage evidence verification pipeline:

```text
Academic Document
       │
       ▼
Document Processing
       │
       ▼
Claim Extraction
       │
       ▼
Claim Classification
       │
       ▼
┌─────────────────────────────┐
│      Evidence Retrieval     │
├─────────────────────────────┤
│ Internal Evidence → FAISS   │
│ Academic Evidence → APIs    │
│ Web Evidence → Search APIs  │
└──────────────┬──────────────┘
               │
               ▼
        LLM Evidence Analysis
               │
               ▼
       Claim Verdict + Confidence
               │
               ▼
       Conflict & Quality Analysis
               │
               ▼
          Final Audit Report
```

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                       CLAIMGUARD AI                           │
└──────────────────────────────────────────────────────────────┘

  PDF / DOCX / TXT
          │
          ▼
┌──────────────────────┐
│  Document Processing │
│       PyMuPDF        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Claim Extraction   │
│   NLP + LLM Hybrid   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Claim Classification │
│ & Importance Scoring │
└──────────┬───────────┘
           │
           ▼
 ┌────────────────────────────────────────┐
 │           Evidence Retrieval           │
 │                                        │
 │  ┌────────────┐  ┌──────────────────┐ │
 │  │ Internal   │  │ External         │ │
 │  │ Evidence   │  │ Evidence         │ │
 │  │            │  │                  │ │
 │  │ Embeddings │  │ Semantic Scholar │ │
 │  │ FAISS      │  │ Crossref         │ │
 │  │ Semantic   │  │ SerpAPI / CSE    │ │
 │  │ Search     │  │                  │ │
 │  └────────────┘  └──────────────────┘ │
 └───────────────────┬────────────────────┘
                     │
                     ▼
            ┌──────────────────┐
            │   LLM Analysis   │
            │   RAG + Structured│
            │      Output      │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Claim Verdict &  │
            │   Confidence     │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Conflict & Source│
            │ Quality Analysis │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Final Audit     │
            │  Report          │
            │ Markdown + JSON   │
            └──────────────────┘
```

---

# 🔄 Complete Pipeline

## 1. Document Processing

The system accepts:

* PDF
* DOCX
* TXT

PyMuPDF is used for PDF processing while `python-docx` handles DOCX files.

The document is converted into structured sentences containing information such as:

```text
sentence_id
page
section
text
```

Additional processing includes:

* Sentence segmentation
* Section detection
* Reference detection
* Boilerplate removal
* Figure-caption filtering
* Table-header filtering

---

## 2. Claim Extraction

Candidate claims are first identified using local NLP techniques.

The system looks for indicators such as:

* Numerical statements
* Percentages
* Comparative statements
* Causal relationships
* Performance claims
* Research findings

The candidate sentences are then passed through an LLM for structured claim extraction.

Each extracted claim contains information such as:

```text
Claim ID
Claim Text
Category
Importance Score
Source Sentence
```

A heuristic fallback is available if the LLM extraction stage fails.

---

## 3. Embedding Generation

The extracted document sentences are converted into vector representations using an embedding model.

These vectors are stored in a **FAISS** similarity-search index.

```text
Document Sentences
       │
       ▼
Embedding Model
       │
       ▼
Vector Representations
       │
       ▼
FAISS Index
```

This allows the system to efficiently retrieve semantically relevant passages for each claim.

---

## 4. Internal Evidence Retrieval

For every extracted claim:

```text
Claim
  │
  ▼
Claim Embedding
  │
  ▼
FAISS Similarity Search
  │
  ▼
Top-K Relevant Sentences
```

The retrieved evidence includes:

* Relevance score
* Page number
* Section
* Original sentence

This provides an evidence trail directly inside the uploaded document.

---

## 5. External Evidence Retrieval

For important claims, the system generates optimized academic search queries.

External sources include:

### Semantic Scholar

Used for:

* Research paper discovery
* Titles
* Abstracts
* Authors
* Publication year
* Venue
* Citation information
* DOI

### Crossref

Used for:

* DOI metadata
* Publication information
* Authors
* Journal information
* Publication year

### Web Search

The system can additionally use:

* SerpAPI
* Google Custom Search

External results are deduplicated and ranked according to source quality and relevance.

---

# 🤖 LLM-Based Evidence Analysis

The retrieved evidence is provided to the LLM using a structured prompt.

The model evaluates:

* Whether evidence supports the claim
* Whether evidence contradicts the claim
* Whether available evidence is insufficient
* Whether the claim is stronger than the evidence supports
* Confidence in the resulting assessment
* Limitations of the evidence
* Suggested claim revision

The output follows a structured format rather than relying on free-form responses.

---

# ⚖️ Claim Verdicts

ClaimGuard AI distinguishes between several evidence states.

### ✅ SUPPORTED

The available evidence provides reasonable support for the claim.

### ❌ CONTRADICTED

Retrieved evidence conflicts with the claim.

### ⚠️ INSUFFICIENT EVIDENCE

The available evidence is not sufficient to determine whether the claim is adequately supported.

### 🟠 OVERSTATED

Evidence exists, but the wording or strength of the claim goes beyond what the evidence justifies.

> **Important:** SUPPORTED does not mean that the claim has been proven true. It means that the claim is supported by the evidence available to the system.

Similarly, **INSUFFICIENT EVIDENCE does not mean FALSE**.

---

# 🛡️ Hallucination Protection

ClaimGuard AI is designed to prevent unsupported LLM conclusions.

The analysis prompt instructs the model to:

* Never invent sources
* Never fabricate citations
* Never treat internal model knowledge as evidence
* Only reason over retrieved evidence
* Explicitly report insufficient evidence
* Provide traceable evidence for every verdict

The intended reasoning chain is:

```text
Claim
  ↓
Retrieved Evidence
  ↓
Evidence Evaluation
  ↓
LLM Reasoning
  ↓
Verdict
```

---

# 📊 Evidence Traceability

One of the central design principles of ClaimGuard AI is traceability.

Every verdict follows the structure:

```text
Claim
   ↓
Internal Evidence
   ↓
External Evidence
   ↓
Evidence Quality
   ↓
LLM Reasoning
   ↓
Verdict
   ↓
Confidence
```

This makes the system's output easier to inspect and audit compared with a conventional black-box fact-checking system.

---

# 🛠️ Technology Stack

| Category                 | Technologies                          |
| ------------------------ | ------------------------------------- |
| **Frontend**             | Streamlit, Custom CSS                 |
| **Programming Language** | Python                                |
| **NLP**                  | spaCy, NLTK, HuggingFace Transformers |
| **LLM**                  | OpenAI / Google Gemini / Anthropic    |
| **Embeddings**           | OpenAI / Google Gemini                |
| **Vector Search**        | FAISS                                 |
| **Document Processing**  | PyMuPDF, python-docx                  |
| **Academic Search**      | Semantic Scholar, Crossref            |
| **Web Search**           | SerpAPI, Google CSE                   |
| **ML**                   | PyTorch, scikit-learn                 |
| **Data Processing**      | Pandas, NumPy                         |
| **Visualization**        | Plotly, PyVis, NetworkX               |
| **Reporting**            | Markdown, ReportLab, WeasyPrint       |
| **Testing**              | pytest, pytest-asyncio                |
| **Configuration**        | Pydantic, YAML                        |
| **Caching**              | Local Pickle / Redis                  |
| **API Retry**            | Tenacity                              |

---

# 📁 Project Structure

```text
ClaimGuard/
│
├── app/
│   ├── config.py
│   ├── pdf_parser.py
│   ├── claim_extractor.py
│   ├── embeddings.py
│   ├── evidence_retriever.py
│   ├── query_generator.py
│   ├── academic_search.py
│   ├── web_search.py
│   ├── llm_analyzer.py
│   ├── report_generator.py
│   ├── pipeline.py
│   ├── utils.py
│   │
│   └── frontend/
│       ├── main.py
│       ├── styles.py
│       └── components.py
│
├── prompts/
│   ├── claim_extraction.txt
│   ├── claim_classification.txt
│   ├── query_generation.txt
│   ├── evidence_analysis.txt
│   └── final_report.txt
│
├── config/
│   └── config.yaml
│
├── evaluation/
│   ├── annotations.json
│   ├── evaluate.py
│   └── metrics.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_claims.py
│   ├── test_retrieval.py
│   └── test_llm.py
│
├── sample_data/
├── cache/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ClaimGuard
```

### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure API Keys

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key

SEMANTIC_SCHOLAR_API_KEY=your_key
CROSSREF_API_KEY=your_key

SERPAPI_API_KEY=your_key
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_id
```

Only the APIs required by your selected configuration need to be enabled.

---

# ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app/frontend/main.py
```

Then open:

```text
http://localhost:8501
```

### Typical Workflow

```text
Upload Research Paper
        ↓
Document Processing
        ↓
Claim Extraction
        ↓
Evidence Retrieval
        ↓
LLM Analysis
        ↓
Dashboard
        ↓
Claim Explorer
        ↓
Final Audit Report
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=app --cov-report=html
```

Testing covers:

* PDF extraction
* Sentence segmentation
* Claim extraction
* Claim classification
* FAISS retrieval
* Embedding cache
* External-search mocking
* LLM response parsing
* Verdict validation

---

# ⚠️ API Rate Limits & Graceful Degradation

External academic and web-search APIs may impose rate limits.

ClaimGuard AI is designed so that temporary external API failures do not necessarily stop the complete pipeline.

```text
External API
     │
     ▼
Request
     │
     ├── Success ──────────► Evidence
     │
     └── Rate Limit/Error
               │
               ▼
           Retry Logic
               │
               ▼
       Graceful Fallback
               │
               ▼
       Continue Pipeline
```

The system can continue using available internal evidence when an external source is temporarily unavailable.

This allows development and testing of the core pipeline without requiring every external search service to be available simultaneously.

---

# 🔬 Evaluation

ClaimGuard AI supports evaluation against human-labelled claims.

The evaluation framework measures:

| Metric                 | Purpose                              |
| ---------------------- | ------------------------------------ |
| **Accuracy**           | Overall correctness                  |
| **Precision**          | Correct positive predictions         |
| **Recall**             | Coverage of relevant predictions     |
| **F1 Score**           | Balance between precision and recall |
| **Confusion Matrix**   | Verdict-level error analysis         |
| **Processing Time**    | Pipeline efficiency                  |
| **API Calls**          | Resource usage                       |
| **Retrieval Latency**  | Search performance                   |
| **JSON Parse Success** | Structured LLM output reliability    |

Run evaluation with:

```bash
python -m evaluation.evaluate
```

---

# 🎨 Frontend

The Streamlit interface provides an interactive workflow for document auditing.

### Main Sections

* **Landing Page** — Project introduction and features
* **Upload** — Upload and analyze research documents
* **Processing** — Pipeline progress and logs
* **Dashboard** — Overall audit statistics
* **Claims Explorer** — Search and filter extracted claims
* **Claim Detail** — Evidence, reasoning, verdict, and confidence
* **Paper Viewer** — Document-level claim inspection
* **Final Report** — Complete audit report with export options

---

# 🔮 Future Scope

Planned improvements include:

* Multi-document comparison
* Biomedical evidence retrieval through PubMed
* Local LLM support using Ollama/vLLM
* Redis-backed distributed caching
* User authentication and project persistence
* REST API
* Advanced evidence graphs
* Citation network analysis
* Duplicate and similar-claim clustering
* Active learning using human feedback

---

# 📌 Current Limitations

The current implementation has several limitations:

1. Single-document analysis
2. English-language focus
3. No real-time collaboration
4. Limited graph visualization
5. No authentication or multi-tenancy
6. Dependence on external APIs for expanded evidence retrieval

---

# 📜 License

This project is released under the **MIT License**.

See `LICENSE` for details.

---

# 🙏 Acknowledgements

* **Semantic Scholar** — Academic paper metadata and search
* **Crossref** — DOI and publication metadata
* **OpenAI** — LLM and embedding APIs
* **Google Gemini** — LLM and embedding capabilities
* **Streamlit** — Interactive application framework
* **FAISS** — Efficient vector similarity search
* **spaCy** — NLP processing

---

## 👩‍💻 Author

**Jatin Kumar Sangewar**

B.Tech Computer Science & Engineering – Data Science

GitHub: [jatinn-sw](https://github.com/jatinn-sw)

---

> **ClaimGuard AI — Verify the Claim. Trace the Evidence. Challenge the Conclusion.**
