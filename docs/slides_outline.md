# 📊 InsightIQ — Round 1 Evaluation Slide Deck Outline

> **Presentation Duration:** 5–7 Minutes  
> **Target Audience:** Engineering Faculty & Project Evaluation Committee (50 Marks)  
> **Key Message:** InsightIQ delivers a complete, verified vertical slice of an autonomous data intelligence platform with an evidence-first architecture that mathematically eliminates LLM hallucinations.

---

### Slide 1: Title & Overview
* **Headline:** InsightIQ — Autonomous Data Intelligence & Decision Support Platform
* **Tagline:** *From Raw Data to Intelligent Decisions*
* **Project Details:** Final Year Project | B.Tech CSE (Data Science)
* **Team:** Kavya (Data Science & Backend / Full Architecture)
* **Key Visual:** Platform logo, architecture icon, and 5-stage lifecycle:
  `Upload → Profile → Clean → Analyze → Conversational Query`

---

### Slide 2: The Problem We Are Solving
* **Core Pain Points:**
  1. **Fragmented Workflows:** Business analysts toggle between spreadsheets, cleaning scripts, BI dashboards, and disconnected AI chatbots.
  2. **Manual & Fragile Data Cleaning:** Detecting corrupted dates, duplicate rows, and domain rule violations consumes 80% of analyst time.
  3. **AI Hallucinations in Analytics:** General LLMs fabricate numerical metrics or run unsafe, unpredictable arbitrary Python scripts (`eval()`).
  4. **Lack of Explainability:** Traditional dashboards show static numbers without contextual reasoning or reproducible audit trails.

---

### Slide 3: Competitive Landscape
* **Comparison Table:**

| Feature | Julius AI | Microsoft Power BI | ThoughtSpot | InsightIQ (Our Platform) |
|---|:---:|:---:|:---:|:---:|
| **Conversational Interface** | Generates raw Python | NLQ to DAX | Search keywords | **Evidence-First JSON Plan** |
| **Hallucination Prevention** | Low (code execution risks) | High (structured) | High (relational) | **100% Guaranteed (No LLM math)** |
| **Automated Data Health** | Basic summary | None (external ETL) | None (pre-modeled) | **Transparent 0–100 Health Score** |
| **Dataset Versioning** | Ephemeral | Database-dependent | Workspace views | **Non-Destructive Git-like Lineage** |
| **Auto-Chart Selection** | AI chosen | Manual configuration | Pre-built templates | **Rule-Based Type Inference Engine** |

---

### Slide 4: Four Proposed Differentiators
1. **Evidence-First Intelligence:** Every numerical answer is backed by verified Pandas computations and auditable formulas. Zero LLM hallucinations.
2. **Automated Data Health Audit:** Multi-dimensional quality evaluation (Completeness, Uniqueness, Validity, Uniformity) producing a single, defendable 0–100 score.
3. **Decision Intelligence:** *(Roadmap)* What-If simulation studio testing operational scenarios before executing business decisions.
4. **Unified Analytical Workflow:** Data preparation, profiling, visualization, and conversational reasoning unified into one cohesive continuum.

---

### Slide 5: System Architecture & Decoupled Design
* **Key Architectural Principle:** Complete decoupling between Core Analytics (`core/`) and UI Presentation (`app/`).
* **Visual:** Mermaid architecture diagram (Decoupled Core $\leftrightarrow$ Streamlit / FastAPI+React).
* **Viva Defense Point:** 
  > *"Because our data algorithms in `core/` have zero dependencies on Streamlit, migrating to FastAPI and React in the final round requires zero modifications to our data processing or AI analyst engines."*

---

### Slide 6: The Evidence-First AI Analyst Pipeline
* **Visual Flow:**
  $$\text{User Query} \xrightarrow{1} \text{Gemini Planner} \xrightarrow{2} \text{JSON Plan} \xrightarrow{3} \text{Safety Validator} \xrightarrow{4} \text{Pandas Engine} \xrightarrow{5} \text{Verified Explainer}$$
* **Strict Security Controls:**
  - **No `eval()` or `exec()`:** The backend never executes arbitrary LLM-generated code.
  - **Operation Whitelist:** Only 5 approved operations allowed (`aggregate`, `ranking`, `trend`, `correlation`, `comparison`).
  - **Honest Out-of-Scope Declines:** Refuses to guess when a query cannot be safely mapped.
  - **Resilient Fallback:** Automatically switches to deterministic semantic parsing if external APIs fail.

---

### Slide 7: Round 1 Prototype Scope vs. Planned Roadmap

| Engine | Round 1 Scope (Completed Today) | Planned Final Scope (Final Evaluation) |
|---|---|---|
| **Data Engine (01)** | CSV/Excel upload ($\le 25\text{MB}$), 0–100 Health Score, Non-destructive cleaning recipes & versioning | Live DB connectors (PostgreSQL), automated ETL pipelines |
| **Analytics Engine (02)** | Descriptive statistics, Pearson correlation heatmap, time-series trends, auto-chart recommender | Interactive pivot grids, geospatial mapping |
| **AI Analyst (03)** | 5-operation evidence-first query planner, strict JSON validation, Plotly visual evidence, dual fallback | Multi-turn conversation context, multi-table joins |
| **Predictive ML (04)** | *Out of Scope (Architecture ready)* | Time-series forecasting (Prophet/ARIMA), K-Means clustering, Churn classifier |
| **Decision Studio (05)**| *Out of Scope (Architecture ready)* | What-If simulation studio (marketing spend vs. revenue impact) |
| **Platform Stack** | Python 3.10+ & Streamlit interactive prototype | FastAPI backend + React/TypeScript frontend + PostgreSQL |

---

### Slide 8: Technical Validation & Test Results
* **Quality & Stability Metrics:**
  - **31 Automated Unit Tests:** 100% pass rate in pytest across ingestion, profiler, cleaner, EDA, and AI analyst validator.
  - **Planted Benchmark Dataset:** Tested against 2,030-row dirty dataset with 260 nulls, 30 duplicates, negative prices, and mixed dates.
  - **Demonstrated Health Score Improvement:** Raw dataset score ($91.2$) $\rightarrow$ Cleaned dataset score ($92.6$), with 100% completeness, 100% uniqueness, and 100% validity.
* **Closing:**
  > *"InsightIQ proves that AI can make analytics accessible to everyone without sacrificing accuracy, transparency, or security."*
