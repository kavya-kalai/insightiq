# InsightIQ

> **Tagline:** From Raw Data to Intelligent Decisions

InsightIQ is an AI-powered data intelligence and decision support platform that unifies automated data engineering, statistical profiling, exploratory data analysis (EDA), and evidence-first conversational analytics into one seamless workspace.

---

## The Problem We Are Solving

1. **Disconnected Tools:** Modern data workflows force non-technical teams to toggle between spreadsheets, cleaning scripts, BI dashboards, and separate AI chat tools.
2. **Manual & Fragile Data Prep:** Detecting missing data, duplicate records, mixed date formats, and statistical outliers requires tedious manual coding.
3. **AI Hallucinations in Analytics:** General LLMs frequently invent fake metrics, fabricate statistics, or write unsafe arbitrary Python code when answering business queries.
4. **Lack of Decision-Linkage:** Existing BI tools display static retrospective dashboards without actionable explanations or data-driven scenario simulations.

---

## Four Key Differentiators

| # | Differentiator | Description |
|---|---|---|
| **1** | **Evidence-First Intelligence** | The LLM never invents numbers and never runs arbitrary generated code. Queries are compiled into a safe JSON plan, validated against a strict operation whitelist, executed deterministically via Pandas, and only then explained by the LLM with full calculation evidence. |
| **2** | **Automated Data Health** | Continuously evaluates dataset quality, generating an interpretable **0–100 Data Health Score** derived from completeness, uniqueness, validity, and consistency dimensions. |
| **3** | **Decision Intelligence** | *(Planned for Final Round)* What-If Simulation Studio testing operational scenarios (e.g., $+15\%$ marketing budget) with documented statistical models rather than opaque guesses. |
| **4** | **Unified Analytical Workflow** | End-to-end continuum: `Upload → Profile → Clean → Analyze → Conversational Query → Decide`. |

---

## Architectural Design

InsightIQ follows a **strictly decoupled modular architecture**:

```
InsightIQ/
├── core/                  # Pure, framework-independent analytical engine
│   ├── ingestion.py       # Safe file upload, parsing, and type inference
│   ├── profiler.py        # Schema detection, anomaly detection, Health Score (0-100)
│   ├── cleaner.py         # Non-destructive cleaning recipes & dataset versioning
│   ├── eda.py             # Statistical summaries & automated Plotly visualization logic
│   └── analyst/           # Evidence-First AI Analyst (Planner, Validator, Executor)
├── app/                   # Presentation layer (Streamlit for Round 1 demo)
│   └── main.py            # Interactive web application & dashboard
├── data/
│   ├── sample/            # Built-in realistic datasets with planted anomalies
│   └── uploads/           # User uploaded datasets (git-ignored)
├── tests/                 # Unit tests (pytest)
├── docs/                  # Blueprint, architecture diagrams, and demo materials
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md
```

> **Why this matters for the viva:** All data algorithms reside in `core/` with zero dependency on the UI framework. This guarantees that transitioning from Streamlit to a FastAPI backend + React/TypeScript frontend for the final evaluation requires zero changes to core analytical algorithms.

---

## Scope Matrix: Round 1 Prototype vs. Final Scope

| Capability | Round 1 Prototype (Active Scope) | Planned Final Scope (16-Week Roadmap) |
|---|---|---|
| **Data Ingestion** | CSV & Excel upload, file size limits ($\le 25\text{ MB}$), type verification | Database connectors (PostgreSQL, Snowflake, BigQuery) |
| **Data Quality** | 0–100 Health Score, missing values, duplicates, outliers (IQR), format errors | Automated continuous anomaly monitoring & Drift alerts |
| **Data Cleaning** | Interactive fix recommendations, before/after diffs, dataset versioning | Multi-table joins, regex rule engines, automated pipelines |
| **Exploratory Analysis** | Automated distributions, correlation heatmaps, trend lines, auto-chart selection | Interactive pivot tables, geospatial heatmaps |
| **AI Analyst** | Evidence-first query planner (5 core operations: aggregate, rank, trend, correlation, group comparison) with strict JSON validation & graceful fallbacks | Multi-turn memory, cross-dataset querying, automated report authoring |
| **ML & Simulation** | *Out of Scope (demonstrated in architecture slides)* | Time-series forecasting (Prophet/ARIMA), K-Means clustering, Churn classifier, What-If Studio |
| **Platform / Architecture**| Streamlit + Python 3.10+ | FastAPI + React (TypeScript) + Tailwind CSS + PostgreSQL + DPDP Compliance |

---

## Quickstart & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd insightiq

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=AIzaSy...your_actual_key_here
```

### 5. Run the Application
```bash
streamlit run app/main.py
```

### 6. Run Test Suite
```bash
pytest tests/ -v
```

