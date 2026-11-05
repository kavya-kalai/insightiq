# 🏛️ InsightIQ System Architecture Blueprint

InsightIQ is designed around a **strictly decoupled, modular architecture**. The business analytical logic, statistical algorithms, and AI query-planning engine are completely isolated from the presentation layer.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    User([Business User]) -->|Upload File or Ask Query| UI[Presentation Layer: Streamlit / React]
    
    subgraph PresentationLayer [Presentation Layer]
        UI -->|File Stream / Action| Dispatcher[App Orchestrator]
    end

    subgraph CoreEngine [InsightIQ Framework-Independent Core Engine]
        Dispatcher --> Ingestion[core.ingestion: Safe Ingestion & Validation]
        Ingestion --> Profiler[core.profiler: 0-100 Health Engine & IQR]
        Profiler --> Cleaner[core.cleaner: Recipes & Version Manager]
        Cleaner --> EDA[core.eda: Plotly Visuals & Auto-Chart Selection]
        
        subgraph EvidenceFirstAI [Evidence-First AI Analyst Engine]
            Dispatcher -->|Natural Language Query| Planner[core.analyst.planner: Gemini / Fallback]
            Planner -->|JSON Execution Plan| Validator[core.analyst.validator: Safety Whitelist]
            Validator -->|Approved Plan| Executor[core.analyst.executor: Deterministic Pandas]
            Executor -->|Verified Numbers| Explainer[core.analyst.explainer: Evidence-Backed Narrative]
        end
    end

    Explainer -->|Verified Conclusion + Calculations + Plotly Figure| UI
    EDA -->|Interactive Plotly Specs| UI
    Cleaner -->|Version Diffs & Lineage| UI
    Profiler -->|Data Health Score 0-100| UI
```

---

## 2. Evidence-First AI Analyst Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Business Stakeholder
    participant UI as Presentation (Streamlit / React)
    participant Planner as 1. Semantic Planner (Gemini LLM)
    participant Validator as 2. Operation Whitelist Validator
    participant Executor as 3. Deterministic Pandas Engine
    participant Explainer as 4. Verified Narrative Explainer
    
    User->>UI: Types query: "Which products generate highest revenue?"
    UI->>Planner: plan_query(question, df)
    Note over Planner: Converts NL into strict JSON Plan:<br/>{op: 'ranking', group: 'product', target: 'revenue', top_n: 5}
    Planner->>Validator: validate_plan(plan, df)
    
    alt Invalid Column or Operation Not in Whitelist
        Validator-->>UI: Reject query honestly with reason
    else Validated & Approved
        Validator->>Executor: execute_plan(df, plan)
        Note over Executor: Executes deterministic Pandas math:<br/>df.groupby('product')['revenue'].sum()<br/>Generates Plotly bar chart
        Executor->>Explainer: explain_results(verified_numbers, calc_str)
        Note over Explainer: Explains numbers as immutable facts.<br/>NEVER recalculates or hallucinates.
        Explainer-->>UI: Conclusion + Calculation + Plotly Chart + Limitations + Next Q
        UI-->>User: Visual evidence rendered on screen
    end
```

---

## 3. Data Health Scoring Mathematical Model

The Data Health Engine calculates an interpretable score $S \in [0, 100]$ across 4 quality dimensions:

$$S = 0.30 \cdot S_{\text{completeness}} + 0.25 \cdot S_{\text{uniqueness}} + 0.25 \cdot S_{\text{validity}} + 0.20 \cdot S_{\text{uniformity}}$$

### Dimension Formulations:
1. **Completeness ($30\%$ weight):**
   $$S_{\text{completeness}} = \max\left(0, 100 \times \left(1 - \frac{\text{Missing Cells}}{\text{Total Cells}}\right)\right)$$
2. **Uniqueness ($25\%$ weight):**
   $$S_{\text{uniqueness}} = \max\left(0, 100 \times \left(1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}}\right)\right)$$
3. **Validity ($25\%$ weight):**
   $$S_{\text{validity}} = \max\left(0, 100 \times \left(1 - \frac{\text{Unparseable Dates + Negative Prices + Bad Tokens}}{\text{Total Rows}}\right)\right)$$
4. **Statistical Uniformity ($20\%$ weight):**
   Evaluated using the non-parametric Interquartile Range ($IQR = Q_3 - Q_1$). Outlier bounds are defined as:
   $$\text{Lower Bound} = Q_1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q_3 + 1.5 \times IQR$$
   Penalizes excess outliers ($> 2\%$) that distort business metrics.

---

## 4. Architectural Decoupling: Round 1 vs. Final Round

```mermaid
graph LR
    subgraph Round1 [Round 1: Rapid Interactive Slice]
        StreamlitApp[Streamlit Web App: app/main.py]
    end

    subgraph Round2 [Round 2: Full Enterprise Stack]
        FastAPI[FastAPI REST API Layer]
        ReactUI[React 18 + TypeScript + Tailwind CSS]
        Postgres[(PostgreSQL DB Metadata & Memory)]
        ReactUI <--> FastAPI
        FastAPI <--> Postgres
    end

    subgraph CoreEngineModule [Shared Python Core Engine: core/]
        CoreLogic[Pure Python Analytical Engine<br/>• Ingestion • Profiler • Cleaner • EDA • AI Analyst]
    end

    StreamlitApp --> CoreLogic
    FastAPI --> CoreLogic
```

### Architectural Guarantee for Evaluators:
- All data logic in `core/` has **zero dependencies** on Streamlit.
- In Round 2, adding the FastAPI layer and React frontend requires **zero changes** to any profiling, cleaning, EDA, or AI Analyst algorithms in `core/`.
