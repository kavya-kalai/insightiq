# ⏱️ InsightIQ 5-Minute Live Demo Script (Round 1 Evaluation)

> **Evaluator Context:** B.Tech CSE (Data Science) Final Year Project — Prototype Demo (50 Marks).  
> **Speaker:** Kavya (Technical Lead & Presenter)  
> **Setup:** Have `streamlit run app/main.py` open at `http://localhost:8501`. Have the browser in full screen.

---

## ⏱️ Minute-by-Minute Demo Flow

### [00:00 – 01:00] Problem Statement & Raw Ingestion
* **Action:** Start on the sidebar. Keep the default **"Sample Dirty Sales (Built-in)"** loaded.
* **What to Say:**
  > *"Good morning respected evaluators. Today, business teams generate massive volumes of data, but turning that data into trusted decisions requires switching between 4 to 5 disconnected tools — spreadsheet cleaning, Python scripts, BI dashboards, and separate AI chat interfaces.*
  > 
  > *Existing conversational AI tools often hallucinate numbers or execute unsafe arbitrary Python scripts. We built **InsightIQ: From Raw Data to Intelligent Decisions**.*
  > 
  > *To test the real-world resilience of our platform, I am loading a realistic 2,000-row enterprise sales dataset containing deliberately planted anomalies: mixed date formats, missing customer records, negative unit prices, duplicate rows, and statistical outliers."*

---

### [01:00 – 02:00] Automated Data Health Audit (Tab 1)
* **Action:** Click on **Tab 1: 📋 1. Data Health Report**. Point to the 5 metric cards and the warning banner.
* **What to Say:**
  > *"The first differentiator of InsightIQ is our **Automated Data Health Engine**. Before performing any analysis, the system audits the raw dataset across 4 mathematical dimensions and calculates a transparent **0 to 100 Health Score**.*
  > 
  > *Our current raw dataset scores **91.2 / 100 (Grade A)**. Notice the transparent breakdown: Completeness is 98.4%, Uniqueness is 98.5%, and Validity is 97.9%.*
  > 
  > *Scrolling down to the Identified Health Anomalies, InsightIQ automatically flagged: 30 duplicate records, 23 unparseable date strings like '2024-02-31', 15 negative prices violating domain rules, and extreme revenue outliers detected via the non-parametric Interquartile Range ($IQR$) method."*

---

### [02:00 – 03:00] Interactive Cleaning Studio & Versioning (Tab 2)
* **Action:** Click on **Tab 2: 🧹 2. Cleaning Studio**. Point to the proposed recipes, keep them checked, and click **🚀 Execute Approved Recipes & Create New Version**.
* **What to Say:**
  > *"Unlike naive automated tools that delete rows or overwrite files in place, InsightIQ uses **Non-Destructive Dataset Versioning**.*
  > 
  > *The engine formulated 4 targeted cleaning recipes: deduplicating exact rows, standardizing mixed dates into ISO 8601 `YYYY-MM-DD`, rectifying negative prices with `abs()`, and imputing missing numbers using the **median** to prevent outlier distortion.*
  > 
  > *I will approve these recipes and click Execute...*
  > 
  > *Notice that InsightIQ created a brand new immutable version: `v2_cleaned`. The original raw dataset is completely untouched. Our Data Health Score immediately jumped to **92.6 / 100**, achieving **100% Completeness, 100% Uniqueness, and 100% Validity** with 0 missing cells remaining."*

---

### [03:00 – 04:00] Automated EDA & Auto-Chart Selection (Tab 3)
* **Action:** Click on **Tab 3: 📊 3. Automated EDA**. Scroll past the correlation heatmap and trend chart down to the **Auto-Chart Recommendation Engine**. Pick `price` for Variable X, then `marketing_spend` and `revenue`.
* **What to Say:**
  > *"Now that our data is clean, Tab 3 presents our **Automated Analytics Engine**.*
  > 
  > *Here we have the Pearson Correlation Heatmap, an interactive Time-Series Trend showing daily sales overlaid with a **7-day rolling moving average** to filter out weekend seasonality, and categorical revenue rankings.*
  > 
  > *A key feature here is our **Auto-Chart Recommendation Engine**: rather than forcing non-technical business users to choose chart types, InsightIQ inspects the semantic data types. If I select `price` alone, it generates a distribution histogram and marginal box plot. If I select `marketing_spend` and `revenue`, it automatically identifies a bivariate numeric relationship and generates a scatter plot with a linear regression trendline."*

---

### [04:00 – 05:00] The Evidence-First AI Analyst (Tab 4 & Q&A Close)
* **Action:** Click on **Tab 4: 🤖 4. Evidence-First AI Analyst**. Click the first quick button: *"Which products generate the highest revenue?"*. Then click *"Is marketing spend correlated with revenue?"*.
* **What to Say:**
  > *"Finally, we present our core differentiator: the **Evidence-First AI Analyst**.*
  > 
  > *In standard LLM tools, asking numerical questions leads to fabricated statistics or risky arbitrary code generation. In InsightIQ, the LLM NEVER computes numbers.*
  > 
  > *Let's ask: 'Which products generate the highest revenue?'*
  > 
  > *Watch the execution flow: 
  > 1. Gemini translates the question into a structured JSON execution plan.
  > 2. Our backend validates the plan against a strict operation whitelist.
  > 3. Deterministic Pandas code executes the calculation.
  > 4. The LLM only explains the verified facts.*
  > 
  > *Look at the screen: every single response provides the executive conclusion, the **exact calculation performed** (`df.groupby('product')['revenue'].sum()`), an interactive chart, the columns used, explicit data limitations, and a suggested follow-up question.*
  > 
  > *If an evaluator asks a question outside our supported operations, the system honestly declines rather than guessing.*
  > 
  > *In conclusion, we have built a working, fully tested vertical slice (31 passing unit tests) with a completely decoupled architecture, ready for our planned FastAPI + React full production platform in Round 2. Thank you, and I am ready for your questions!"*

---

## 💡 Quick Tips for the Live Demo
1. **If Wi-Fi or API slows down:** Do not panic! InsightIQ has a built-in deterministic fallback engine that runs locally with 100% precision and zero latency.
2. **If an evaluator asks you to upload a file:** Switch the sidebar radio button to "Upload CSV / Excel" and upload any test CSV; the profiler and analyst will immediately adapt to its schema.
3. **If asked to show the code:** Open [`core/analyst/validator.py`](file:///c:/Users/Kavya%20K/OneDrive/Documents/GitHub/InSightIQ/core/analyst/validator.py) and [`core/analyst/executor.py`](file:///c:/Users/Kavya%20K/OneDrive/Documents/GitHub/InSightIQ/core/analyst/executor.py) to show the strict operation whitelist and pure Pandas execution.

