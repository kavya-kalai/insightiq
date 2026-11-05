# 💡 5 Verified Demo Questions for InsightIQ AI Analyst

These 5 questions are pre-tested, verified, and mapped to the 5 supported operations. Each button in the UI corresponds to one of these queries.

---

### Question 1: Ranking (Top N)
* **Prompt:** `"Which products generate the highest revenue?"`
* **Operation:** `ranking`
* **Underlying Calculation:** 
  ```python
  df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)
  ```
* **Expected Output:**
  - **Leader:** `Laptop Pro` (leading revenue by a wide margin) followed by `4K Monitor` and `Ergonomic Chair`.
  - **Visualization:** Horizontal Bar Chart with Blues color scale.
  - **Columns Used:** `product`, `revenue`.
  - **Viva Highlight:** Emphasize that the ranking is calculated directly on cleaned transaction records using deterministic groupby aggregations.

---

### Question 2: Simple Aggregation
* **Prompt:** `"What is the total revenue?"`
* **Operation:** `aggregate`
* **Underlying Calculation:** 
  ```python
  df['revenue'].dropna().sum()
  ```
* **Expected Output:**
  - **Result:** Cumulative total across all 1,956+ cleaned transactions ($\sim \$5.7\text{M} - \$6.2\text{M}$).
  - **Visualization:** Plotly KPI Indicator Gauge.
  - **Columns Used:** `revenue`.
  - **Viva Highlight:** Highlight that the LLM did not do arithmetic; Pandas computed the exact sum, and the LLM merely formatted it into an executive summary.

---

### Question 3: Time-Series Trend Over Time
* **Prompt:** `"What is the revenue trend over time?"`
* **Operation:** `trend`
* **Underlying Calculation:** 
  ```python
  df.groupby(pd.to_datetime('date').dt.date)['revenue'].sum()
  ```
* **Expected Output:**
  - **Result:** Daily totals across the 12-month calendar window with a 7-day rolling moving average.
  - **Visualization:** Dual-trace interactive line chart (Daily raw sales + 7-Day Moving Trend).
  - **Columns Used:** `date`, `revenue`.
  - **Viva Highlight:** Point out that raw dates had mixed formats (`DD/MM/YYYY`, `MM-DD-YYYY`, textual dates) which were standardized into ISO 8601 during Milestone 4 before time-series aggregation could succeed.

---

### Question 4: Pearson Correlation Analysis
* **Prompt:** `"Is marketing spend correlated with revenue?"`
* **Operation:** `correlation`
* **Underlying Calculation:** 
  ```python
  df['marketing_spend'].corr(df['revenue'])
  ```
* **Expected Output:**
  - **Result:** Strong positive linear correlation ($r \approx +0.94$ to $+0.98$).
  - **Visualization:** Bivariate Scatter Plot with dashed linear regression fit line ($y = mx + b$).
  - **Columns Used:** `marketing_spend`, `revenue`.
  - **Viva Highlight:** Point out the **Limitations** section where the system explicitly notes: *"Correlation indicates linear statistical association, not direct causality. Latent confounding variables may influence both metrics."* Evaluators love seeing statistical rigor!

---

### Question 5: Bivariate Group Comparison
* **Prompt:** `"Compare revenue between North and South regions"`
* **Operation:** `comparison`
* **Underlying Calculation:** 
  ```python
  val_north = df[df['region'] == 'North']['revenue'].sum()
  val_south = df[df['region'] == 'South']['revenue'].sum()
  pct_diff = ((val_north - val_south) / val_south) * 100
  ```
* **Expected Output:**
  - **Result:** Side-by-side total revenue for North vs. South with absolute difference and percentage delta.
  - **Visualization:** Comparative two-bar chart.
  - **Columns Used:** `region`, `revenue`.
  - **Viva Highlight:** Show how the system handles comparative queries without writing custom SQL, validating that both regions exist in the categorical space before computing the delta.

