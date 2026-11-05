"""InsightIQ Application - Presentation Layer (Streamlit).

Unified web dashboard for the Round 1 Prototype:
1. Ingestion: Load built-in sample or upload custom CSV/Excel.
2. Data Health Report: 0-100 score with 4-dimension audit.
3. Interactive Cleaning Studio: Recipe approval, diff preview, non-destructive versioning.
4. Automated EDA: Plotly distributions, correlations, time-series, auto-chart selection.
5. Evidence-First AI Analyst: Zero-hallucination query execution with verified proof.
"""

from __future__ import annotations
import os
from pathlib import Path
import sys
import dotenv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure core package is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

dotenv.load_dotenv(ROOT_DIR / ".env")

from core.analyst import ask_analyst
from core.cleaner import (
    DatasetVersionManager,
    apply_cleaning_pipeline,
    suggest_cleaning_recipes,
)
from core.eda import (
    compute_correlation_matrix,
    compute_summary_statistics,
    create_correlation_heatmap,
    create_distribution_chart,
    create_time_series_trend,
    create_top_categories_chart,
    recommend_chart,
)
from core.ingestion import load_dataset
from core.profiler import infer_semantic_type, profile_dataset

# Page Configuration
st.set_page_config(
    page_title="InsightIQ | Autonomous Data Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize app state variables."""
    if "version_manager" not in st.session_state:
        sample_path = ROOT_DIR / "data" / "sample" / "sales_data_dirty.csv"
        ingest_res = load_dataset(sample_path)
        if ingest_res.is_valid and ingest_res.dataframe is not None:
            st.session_state.version_manager = DatasetVersionManager(
                ingest_res.dataframe, "Sample Dirty Sales"
            )
        else:
            st.session_state.version_manager = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


init_session_state()
manager: DatasetVersionManager | None = st.session_state.version_manager


# ==========================================
# SIDEBAR: DATASET & VERSION CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## ⚡ InsightIQ")
    st.caption("From Raw Data to Intelligent Decisions")
    st.markdown("---")

    # Dataset Source Selector
    st.markdown("### 📁 Dataset Source")
    data_source = st.radio(
        "Choose Source:",
        ["Sample Dirty Sales (Built-in)", "Upload CSV / Excel"],
        label_visibility="collapsed",
    )

    if data_source == "Upload CSV / Excel":
        uploaded_file = st.file_uploader(
            "Upload File (Max 25 MB)",
            type=["csv", "xlsx", "xls"],
            help="Strict MIME validation & automatic encoding detection",
        )
        if uploaded_file is not None:
            if st.button("Load Uploaded Dataset", type="primary", use_container_width=True):
                ingest_res = load_dataset(uploaded_file, filename=uploaded_file.name)
                if ingest_res.is_valid and ingest_res.dataframe is not None:
                    st.session_state.version_manager = DatasetVersionManager(
                        ingest_res.dataframe, uploaded_file.name
                    )
                    st.success(f"Loaded {uploaded_file.name} ({ingest_res.row_count:,} rows)")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {', '.join(ingest_res.errors)}")
    else:
        if st.button("Reset to Default Sample", use_container_width=True):
            sample_path = ROOT_DIR / "data" / "sample" / "sales_data_dirty.csv"
            res = load_dataset(sample_path)
            if res.dataframe is not None:
                st.session_state.version_manager = DatasetVersionManager(
                    res.dataframe, "Sample Dirty Sales"
                )
                st.session_state.chat_history = []
                st.rerun()

    st.markdown("---")

    # Dataset Version Selector
    if manager:
        st.markdown("### 🧬 Dataset Versions")
        versions = manager.list_versions()
        ver_options = [v["version_id"] for v in versions]
        ver_labels = {
            v["version_id"]: f"{v['version_id']} ({v['name']} - Score: {v['health_score']:.1f})"
            for v in versions
        }
        selected_ver = st.selectbox(
            "Active Version:",
            ver_options,
            index=ver_options.index(manager.active_version_id),
            format_func=lambda x: ver_labels.get(x, x),
        )
        if selected_ver != manager.active_version_id:
            manager.active_version_id = selected_ver
            st.rerun()

        active_v = manager.get_active_version()
        st.caption(f"Rows: **{len(active_v.df):,}** | Cols: **{len(active_v.df.columns)}**")
        st.caption(f"Created: {active_v.created_at}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #94A3B8;'>InsightIQ v0.1 • B.Tech CSE (Data Science) Project</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN CONTENT AREA
# ==========================================
st.markdown('<div class="main-header">InsightIQ Autonomous Data Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified Data Profiling, Automated Cleaning, Interactive EDA, and Evidence-First AI Analytics</div>', unsafe_allow_html=True)

if not manager:
    st.warning("Please load or upload a dataset from the sidebar to begin.")
    st.stop()

current_version = manager.get_active_version()
df = current_version.df

# Main Navigation Tabs
tab_health, tab_cleaner, tab_eda, tab_analyst, tab_arch = st.tabs([
    "📋 1. Data Health Report",
    "🧹 2. Cleaning Studio",
    "📊 3. Automated EDA",
    "🤖 4. Evidence-First AI Analyst",
    "🛡️ 5. System Architecture & Audit",
])


# ------------------------------------------
# TAB 1: DATA HEALTH REPORT
# ------------------------------------------
with tab_health:
    st.markdown(f"### 📋 Data Health Audit: `{current_version.name}`")
    report = profile_dataset(df, current_version.name)
    sb = report.score

    # Score Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Overall Health</div>
            <div class="metric-val">{sb.overall_health_score:.1f}</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #16A34A;">{sb.grade}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Completeness (30%)</div>
            <div class="metric-val">{sb.completeness_score:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748B;">{report.total_missing_cells:,} Nulls</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Uniqueness (25%)</div>
            <div class="metric-val">{sb.uniqueness_score:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748B;">{report.total_duplicate_rows:,} Dups</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Validity (25%)</div>
            <div class="metric-val">{sb.validity_score:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748B;">Domain & Types</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Uniformity (20%)</div>
            <div class="metric-val">{sb.outlier_score:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748B;">IQR Outliers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Issues Box
    if report.top_issues:
        st.markdown("#### ⚠️ Identified Health Anomalies")
        for issue in report.top_issues:
            st.warning(issue)
    else:
        st.success("✅ No critical data health anomalies detected. Dataset is clean!")

    # Column Diagnostics Table
    st.markdown("#### 🔬 Column-by-Column Diagnostics")
    col_data = []
    for c_name, cp in report.columns.items():
        col_data.append({
            "Column": c_name,
            "Semantic Type": cp.semantic_type,
            "Physical Dtype": cp.physical_dtype,
            "Null Count": f"{cp.null_count:,} ({cp.null_percentage:.1f}%)",
            "Unique Count": f"{cp.unique_count:,}",
            "IQR Outliers": f"{cp.outlier_count:,} ({cp.outlier_percentage:.1f}%)" if cp.semantic_type == "numeric" else "—",
            "Validity Warnings": f"{cp.invalid_count:,} invalid" if cp.invalid_count > 0 else "None",
            "Sample Values": str(cp.sample_values[:3]),
        })
    st.dataframe(pd.DataFrame(col_data), use_container_width=True)

    # Raw Data Preview
    with st.expander("🔍 View Raw Records Sample"):
        st.dataframe(df.head(25), use_container_width=True)


# ------------------------------------------
# TAB 2: CLEANING STUDIO
# ------------------------------------------
with tab_cleaner:
    st.markdown("### 🧹 Interactive Cleaning Studio & Recipe Approval")
    st.write(
        "InsightIQ automatically formulates targeted cleaning recipes based on the Data Health Audit. "
        "Review and approve individual fixes below, then click **Execute Cleaning Pipeline** to create a non-destructive versioned dataset."
    )

    report_for_cleaning = profile_dataset(df, current_version.name)
    recipes = suggest_cleaning_recipes(report_for_cleaning)

    if not recipes:
        st.info("No cleaning recipes needed for this dataset version. It is already fully cleaned!")
    else:
        st.markdown("#### 📝 Proposed Cleaning Recipes")
        approved_recipes = []
        for i, rec in enumerate(recipes):
            col_a, col_b = st.columns([0.08, 0.92])
            with col_a:
                is_checked = st.checkbox(
                    "Approve",
                    value=rec.is_approved,
                    key=f"rec_check_{rec.recipe_id}_{i}",
                    label_visibility="collapsed",
                )
            with col_b:
                st.markdown(f"**{rec.description}**")
                st.caption(f"Estimated Impact: {rec.estimated_impact}")
            rec.is_approved = is_checked
            if is_checked:
                approved_recipes.append(rec)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Execute Approved Recipes & Create New Version", type="primary"):
            if not approved_recipes:
                st.warning("Please select at least one cleaning recipe to execute.")
            else:
                with st.spinner("Applying cleaning recipes non-destructively..."):
                    cleaned_df, diffs = apply_cleaning_pipeline(df, approved_recipes)
                    new_ver = manager.create_new_version(
                        new_df=cleaned_df,
                        name=f"Cleaned ({current_version.version_id})",
                        applied_recipes=approved_recipes,
                        diffs=diffs,
                    )
                st.success(
                    f"Created version `{new_ver.version_id}`! "
                    f"Health score improved from **{current_version.health_score:.1f}** to **{new_ver.health_score:.1f}**."
                )
                
                # Show Diff Breakdown
                st.markdown("#### 📊 Execution Diff Summary")
                diff_rows = [
                    {
                        "Action": d.action_type,
                        "Target Column": d.column or "All",
                        "Rows Affected": d.affected_rows_count,
                        "Nulls Before -> After": f"{d.nulls_before} -> {d.nulls_after}",
                        "Details": d.details,
                    }
                    for d in diffs
                ]
                st.table(pd.DataFrame(diff_rows))
                st.rerun()


# ------------------------------------------
# TAB 3: AUTOMATED EDA
# ------------------------------------------
with tab_eda:
    st.markdown("### 📊 Automated Exploratory Data Analysis (EDA)")

    # 1. Descriptive Summary Statistics
    with st.expander("📈 Descriptive Summary Statistics", expanded=True):
        stats = compute_summary_statistics(df)
        if stats["numeric"]:
            st.markdown("##### Numeric Variables")
            st.dataframe(pd.DataFrame(stats["numeric"]).T, use_container_width=True)
        if stats["categorical"]:
            st.markdown("##### Categorical Variables")
            st.dataframe(pd.DataFrame(stats["categorical"]).T, use_container_width=True)

    # 2. Visualizations
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        st.markdown("#### 🔥 Pearson Correlation Heatmap")
        corr_fig = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)

    with eda_col2:
        st.markdown("#### ⏳ Time-Series Trend Analysis")
        date_candidates = [c for c in df.columns if infer_semantic_type(df[c]) == "datetime"]
        num_candidates = [c for c in df.columns if infer_semantic_type(df[c]) == "numeric"]
        
        if date_candidates and num_candidates:
            selected_date = date_candidates[0]
            selected_val = st.selectbox("Metric to Trend:", num_candidates, index=0)
            trend_fig = create_time_series_trend(df, selected_date, selected_val)
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Time-series trend requires at least 1 datetime and 1 numeric column.")

    st.markdown("---")

    # 3. Categorical Rankings & Distribution Explorer
    eda_col3, eda_col4 = st.columns(2)
    with eda_col3:
        st.markdown("#### 🏆 Top Category Performance")
        cat_candidates = [c for c in df.columns if infer_semantic_type(df[c]) in {"categorical", "identifier/text"}]
        if cat_candidates:
            sel_cat = st.selectbox("Categorical Group:", cat_candidates, index=0)
            sel_num = st.selectbox("Measure by Metric:", [None] + num_candidates, index=0)
            cat_fig = create_top_categories_chart(df, sel_cat, sel_num)
            st.plotly_chart(cat_fig, use_container_width=True)

    with eda_col4:
        st.markdown("#### 📦 Distribution & Outliers (Box + Histogram)")
        if num_candidates:
            sel_dist_col = st.selectbox("Select Numeric Column:", num_candidates, index=0)
            dist_fig = create_distribution_chart(df, sel_dist_col)
            st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("---")

    # 4. Auto-Chart Recommendation Engine
    st.markdown("#### 🤖 Auto-Chart Recommendation Engine")
    st.caption("Select 1 or 2 columns and InsightIQ will automatically recommend and render the optimal visualization.")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        rec_x = st.selectbox("Variable X:", list(df.columns), index=0)
    with r_col2:
        rec_y = st.selectbox("Variable Y (Optional):", [None] + list(df.columns), index=0)

    chart_name, recommended_fig = recommend_chart(df, rec_x, rec_y)
    st.markdown(f"**Recommended Visualization:** <span class='pill-badge'>{chart_name}</span>", unsafe_allow_html=True)
    st.plotly_chart(recommended_fig, use_container_width=True)


# ------------------------------------------
# TAB 4: EVIDENCE-FIRST AI ANALYST
# ------------------------------------------
with tab_analyst:
    st.markdown("### 🤖 Evidence-First Conversational AI Analyst")
    st.markdown(
        "Ask questions in natural language. InsightIQ translates your question into a constrained execution plan, "
        "validates it against a strict operation whitelist, computes the answer deterministically with Pandas, and explains the verified facts."
    )

    # Demo Quick-Questions
    st.markdown("##### 💡 5 Verified Demo Queries (Click to test):")
    demo_questions = [
        "Which products generate the highest revenue?",
        "What is the total revenue?",
        "What is the revenue trend over time?",
        "Is marketing spend correlated with revenue?",
        "Compare revenue between North and South regions",
    ]
    
    btn_cols = st.columns(len(demo_questions))
    selected_query = None
    for idx, q_text in enumerate(demo_questions):
        with btn_cols[idx]:
            if st.button(q_text, key=f"demo_btn_{idx}", use_container_width=True):
                selected_query = q_text

    user_query = st.text_input(
        "Or type your question:",
        value=selected_query or "",
        placeholder="e.g. Which region generates highest revenue?",
    )

    if st.button("Ask InsightIQ Analyst", type="primary") or selected_query:
        query_to_run = user_query.strip()
        if not query_to_run:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing verified dataset..."):
                res = ask_analyst(query_to_run, df)

            st.markdown("---")
            if res.is_success:
                # Conclusion Box
                st.markdown("#### 🎯 Conclusion")
                st.success(res.conclusion)

                # Calculation Evidence & Operation
                evi_col1, evi_col2 = st.columns([0.6, 0.4])
                with evi_col1:
                    st.markdown("#### 📐 Calculation Performed (Verified Proof)")
                    st.code(res.calculation_performed, language="python")
                    st.markdown(f"**Columns Used:** `{', '.join(res.columns_used)}`")
                    st.markdown(f"**Limitations:** {res.limitations}")
                    st.info(f"🔮 **Suggested Next Investigation:** {res.suggested_next_question}")

                with evi_col2:
                    if res.figure:
                        st.markdown("#### 📈 Evidence Visualization")
                        st.plotly_chart(res.figure, use_container_width=True)

                if res.used_fallback:
                    st.caption("ℹ️ *Executed via Deterministic Rule & Template Engine (100% verified math).*")
            else:
                st.error(f"❌ Query Rejected: {res.conclusion}")
                st.info(f"💡 {res.suggested_next_question}")


# ------------------------------------------
# TAB 5: SYSTEM ARCHITECTURE & AUDIT LOG
# ------------------------------------------
with tab_arch:
    st.markdown("### 🛡️ System Architecture & Defense Blueprint")

    st.markdown("""
    #### 1. Decoupled Core Architecture
    InsightIQ separates business math from presentation. All data intelligence algorithms reside in `core/` 
    with zero dependency on the Streamlit web framework.
    """)

    st.code("""
InsightIQ Architecture:
User Query / Upload -> Presentation Layer (Streamlit / FastAPI + React)
        |
        v
[core/ingestion.py]  --> Type & Size Allowlist Validation, Encoding Fallbacks
[core/profiler.py]   --> 0-100 Health Score, Missing, IQR Outliers, Format Checks
[core/cleaner.py]    --> Non-Destructive Recipes, Diff Engine, Version Manager
[core/eda.py]        --> Plotly Stats, Correlations, Auto-Chart Selector
[core/analyst/]      --> LLM Planner -> Safety Whitelist Validator -> Pandas Executor -> Explainer
    """, language="text")

    st.markdown("#### 2. Dataset Version History")
    all_versions = manager.list_versions()
    st.table(pd.DataFrame(all_versions))

    st.markdown("#### 3. Security & Anti-Hallucination Controls")
    security_controls = [
        {"Control": "No Arbitrary Code Execution", "Status": "Active", "Defense": "No eval() or exec(); operations execute only through whitelisted Pandas logic."},
        {"Control": "Numerical Truth Guarantee", "Status": "Active", "Defense": "LLM never computes arithmetic; numbers are derived deterministically."},
        {"Control": "Secret Hygiene", "Status": "Active", "Defense": "API keys stored strictly in .env; untracked by git."},
        {"Control": "DoS & Payload Protection", "Status": "Active", "Defense": "25 MB file size limit and strict extension allowlists."},
        {"Control": "Dual-Path Fallback", "Status": "Active", "Defense": "System functions with 100% precision even if external LLM APIs fail."},
    ]
    st.table(pd.DataFrame(security_controls))
