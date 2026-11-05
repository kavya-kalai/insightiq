"""InsightIQ Automated Exploratory Data Analysis (EDA) Engine.

Generates statistical summaries, interactive Plotly visualizations, and
implements rule-based automatic chart selection based on column types:
1. Descriptive summary statistics (central tendency, dispersion, shape).
2. Distribution analysis (histograms with marginal box plots).
3. Pearson correlation matrix and interactive heatmap.
4. Time-series trend identification with moving averages.
5. Categorical ranking (top N categories by volume or metric sum).
6. Automatic chart recommendation based on variable data types.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.profiler import infer_semantic_type


PLOT_THEME = "plotly_white"
PRIMARY_COLOR = "#2563EB"  # Professional Indigo/Blue
ACCENT_COLOR = "#0D9488"   # Teal


def compute_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate descriptive statistics for numeric and categorical columns."""
    stats: dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    for col in df.columns:
        series = df[col]
        sem_type = infer_semantic_type(series)

        if sem_type == "numeric":
            num_s = pd.to_numeric(series, errors="coerce").dropna()
            if not num_s.empty:
                stats["numeric"][col] = {
                    "count": int(num_s.count()),
                    "mean": round(float(num_s.mean()), 2),
                    "std": round(float(num_s.std()), 2) if len(num_s) > 1 else 0.0,
                    "min": round(float(num_s.min()), 2),
                    "q25": round(float(num_s.quantile(0.25)), 2),
                    "median": round(float(num_s.median()), 2),
                    "q75": round(float(num_s.quantile(0.75)), 2),
                    "max": round(float(num_s.max()), 2),
                    "skewness": round(float(num_s.skew()), 2) if len(num_s) > 2 else 0.0,
                    "kurtosis": round(float(num_s.kurt()), 2) if len(num_s) > 3 else 0.0,
                }
        elif sem_type in {"categorical", "identifier/text"}:
            cat_s = series.dropna().astype(str)
            if not cat_s.empty:
                top_mode = cat_s.mode().iloc[0] if not cat_s.mode().empty else "N/A"
                top_freq = int((cat_s == top_mode).sum())
                stats["categorical"][col] = {
                    "count": int(cat_s.count()),
                    "unique": int(cat_s.nunique()),
                    "top": top_mode,
                    "top_frequency": top_freq,
                    "top_percentage": round((top_freq / len(cat_s)) * 100, 2),
                }

    return stats


def compute_correlation_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calculate Pearson correlation matrix and identify top correlated column pairs."""
    # Isolate numeric columns
    numeric_cols: list[str] = []
    num_df = pd.DataFrame()

    for col in df.columns:
        if infer_semantic_type(df[col]) == "numeric":
            numeric_cols.append(col)
            num_df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(numeric_cols) < 2:
        return pd.DataFrame(), []

    corr_df = num_df.corr(method="pearson").round(3)

    # Extract distinct non-diagonal pairs ranked by absolute correlation
    pairs: list[dict[str, Any]] = []
    seen = set()
    for col1 in corr_df.columns:
        for col2 in corr_df.columns:
            if col1 != col2 and (col2, col1) not in seen:
                seen.add((col1, col2))
                r_val = corr_df.loc[col1, col2]
                if not np.isnan(r_val):
                    pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "correlation": round(float(r_val), 3),
                        "abs_correlation": round(abs(float(r_val)), 3),
                    })

    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return corr_df, pairs


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generate interactive correlation heatmap."""
    corr_df, _ = compute_correlation_matrix(df)

    if corr_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient numeric columns for correlation matrix",
            showarrow=False,
            font={"size": 14},
        )
        return fig

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Pearson Correlation Heatmap",
        template=PLOT_THEME,
    )
    fig.update_layout(
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_distribution_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Generate distribution histogram with interactive box plot marginal."""
    num_s = pd.to_numeric(df[column], errors="coerce").dropna()
    if num_s.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No numeric data in column '{column}'", showarrow=False)
        return fig

    fig = px.histogram(
        x=num_s,
        nbins=35,
        marginal="box",
        title=f"Distribution & Outlier Spread: {column}",
        labels={"x": column, "count": "Frequency"},
        color_discrete_sequence=[PRIMARY_COLOR],
        template=PLOT_THEME,
    )
    fig.update_layout(
        bargap=0.08,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg_func: str = "sum",
) -> go.Figure:
    """Generate a time-series trend line chart with a 7-day rolling average."""
    temp_df = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna()

    if temp_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid date-value pairs to plot trend", showarrow=False)
        return fig

    # Group by daily date
    grouped = temp_df.groupby(temp_df["date"].dt.date)["value"].agg(agg_func).reset_index()
    grouped.columns = ["date", "metric_value"]
    grouped = grouped.sort_values("date").reset_index(drop=True)

    # 7-day rolling trend
    grouped["rolling_avg_7d"] = grouped["metric_value"].rolling(window=7, min_periods=1).mean().round(2)

    fig = go.Figure()
    # Daily bar or light line
    fig.add_trace(go.Scatter(
        x=grouped["date"],
        y=grouped["metric_value"],
        mode="lines",
        name=f"Daily {agg_func.capitalize()} ({value_col})",
        line={"color": "#94A3B8", "width": 1.5},
        opacity=0.6,
    ))
    # Rolling Trend Line
    fig.add_trace(go.Scatter(
        x=grouped["date"],
        y=grouped["rolling_avg_7d"],
        mode="lines",
        name="7-Day Rolling Trend",
        line={"color": PRIMARY_COLOR, "width": 3},
    ))

    fig.update_layout(
        title=f"Time-Series Trend: {value_col} over Time",
        xaxis_title="Date",
        yaxis_title=f"{agg_func.capitalize()} of {value_col}",
        template=PLOT_THEME,
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_top_categories_chart(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str | None = None,
    top_n: int = 8,
) -> go.Figure:
    """Generate ranked horizontal bar chart of top categories."""
    if val_col and val_col in df:
        temp_df = pd.DataFrame({
            "cat": df[cat_col].astype(str),
            "val": pd.to_numeric(df[val_col], errors="coerce"),
        }).dropna()
        grouped = temp_df.groupby("cat")["val"].sum().reset_index()
        grouped = grouped.sort_values("val", ascending=True).tail(top_n)
        fig = px.bar(
            grouped,
            x="val",
            y="cat",
            orientation="h",
            title=f"Top {top_n} {cat_col} by Total {val_col}",
            labels={"val": f"Total {val_col}", "cat": cat_col},
            color="val",
            color_continuous_scale="Blues",
            template=PLOT_THEME,
        )
    else:
        counts = df[cat_col].astype(str).value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        counts = counts.sort_values("count", ascending=True).tail(top_n)
        fig = px.bar(
            counts,
            x="count",
            y=cat_col,
            orientation="h",
            title=f"Top {top_n} Categories: {cat_col}",
            labels={"count": "Transaction Count", cat_col: cat_col},
            color="count",
            color_continuous_scale="Teal",
            template=PLOT_THEME,
        )

    fig.update_layout(
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        font={"family": "Inter, sans-serif"},
        coloraxis_showscale=False,
    )
    return fig


def recommend_chart(
    df: pd.DataFrame,
    col_x: str,
    col_y: str | None = None,
) -> tuple[str, go.Figure]:
    """Rule-based automatic chart recommendation based on column data types.
    
    Returns:
        (chart_type_name, plotly_figure)
    """
    type_x = infer_semantic_type(df[col_x])
    type_y = infer_semantic_type(df[col_y]) if col_y and col_y in df else None

    # Case 1: Single Numeric Variable -> Distribution (Histogram + Box)
    if type_x == "numeric" and type_y is None:
        return "Distribution (Histogram & Box Plot)", create_distribution_chart(df, col_x)

    # Case 2: Single Categorical Variable -> Ranked Bar Chart
    if type_x in {"categorical", "identifier/text"} and type_y is None:
        return "Categorical Frequency Bar Chart", create_top_categories_chart(df, col_x)

    # Case 3: Datetime + Numeric -> Time Series Trend Line
    if col_y and ((type_x == "datetime" and type_y == "numeric") or (type_x == "numeric" and type_y == "datetime")):
        date_col = col_x if type_x == "datetime" else col_y
        num_col = col_y if type_x == "datetime" else col_x
        return "Time-Series Trend Line", create_time_series_trend(df, date_col, num_col)

    # Case 4: Two Numeric Variables -> Scatter Plot with Linear Trend
    if col_y and type_x == "numeric" and type_y == "numeric":
        scatter_df = pd.DataFrame({
            col_x: pd.to_numeric(df[col_x], errors="coerce"),
            col_y: pd.to_numeric(df[col_y], errors="coerce"),
        }).dropna()
        # Sample for rendering performance if > 3000 rows
        if len(scatter_df) > 3000:
            scatter_df = scatter_df.sample(3000, random_state=42)

        fig = px.scatter(
            scatter_df,
            x=col_x,
            y=col_y,
            title=f"Bivariate Scatter: {col_y} vs. {col_x}",
            color_discrete_sequence=[PRIMARY_COLOR],
            opacity=0.7,
            template=PLOT_THEME,
        )
        # Add regression trendline via numpy polyfit if enough points
        if len(scatter_df) >= 2:
            x_vals = scatter_df[col_x].values
            y_vals = scatter_df[col_y].values
            if len(np.unique(x_vals)) > 1:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                line_x = np.array([x_vals.min(), x_vals.max()])
                line_y = slope * line_x + intercept
                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=f"Linear Fit (slope={slope:.2f})",
                    line={"color": "#EF4444", "width": 2, "dash": "dash"},
                ))
        fig.update_layout(margin={"l": 40, "r": 40, "t": 50, "b": 40})
        return "Bivariate Scatter Plot with Linear Trend", fig

    # Case 5: Categorical + Numeric -> Aggregated Bar Chart
    if col_y and ((type_x == "categorical" and type_y == "numeric") or (type_x == "numeric" and type_y == "categorical")):
        cat = col_x if type_x == "categorical" else col_y
        num = col_y if type_x == "categorical" else col_x
        return "Categorical Metric Aggregation", create_top_categories_chart(df, cat, num)

    # Fallback default: Table / Count
    fig = px.bar(
        df[[col_x]].dropna().head(20),
        title=f"Overview of {col_x}",
        template=PLOT_THEME,
    )
    return "Categorical Overview", fig
