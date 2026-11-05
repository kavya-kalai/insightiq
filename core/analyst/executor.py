"""InsightIQ Plan Executor.

Executes validated analytical operations purely through deterministic Pandas operations:
- No eval(), no exec(), no arbitrary code execution.
- Computes exact mathematical figures from the active dataset.
- Produces corresponding Plotly interactive visualizations.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.analyst.schema import ExecutionPlan
from core.eda import PLOT_THEME, PRIMARY_COLOR, create_time_series_trend


def apply_plan_filters(df: pd.DataFrame, plan: ExecutionPlan) -> pd.DataFrame:
    """Safely apply structured filter conditions without dynamic string evaluation."""
    filtered_df = df.copy()
    for f in plan.filters:
        col = f.column
        op = f.operator
        val = f.value
        if col not in filtered_df.columns:
            continue

        if op == "==":
            filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]
        elif op == "!=":
            filtered_df = filtered_df[filtered_df[col].astype(str) != str(val)]
        elif op == ">":
            filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") > float(val)]
        elif op == "<":
            filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") < float(val)]
        elif op == ">=":
            filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") >= float(val)]
        elif op == "<=":
            filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") <= float(val)]
        elif op == "in" and isinstance(val, list):
            filtered_df = filtered_df[filtered_df[col].astype(str).isin([str(v) for v in val])]

    return filtered_df


def execute_plan(
    df: pd.DataFrame,
    plan: ExecutionPlan,
) -> tuple[dict[str, Any], str, list[str], go.Figure | None]:
    """Execute a validated plan deterministically on the DataFrame.
    
    Returns:
        (computed_data, calculation_performed, columns_used, figure)
    """
    work_df = apply_plan_filters(df, plan)
    op = plan.operation

    if op == "aggregate":
        target = plan.target_column or df.columns[0]
        func = plan.agg_func
        num_s = pd.to_numeric(work_df[target], errors="coerce").dropna()
        
        if func == "sum":
            val = round(float(num_s.sum()), 2)
        elif func == "mean":
            val = round(float(num_s.mean()), 2)
        elif func == "median":
            val = round(float(num_s.median()), 2)
        elif func == "min":
            val = round(float(num_s.min()), 2)
        elif func == "max":
            val = round(float(num_s.max()), 2)
        else:
            val = int(len(num_s))

        computed_data = {
            "metric": target,
            "aggregation": func,
            "value": val,
            "record_count": len(num_s),
        }
        calc_str = f"df['{target}'].dropna().{func}() -> {val:,}"
        cols_used = [target]

        # Simple bar / KPI chart
        fig = go.Figure(go.Indicator(
            mode="number+gauge",
            value=val,
            title={"text": f"{func.upper()} of {target}"},
        ))
        fig.update_layout(template=PLOT_THEME, height=300)
        return computed_data, calc_str, cols_used, fig

    elif op == "ranking":
        group_col = plan.group_column or df.columns[0]
        target_col = plan.target_column or df.columns[1]
        top_n = plan.top_n
        func = plan.agg_func

        temp_df = pd.DataFrame({
            "group": work_df[group_col].astype(str),
            "target": pd.to_numeric(work_df[target_col], errors="coerce"),
        }).dropna()

        grouped = temp_df.groupby("group")["target"].agg(func).reset_index()
        ranked = grouped.sort_values("target", ascending=False).head(top_n).reset_index(drop=True)
        ranked["target"] = ranked["target"].round(2)

        items = ranked.to_dict(orient="records")
        computed_data = {
            "group_column": group_col,
            "metric": target_col,
            "top_n": top_n,
            "rankings": items,
        }
        calc_str = f"df.groupby('{group_col}')['{target_col}'].{func}().sort_values(ascending=False).head({top_n})"
        cols_used = [group_col, target_col]

        # Horizontal Bar Chart
        plot_df = ranked.sort_values("target", ascending=True)
        fig = px.bar(
            plot_df,
            x="target",
            y="group",
            orientation="h",
            title=f"Top {top_n} {group_col} by {target_col} ({func})",
            labels={"target": f"{func.capitalize()} of {target_col}", "group": group_col},
            color="target",
            color_continuous_scale="Blues",
            template=PLOT_THEME,
        )
        fig.update_layout(margin={"l": 40, "r": 40, "t": 50, "b": 40}, coloraxis_showscale=False)
        return computed_data, calc_str, cols_used, fig

    elif op == "trend":
        date_col = plan.date_column or "date"
        target_col = plan.target_column or "revenue"
        func = plan.agg_func

        temp_df = pd.DataFrame({
            "date": pd.to_datetime(work_df[date_col], errors="coerce"),
            "target": pd.to_numeric(work_df[target_col], errors="coerce"),
        }).dropna()

        grouped = temp_df.groupby(temp_df["date"].dt.date)["target"].agg(func).reset_index()
        grouped.columns = ["date", "val"]
        grouped = grouped.sort_values("date").reset_index(drop=True)

        total_period_val = round(float(grouped["val"].sum()), 2)
        avg_daily_val = round(float(grouped["val"].mean()), 2)

        computed_data = {
            "date_column": date_col,
            "metric": target_col,
            "total": total_period_val,
            "daily_average": avg_daily_val,
            "start_date": str(grouped["date"].min()),
            "end_date": str(grouped["date"].max()),
        }
        calc_str = f"df.groupby(pd.to_datetime('{date_col}').dt.date)['{target_col}'].{func}()"
        cols_used = [date_col, target_col]

        fig = create_time_series_trend(work_df, date_col, target_col, agg_func=func)
        return computed_data, calc_str, cols_used, fig

    elif op == "correlation":
        col1 = plan.col1 or df.columns[0]
        col2 = plan.col2 or df.columns[1]

        clean_corr_df = pd.DataFrame({
            col1: pd.to_numeric(work_df[col1], errors="coerce"),
            col2: pd.to_numeric(work_df[col2], errors="coerce"),
        }).dropna()

        r_val = round(float(clean_corr_df[col1].corr(clean_corr_df[col2])), 3)
        computed_data = {
            "col1": col1,
            "col2": col2,
            "pearson_r": r_val,
            "relationship": "Strong Positive" if r_val >= 0.7 else (
                "Moderate Positive" if r_val >= 0.3 else (
                    "Weak / Negligible" if r_val > -0.3 else (
                        "Moderate Negative" if r_val > -0.7 else "Strong Negative"
                    )
                )
            ),
            "sample_size": len(clean_corr_df),
        }
        calc_str = f"df['{col1}'].corr(df['{col2}']) -> r = {r_val}"
        cols_used = [col1, col2]

        # Scatter plot
        fig = px.scatter(
            clean_corr_df.sample(min(len(clean_corr_df), 1500), random_state=42),
            x=col1,
            y=col2,
            title=f"Correlation Analysis: {col1} vs {col2} (r = {r_val})",
            template=PLOT_THEME,
            opacity=0.6,
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        # Linear fit line
        if len(clean_corr_df) >= 2:
            x_vals = clean_corr_df[col1].values
            y_vals = clean_corr_df[col2].values
            if len(np.unique(x_vals)) > 1:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                line_x = np.array([x_vals.min(), x_vals.max()])
                line_y = slope * line_x + intercept
                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=f"Trend (slope={slope:.2f})",
                    line={"color": "#EF4444", "width": 2, "dash": "dash"},
                ))
        return computed_data, calc_str, cols_used, fig

    elif op == "comparison":
        group_col = plan.group_column or df.columns[0]
        target_col = plan.target_column or df.columns[1]
        g1, g2 = plan.group_values[0], plan.group_values[1]

        d1 = work_df[work_df[group_col].astype(str) == str(g1)]
        d2 = work_df[work_df[group_col].astype(str) == str(g2)]

        val1 = round(float(pd.to_numeric(d1[target_col], errors="coerce").sum()), 2)
        val2 = round(float(pd.to_numeric(d2[target_col], errors="coerce").sum()), 2)
        diff = round(val1 - val2, 2)
        pct_diff = round(((val1 - val2) / max(0.01, val2)) * 100, 2)

        computed_data = {
            "group_column": group_col,
            "metric": target_col,
            "group_1": {"name": g1, "total": val1, "count": len(d1)},
            "group_2": {"name": g2, "total": val2, "count": len(d2)},
            "absolute_difference": diff,
            "percent_difference": pct_diff,
            "leader": g1 if val1 >= val2 else g2,
        }
        calc_str = (
            f"Comparison of sum('{target_col}') by '{group_col}': "
            f"{g1}={val1:,} vs {g2}={val2:,} (Delta = {pct_diff:+.1f}%)"
        )
        cols_used = [group_col, target_col]

        comp_df = pd.DataFrame([
            {"Group": g1, "Value": val1},
            {"Group": g2, "Value": val2},
        ])
        fig = px.bar(
            comp_df,
            x="Group",
            y="Value",
            title=f"Comparison: {g1} vs. {g2} by {target_col}",
            color="Group",
            color_discrete_sequence=[PRIMARY_COLOR, "#0D9488"],
            template=PLOT_THEME,
        )
        fig.update_layout(showlegend=False, margin={"l": 40, "r": 40, "t": 50, "b": 40})
        return computed_data, calc_str, cols_used, fig

    raise ValueError(f"Unsupported operation '{op}'")
