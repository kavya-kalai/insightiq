"""InsightIQ AI Analyst Planner.

Translates user natural language questions into safe, structured JSON ExecutionPlans.
1. Primary: Uses Google Gemini API (gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash) with response_schema.
2. Fallback: If API key is absent, network fails, or quota/permissions fail, activates
   the Deterministic Semantic Fallback Parser to guarantee zero demo crashes.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any
import pandas as pd
from core.analyst.schema import ExecutionPlan
from core.profiler import infer_semantic_type


def rule_based_fallback_planner(question: str, df: pd.DataFrame) -> ExecutionPlan:
    """Deterministic regex-based semantic planner used when LLM API is unreachable."""
    q_lower = question.lower().strip()
    columns = list(df.columns)
    col_types = {c: infer_semantic_type(df[c]) for c in columns}
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in {"categorical", "identifier/text"}]
    date_cols = [c for c, t in col_types.items() if t == "datetime"]

    # 1. Operation: Correlation
    if any(k in q_lower for k in ["correlat", "relationship between", "depend on", "vs"]):
        # Look for two numeric columns mentioned
        matched_nums = [c for c in numeric_cols if c.lower() in q_lower or c.replace("_", " ") in q_lower]
        if len(matched_nums) >= 2:
            return ExecutionPlan(
                operation="correlation",
                col1=matched_nums[0],
                col2=matched_nums[1],
                reasoning="Identified correlation query between two numeric attributes.",
            )
        elif len(numeric_cols) >= 2:
            # Default to first two numeric columns (e.g. marketing_spend & revenue)
            return ExecutionPlan(
                operation="correlation",
                col1=numeric_cols[0],
                col2=numeric_cols[1],
                reasoning="Correlation query detected; analyzing primary numeric columns.",
            )

    # 2. Operation: Group Comparison (e.g. "compare North and South")
    if "compare" in q_lower or "difference between" in q_lower or " vs " in q_lower:
        # Check if any categories are explicitly named
        for cat_col in cat_cols:
            unique_vals = [str(v) for v in df[cat_col].dropna().unique()]
            matched_vals = [v for v in unique_vals if v.lower() in q_lower]
            if len(matched_vals) >= 2:
                target_metric = next((c for c in numeric_cols if c.lower() in q_lower), numeric_cols[0] if numeric_cols else "revenue")
                return ExecutionPlan(
                    operation="comparison",
                    group_column=cat_col,
                    group_values=matched_vals[:2],
                    target_column=target_metric,
                    reasoning=f"Comparison query between groups {matched_vals[:2]} on {target_metric}.",
                )

    # 3. Operation: Trend Over Time
    if any(k in q_lower for k in ["trend", "over time", "monthly", "daily", "timeline", "sales over"]):
        date_col = date_cols[0] if date_cols else "date"
        target_metric = next((c for c in numeric_cols if c.lower() in q_lower), numeric_cols[0] if numeric_cols else "revenue")
        return ExecutionPlan(
            operation="trend",
            date_column=date_col,
            target_column=target_metric,
            agg_func="sum",
            reasoning=f"Time-series trend analysis of {target_metric} along {date_col}.",
        )

    # 4. Operation: Ranking / Top N (e.g. "which products generate highest revenue?")
    if any(k in q_lower for k in ["top", "highest", "best", "most", "rank", "which product", "which region"]):
        # Identify grouping column
        group_col = next((c for c in cat_cols if c.lower() in q_lower), cat_cols[0] if cat_cols else "product")
        target_metric = next((c for c in numeric_cols if c.lower() in q_lower), numeric_cols[0] if numeric_cols else "revenue")
        
        # Check for specific Top N number
        num_match = re.search(r"top\s+(\d+)", q_lower)
        top_n = int(num_match.group(1)) if num_match else 5

        return ExecutionPlan(
            operation="ranking",
            group_column=group_col,
            target_column=target_metric,
            agg_func="sum",
            top_n=top_n,
            reasoning=f"Ranking query for top {top_n} {group_col} by {target_metric}.",
        )

    # 5. Operation: Simple Aggregation (e.g. "total revenue", "average price")
    if any(k in q_lower for k in ["total", "sum", "average", "mean", "count", "overall", "how much"]):
        target_metric = next((c for c in numeric_cols if c.lower() in q_lower), numeric_cols[0] if numeric_cols else "revenue")
        agg_func = "mean" if any(w in q_lower for w in ["average", "avg", "mean"]) else "sum"
        return ExecutionPlan(
            operation="aggregate",
            target_column=target_metric,
            agg_func=agg_func,
            reasoning=f"Aggregation query calculating {agg_func} of {target_metric}.",
        )

    # Unsupported query
    return ExecutionPlan(
        operation="unsupported",
        is_supported=False,
        unsupported_reason=(
            f"InsightIQ could not map your query '{question}' to a supported operation "
            "(aggregate, ranking, trend over time, correlation, or group comparison). "
            "Please try asking about totals, top products/regions, trends over time, or correlations."
        ),
        reasoning="Query did not match supported analytical operations.",
    )


def plan_query(question: str, df: pd.DataFrame) -> tuple[ExecutionPlan, bool]:
    """Generate structured ExecutionPlan using Gemini with automatic fallback.
    
    Returns:
        (plan, used_fallback)
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return rule_based_fallback_planner(question, df), True

    # Attempt Gemini API
    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        col_summary = []
        for c in df.columns:
            sem = infer_semantic_type(df[c])
            col_summary.append(f"- {c} ({sem})")
        schema_text = "\n".join(col_summary)

        prompt = f"""You are InsightIQ's Query Planner.
Your role: Convert a user question into a strict JSON execution plan.
Allowed operations: 'aggregate', 'ranking', 'trend', 'correlation', 'comparison', 'unsupported'.

Dataset Schema:
{schema_text}

Rules:
1. NEVER invent columns. Only use columns from the schema.
2. If the user asks something outside the 5 operations, set operation to 'unsupported' and is_supported to false.
3. For 'ranking', provide group_column, target_column, and top_n.
4. For 'trend', provide date_column and target_column.
5. For 'correlation', provide col1 and col2 (both must be numeric).
6. For 'comparison', provide group_column, target_column, and group_values (2 values).
7. For 'aggregate', provide target_column and agg_func.

User Question: "{question}"
Return ONLY a valid JSON object matching the ExecutionPlan schema.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ExecutionPlan,
            },
        )
        if response.text:
            plan_data = json.loads(response.text)
            return ExecutionPlan(**plan_data), False
        else:
            return rule_based_fallback_planner(question, df), True

    except Exception:
        # Graceful fallback on 403, 429, connection errors, or JSON decode failure
        return rule_based_fallback_planner(question, df), True
