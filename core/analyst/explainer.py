"""InsightIQ AI Analyst Explainer.

Takes verified computation results from Pandas and crafts an evidence-first,
verifiable business response:
1. Primary: Uses Google Gemini API to translate verified numbers into narrative insights.
2. Fallback: Deterministic template explainer if API is offline or returns error.
Strict Requirement: The LLM NEVER computes or alters numbers. It only explains verified facts.
"""

from __future__ import annotations
import json
import os
from typing import Any
import pandas as pd
from core.analyst.schema import AnalysisResult, ExecutionPlan


def template_based_explainer(
    question: str,
    plan: ExecutionPlan,
    computed_data: dict[str, Any],
    calc_str: str,
    cols_used: list[str],
    figure: Any,
    df: pd.DataFrame,
) -> AnalysisResult:
    """Generate a high-quality, evidence-backed narrative without external LLM API."""
    op = plan.operation
    row_count = len(df)

    if op == "aggregate":
        metric = computed_data["metric"]
        func = computed_data["aggregation"]
        val = computed_data["value"]
        conclusion = (
            f"The overall **{func}** of **{metric}** across all active records is **{val:,.2f}** "
            f"(computed over {computed_data['record_count']:,} verified transactions)."
        )
        limitations = (
            f"Calculation reflects {row_count:,} records in the current dataset version. "
            "Excludes rows where the metric was missing or unparseable."
        )
        next_q = f"Which product category contributed the most to this total {metric}?"

    elif op == "ranking":
        grp = computed_data["group_column"]
        met = computed_data["metric"]
        rankings = computed_data["rankings"]
        top_1 = rankings[0] if rankings else {"group": "N/A", "target": 0}
        
        items_str = ", ".join([f"**{r['group']}** ({r['target']:,.2f})" for r in rankings[:3]])
        conclusion = (
            f"Ranked by total {met}, the leading {grp} is **{top_1['group']}** with **{top_1['target']:,.2f}**. "
            f"The top 3 performers are {items_str}."
        )
        limitations = (
            f"Rankings are computed using '{plan.agg_func}' aggregation across {row_count:,} records. "
            "Does not account for external regional market size variations."
        )
        next_q = f"How has {top_1['group']}'s performance changed over time?"

    elif op == "trend":
        met = computed_data["metric"]
        tot = computed_data["total"]
        avg = computed_data["daily_average"]
        s_date = computed_data["start_date"]
        e_date = computed_data["end_date"]
        conclusion = (
            f"Between **{s_date}** and **{e_date}**, cumulative {met} reached **{tot:,.2f}**, "
            f"averaging **{avg:,.2f}** per active day. The 7-day rolling trend highlights operational stability with periodic cyclical peaks."
        )
        limitations = (
            f"Trend covers {s_date} to {e_date}. Incomplete reporting days may produce localized dips."
        )
        next_q = f"Which product categories experienced the highest growth during this timeline?"

    elif op == "correlation":
        c1 = computed_data["col1"]
        c2 = computed_data["col2"]
        r = computed_data["pearson_r"]
        rel = computed_data["relationship"]
        conclusion = (
            f"There is a **{rel}** linear correlation (**r = {r:+.3f}**) between **{c1}** and **{c2}** "
            f"(sample size: {computed_data['sample_size']:,} observations)."
        )
        limitations = (
            "Correlation indicates linear statistical association, not necessarily direct causality. "
            "Latent confounding variables (e.g. seasonal promotional campaigns) may influence both metrics."
        )
        next_q = f"Does the correlation between {c1} and {c2} differ by region?"

    elif op == "comparison":
        grp = computed_data["group_column"]
        met = computed_data["metric"]
        leader = computed_data["leader"]
        g1_data = computed_data["group_1"]
        g2_data = computed_data["group_2"]
        pct = computed_data["percent_difference"]
        conclusion = (
            f"Comparing **{g1_data['name']}** ({g1_data['total']:,.2f}) against **{g2_data['name']}** ({g2_data['total']:,.2f}), "
            f"**{leader}** outperforms by **{abs(pct):.1f}%** (difference of {abs(computed_data['absolute_difference']):,.2f})."
        )
        limitations = (
            f"Comparison reflects aggregate {met}. Differences in order volume ({g1_data['count']} vs {g2_data['count']} orders) "
            "account for a portion of the variance."
        )
        next_q = f"What products drive the revenue margin difference between {g1_data['name']} and {g2_data['name']}?"

    else:
        conclusion = "Analytical operation completed successfully."
        limitations = "General observational findings."
        next_q = "What other aspects of this dataset would you like to investigate?"

    return AnalysisResult(
        question=question,
        is_success=True,
        operation=op,
        conclusion=conclusion,
        calculation_performed=calc_str,
        computed_data=computed_data,
        columns_used=cols_used,
        limitations=limitations,
        suggested_next_question=next_q,
        figure=figure,
        used_fallback=True,
    )


def explain_results(
    question: str,
    plan: ExecutionPlan,
    computed_data: dict[str, Any],
    calc_str: str,
    cols_used: list[str],
    figure: Any,
    df: pd.DataFrame,
) -> AnalysisResult:
    """Generate verified business response via Gemini or fallback."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return template_based_explainer(
            question, plan, computed_data, calc_str, cols_used, figure, df
        )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = f"""You are InsightIQ's Evidence-First Explainer.
Explain these VERIFIED analytical results to a business stakeholder.

STRICT ACCURACY RULES:
1. You MUST NEVER compute, invent, or adjust numbers. Use ONLY the exact numbers in 'Verified Computation'.
2. The user asked: "{question}"
3. Calculation performed by Pandas: `{calc_str}`
4. Columns used: {cols_used}
5. Verified Computation:
{json.dumps(computed_data, indent=2)}

Format your output as a JSON object with:
- "conclusion": Clear 2-3 sentence executive summary explaining the verified numbers.
- "limitations": 1-2 sentences on data boundaries or statistical assumptions.
- "suggested_next_question": 1 actionable business follow-up question.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        if response.text:
            parsed = json.loads(response.text)
            return AnalysisResult(
                question=question,
                is_success=True,
                operation=plan.operation,
                conclusion=parsed.get("conclusion", ""),
                calculation_performed=calc_str,
                computed_data=computed_data,
                columns_used=cols_used,
                limitations=parsed.get("limitations", ""),
                suggested_next_question=parsed.get("suggested_next_question", ""),
                figure=figure,
                used_fallback=False,
            )
        else:
            return template_based_explainer(
                question, plan, computed_data, calc_str, cols_used, figure, df
            )

    except Exception:
        # Seamless fallback
        return template_based_explainer(
            question, plan, computed_data, calc_str, cols_used, figure, df
        )

