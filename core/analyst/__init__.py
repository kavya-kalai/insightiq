"""InsightIQ Evidence-First AI Analyst Engine.

Provides the end-to-end evidence-first conversational pipeline:
User Question -> Structured Plan -> Whitelist Validator -> Pandas Executor -> Explainer.
"""

from __future__ import annotations
import pandas as pd
from core.analyst.executor import execute_plan
from core.analyst.explainer import explain_results
from core.analyst.planner import plan_query
from core.analyst.schema import AnalysisResult, ExecutionPlan
from core.analyst.validator import validate_plan


def ask_analyst(question: str, df: pd.DataFrame) -> AnalysisResult:
    """End-to-end evidence-first analyst pipeline."""
    # 1. Semantic Planning (Gemini with Fallback)
    plan, used_planner_fallback = plan_query(question, df)

    # 2. Safety Whitelist Validation
    is_valid, validation_msg = validate_plan(plan, df)
    if not is_valid:
        return AnalysisResult(
            question=question,
            is_success=False,
            operation=plan.operation,
            conclusion=validation_msg,
            calculation_performed="None (Blocked by Safety Validator)",
            computed_data={},
            columns_used=[],
            limitations="Query rejected by backend safety validation.",
            suggested_next_question="Try asking: 'What are the top 5 products by revenue?' or 'What is total revenue?'",
            figure=None,
            used_fallback=used_planner_fallback,
            error_message=validation_msg,
        )

    # 3. Deterministic Pandas Execution (Zero arbitrary code execution)
    try:
        computed_data, calc_str, cols_used, fig = execute_plan(df, plan)
    except Exception as exc:
        return AnalysisResult(
            question=question,
            is_success=False,
            operation=plan.operation,
            conclusion=f"Execution error while processing dataset: {str(exc)}",
            calculation_performed=f"Failed execution for operation '{plan.operation}'",
            computed_data={},
            columns_used=[],
            limitations="Execution failed during computation.",
            suggested_next_question="Try rephrasing the question or asking about a different column.",
            figure=None,
            used_fallback=True,
            error_message=str(exc),
        )

    # 4. Verified Explanation (Gemini with Fallback)
    result = explain_results(
        question=question,
        plan=plan,
        computed_data=computed_data,
        calc_str=calc_str,
        cols_used=cols_used,
        figure=fig,
        df=df,
    )
    result.used_fallback = result.used_fallback or used_planner_fallback
    return result


__all__ = ["ask_analyst", "ExecutionPlan", "AnalysisResult", "validate_plan", "execute_plan"]
