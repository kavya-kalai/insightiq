"""InsightIQ Plan Validator.

Enforces strict backend safety:
- Whitelist of permitted operations only.
- Validates columns exist in the active dataset schema.
- Enforces semantic datatype compatibility (e.g. numeric targets for sum/mean).
- Zero arbitrary code execution permitted.
"""

from __future__ import annotations
import pandas as pd
from core.analyst.schema import ExecutionPlan
from core.profiler import infer_semantic_type


ALLOWED_OPERATIONS = {"aggregate", "ranking", "trend", "correlation", "comparison"}


def validate_plan(plan: ExecutionPlan, df: pd.DataFrame) -> tuple[bool, str]:
    """Validate an execution plan against the DataFrame's schema and safety rules.
    
    Returns:
        (is_valid, error_reason)
    """
    if not plan.is_supported or plan.operation == "unsupported":
        reason = plan.unsupported_reason or (
            "The question cannot be safely mapped to our supported operations "
            "(aggregate, ranking, trend over time, correlation, group comparison)."
        )
        return False, reason

    if plan.operation not in ALLOWED_OPERATIONS:
        return False, f"Operation '{plan.operation}' is not in the approved safety whitelist."

    columns_set = set(df.columns)

    # 1. Validate Filters
    for filt in plan.filters:
        if filt.column not in columns_set:
            return False, f"Filter column '{filt.column}' does not exist in dataset."

    # 2. Validate Operation-Specific Requirements
    if plan.operation == "aggregate":
        if not plan.target_column or plan.target_column not in columns_set:
            return False, f"Target column '{plan.target_column}' does not exist in dataset."
        sem = infer_semantic_type(df[plan.target_column])
        if plan.agg_func in {"sum", "mean", "median"} and sem != "numeric":
            return False, f"Cannot perform '{plan.agg_func}' on non-numeric column '{plan.target_column}'."

    elif plan.operation == "ranking":
        if not plan.group_column or plan.group_column not in columns_set:
            return False, f"Grouping column '{plan.group_column}' does not exist in dataset."
        if not plan.target_column or plan.target_column not in columns_set:
            return False, f"Ranking target column '{plan.target_column}' does not exist in dataset."
        if plan.top_n <= 0:
            plan.top_n = 5

    elif plan.operation == "trend":
        # Find or validate date column
        if not plan.date_column or plan.date_column not in columns_set:
            # Attempt to auto-find a datetime column
            date_candidates = [c for c in df.columns if infer_semantic_type(df[c]) == "datetime"]
            if date_candidates:
                plan.date_column = date_candidates[0]
            else:
                return False, "No valid date/time column found in dataset for trend analysis."
        if not plan.target_column or plan.target_column not in columns_set:
            return False, f"Trend metric column '{plan.target_column}' does not exist in dataset."

    elif plan.operation == "correlation":
        if not plan.col1 or plan.col1 not in columns_set:
            return False, f"First correlation column '{plan.col1}' does not exist in dataset."
        if not plan.col2 or plan.col2 not in columns_set:
            return False, f"Second correlation column '{plan.col2}' does not exist in dataset."
        if infer_semantic_type(df[plan.col1]) != "numeric" or infer_semantic_type(df[plan.col2]) != "numeric":
            return False, f"Both columns ('{plan.col1}', '{plan.col2}') must be numeric for correlation."

    elif plan.operation == "comparison":
        if not plan.group_column or plan.group_column not in columns_set:
            return False, f"Grouping column '{plan.group_column}' does not exist in dataset."
        if not plan.target_column or plan.target_column not in columns_set:
            return False, f"Comparison metric '{plan.target_column}' does not exist in dataset."
        if len(plan.group_values) < 2:
            # Attempt to take top 2 distinct values from group column
            top_groups = df[plan.group_column].dropna().astype(str).unique().tolist()
            if len(top_groups) >= 2:
                plan.group_values = top_groups[:2]
            else:
                return False, f"Column '{plan.group_column}' does not have at least 2 distinct values to compare."

    return True, "Plan verified and approved for execution."

