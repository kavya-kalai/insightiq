"""InsightIQ AI Analyst Schemas & Data Contracts."""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


OperationType = Literal[
    "aggregate",
    "ranking",
    "trend",
    "correlation",
    "comparison",
    "unsupported",
]


class FilterCondition(BaseModel):
    """Safe equality/membership filter applied before execution."""
    column: str
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in"] = "=="
    value: Any


class ExecutionPlan(BaseModel):
    """Constrained execution plan produced by the planner and validated by backend."""
    operation: OperationType
    target_column: str | None = None
    group_column: str | None = None
    date_column: str | None = None
    agg_func: Literal["sum", "mean", "count", "median", "min", "max"] = "sum"
    top_n: int = 5
    group_values: list[str] = Field(default_factory=list)
    col1: str | None = None
    col2: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    reasoning: str = ""
    is_supported: bool = True
    unsupported_reason: str | None = None


class AnalysisResult(BaseModel):
    """Complete, verified response returned to the user/UI."""
    model_config = {"arbitrary_types_allowed": True}

    question: str
    is_success: bool
    operation: str
    conclusion: str
    calculation_performed: str
    computed_data: Any
    columns_used: list[str]
    limitations: str
    suggested_next_question: str
    figure: Any = None  # Plotly Figure
    used_fallback: bool = False
    error_message: str | None = None
