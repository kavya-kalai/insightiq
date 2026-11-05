"""InsightIQ AI Analyst Schemas & Data Contracts."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel


OperationType = Literal[
    "aggregate",
    "ranking",
    "trend",
    "correlation",
    "comparison",
    "unsupported",
]


class FilterCondition(BaseModel):  # type: ignore
    """Safe equality/membership filter applied before execution."""
    column: str
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in"] = "=="
    value: Any = None


class ExecutionPlan(BaseModel):  # type: ignore
    """Constrained execution plan produced by the planner and validated by backend."""
    operation: OperationType = "unsupported"
    target_column: str | None = None
    group_column: str | None = None
    date_column: str | None = None
    agg_func: Literal["sum", "mean", "count", "median", "min", "max"] = "sum"
    top_n: int = 5
    group_values: list[str] = []
    col1: str | None = None
    col2: str | None = None
    filters: list[FilterCondition] = []
    reasoning: str = ""
    is_supported: bool = True
    unsupported_reason: str | None = None


@dataclass
class AnalysisResult:
    """Complete, verified response returned to the user/UI."""
    question: str
    is_success: bool
    operation: str
    conclusion: str
    calculation_performed: str
    computed_data: Any
    columns_used: list[str] = field(default_factory=list)
    limitations: str = ""
    suggested_next_question: str = ""
    figure: Any = None  # Interactive Plotly Figure
    used_fallback: bool = False
    error_message: str | None = None
