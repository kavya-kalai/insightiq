"""Unit tests for the Evidence-First AI Analyst Engine."""

from pathlib import Path
import pandas as pd
import pytest
from core.analyst import ask_analyst, validate_plan
from core.analyst.schema import ExecutionPlan
from core.cleaner import apply_cleaning_pipeline, suggest_cleaning_recipes
from core.ingestion import load_dataset
from core.profiler import profile_dataset


@pytest.fixture
def clean_sales_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "sample" / "sales_data_dirty.csv"
    res = load_dataset(sample_path)
    assert res.dataframe is not None
    rep = profile_dataset(res.dataframe)
    recipes = suggest_cleaning_recipes(rep)
    cleaned, _ = apply_cleaning_pipeline(res.dataframe, recipes)
    return cleaned


def test_analyst_aggregate(clean_sales_df: pd.DataFrame):
    res = ask_analyst("What is the total revenue?", clean_sales_df)
    assert res.is_success is True
    assert res.operation == "aggregate"
    assert "revenue" in res.columns_used
    assert res.computed_data["value"] > 0
    assert "df['revenue']" in res.calculation_performed
    assert len(res.conclusion) > 10
    assert res.figure is not None


def test_analyst_ranking(clean_sales_df: pd.DataFrame):
    res = ask_analyst("Which products generate the highest revenue?", clean_sales_df)
    assert res.is_success is True
    assert res.operation == "ranking"
    assert "product" in res.columns_used
    assert "revenue" in res.columns_used
    assert len(res.computed_data["rankings"]) >= 3
    # First item must have higher or equal revenue than second item
    r = res.computed_data["rankings"]
    assert r[0]["target"] >= r[1]["target"]
    assert res.figure is not None


def test_analyst_trend(clean_sales_df: pd.DataFrame):
    res = ask_analyst("What is the revenue trend over time?", clean_sales_df)
    assert res.is_success is True
    assert res.operation == "trend"
    assert "date" in res.columns_used
    assert "revenue" in res.columns_used
    assert res.computed_data["total"] > 0
    assert res.figure is not None


def test_analyst_correlation(clean_sales_df: pd.DataFrame):
    res = ask_analyst("Is marketing spend correlated with revenue?", clean_sales_df)
    assert res.is_success is True
    assert res.operation == "correlation"
    assert -1.0 <= res.computed_data["pearson_r"] <= 1.0
    assert res.figure is not None


def test_analyst_comparison(clean_sales_df: pd.DataFrame):
    res = ask_analyst("Compare revenue between North and South regions", clean_sales_df)
    assert res.is_success is True
    assert res.operation == "comparison"
    assert "group_1" in res.computed_data
    assert "group_2" in res.computed_data
    assert res.figure is not None


def test_validator_rejects_non_existent_column(clean_sales_df: pd.DataFrame):
    plan = ExecutionPlan(
        operation="aggregate",
        target_column="non_existent_fake_metric",
        agg_func="sum",
    )
    is_valid, msg = validate_plan(plan, clean_sales_df)
    assert is_valid is False
    assert "does not exist in dataset" in msg


def test_validator_rejects_unsupported_operation(clean_sales_df: pd.DataFrame):
    plan = ExecutionPlan(
        operation="unsupported",
        is_supported=False,
        unsupported_reason="Cannot perform arbitrary task.",
    )
    is_valid, msg = validate_plan(plan, clean_sales_df)
    assert is_valid is False


def test_analyst_honestly_rejects_out_of_scope_query(clean_sales_df: pd.DataFrame):
    res = ask_analyst("Write a poem about cats and dogs", clean_sales_df)
    assert res.is_success is False
    assert "cannot be safely mapped" in res.conclusion or "rejected" in res.limitations.lower()
    assert res.figure is None
