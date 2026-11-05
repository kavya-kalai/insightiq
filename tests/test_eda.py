"""Unit tests for the Automated EDA Engine and Chart Recommendation."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import pytest
from core.cleaner import DatasetVersionManager, apply_cleaning_pipeline, suggest_cleaning_recipes
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
from core.profiler import profile_dataset


@pytest.fixture
def cleaned_sales_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "sample" / "sales_data_dirty.csv"
    res = load_dataset(sample_path)
    assert res.dataframe is not None
    rep = profile_dataset(res.dataframe)
    recipes = suggest_cleaning_recipes(rep)
    cleaned, _ = apply_cleaning_pipeline(res.dataframe, recipes)
    return cleaned


def test_compute_summary_statistics(cleaned_sales_df: pd.DataFrame):
    stats = compute_summary_statistics(cleaned_sales_df)
    assert "numeric" in stats
    assert "categorical" in stats
    assert "revenue" in stats["numeric"]
    assert stats["numeric"]["revenue"]["count"] > 1800
    assert stats["numeric"]["revenue"]["mean"] > 0
    assert stats["numeric"]["revenue"]["q25"] <= stats["numeric"]["revenue"]["median"]
    assert "product" in stats["categorical"]
    assert stats["categorical"]["product"]["unique"] == 7


def test_correlation_matrix(cleaned_sales_df: pd.DataFrame):
    corr_df, pairs = compute_correlation_matrix(cleaned_sales_df)
    assert not corr_df.empty
    assert "revenue" in corr_df.columns
    assert "price" in corr_df.columns
    assert len(pairs) >= 3
    # Absolute correlation between 0 and 1
    assert 0.0 <= pairs[0]["abs_correlation"] <= 1.0


def test_create_correlation_heatmap(cleaned_sales_df: pd.DataFrame):
    fig = create_correlation_heatmap(cleaned_sales_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_distribution_chart(cleaned_sales_df: pd.DataFrame):
    fig = create_distribution_chart(cleaned_sales_df, "revenue")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_create_time_series_trend(cleaned_sales_df: pd.DataFrame):
    fig = create_time_series_trend(cleaned_sales_df, "date", "revenue")
    assert isinstance(fig, go.Figure)
    # Contains daily line + 7-day rolling trend trace
    assert len(fig.data) == 2


def test_create_top_categories_chart(cleaned_sales_df: pd.DataFrame):
    fig = create_top_categories_chart(cleaned_sales_df, "product", "revenue", top_n=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_automatic_chart_selection(cleaned_sales_df: pd.DataFrame):
    # 1. Single numeric -> Distribution
    name1, fig1 = recommend_chart(cleaned_sales_df, "price")
    assert "Distribution" in name1
    assert isinstance(fig1, go.Figure)

    # 2. Single categorical -> Categorical Frequency
    name2, fig2 = recommend_chart(cleaned_sales_df, "region")
    assert "Categorical" in name2
    assert isinstance(fig2, go.Figure)

    # 3. Datetime + Numeric -> Time Series
    name3, fig3 = recommend_chart(cleaned_sales_df, "date", "revenue")
    assert "Time-Series" in name3
    assert isinstance(fig3, go.Figure)

    # 4. Numeric + Numeric -> Scatter
    name4, fig4 = recommend_chart(cleaned_sales_df, "marketing_spend", "revenue")
    assert "Scatter" in name4
    assert isinstance(fig4, go.Figure)

    # 5. Categorical + Numeric -> Aggregation
    name5, fig5 = recommend_chart(cleaned_sales_df, "product", "revenue")
    assert "Aggregation" in name5
    assert isinstance(fig5, go.Figure)
