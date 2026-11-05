"""Unit tests for the Data Profiling and Health Scoring Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.ingestion import load_dataset
from core.profiler import (
    calculate_health_score,
    detect_iqr_outliers,
    infer_semantic_type,
    profile_dataset,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Fixture with a pristine synthetic dataset."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "department": ["Sales", "HR", "Sales", "IT", "HR", "IT", "Sales", "Finance"],
        "joined_date": pd.date_range("2023-01-01", periods=8, freq="ME").astype(str),
    })


def test_perfect_health_score(clean_df: pd.DataFrame):
    report = profile_dataset(clean_df, dataset_name="Clean Data")
    sb = report.score
    assert sb.completeness_score == 100.0
    assert sb.uniqueness_score == 100.0
    assert sb.validity_score == 100.0
    assert sb.overall_health_score >= 98.0
    assert "A" in sb.grade


def test_semantic_type_inference():
    # Numeric stored as strings due to dirty entry
    dirty_num = pd.Series(["10.5", "20.1", "30.0", "INVALID", "40.5", "50.0"])
    assert infer_semantic_type(dirty_num) == "numeric"

    # Datetime stored as strings
    dirty_dates = pd.Series(["2024-01-01", "15/02/2024", "03-10-2024", "Mar 12, 2024"])
    assert infer_semantic_type(dirty_dates) == "datetime"

    # Categorical
    categories = pd.Series(["North", "South", "North", "East", "West", "North"])
    assert infer_semantic_type(categories) == "categorical"


def test_iqr_outlier_detection():
    # Normal range 10-20, plus an extreme outlier of 1000
    data = pd.Series([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1000])
    q1, median, q3, lower_b, upper_b, out_cnt, out_pct, samples = detect_iqr_outliers(data)

    assert out_cnt == 1
    assert 1000.0 in samples
    assert upper_b < 1000.0


def test_score_breakdown_math():
    sb = calculate_health_score(
        total_cells=1000,
        total_missing_cells=100,  # 10% missing -> Completeness 90%
        total_rows=100,
        duplicate_rows=10,        # 10% dup -> Uniqueness 90%
        total_invalid_values=5,   # 5% invalid -> Validity 95%
        total_numeric_values=400,
        total_outliers=8,         # 2% outliers -> Outlier ~100%
    )
    assert sb.completeness_score == 90.0
    assert sb.uniqueness_score == 90.0
    assert sb.validity_score == 95.0
    # Expected overall: 90*0.3 + 90*0.25 + 95*0.25 + 100*0.20 = 27 + 22.5 + 23.75 + 20 = 93.25 -> 93.2 or 93.3
    expected_weighted = (90 * 0.30) + (90 * 0.25) + (95 * 0.25) + (sb.outlier_score * 0.20)
    assert abs(sb.overall_health_score - round(expected_weighted, 1)) < 0.2


def test_profile_sample_dirty_dataset():
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "sample" / "sales_data_dirty.csv"
    ingest_res = load_dataset(sample_path)
    assert ingest_res.is_valid is True
    assert ingest_res.dataframe is not None

    report = profile_dataset(ingest_res.dataframe, "Dirty Sales")
    # Verify planted anomalies were captured
    assert report.total_duplicate_rows == 30
    assert report.total_missing_cells > 200
    assert report.columns["date"].invalid_count >= 20
    assert report.columns["price"].invalid_count >= 15  # Negative prices & strings
    assert report.columns["revenue"].outlier_count > 0
    assert len(report.top_issues) > 3
    assert len(report.to_markdown_report()) > 500
