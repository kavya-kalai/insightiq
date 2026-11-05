"""Unit tests for Data Cleaning and Dataset Versioning."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.cleaner import (
    CleaningRecipe,
    DatasetVersionManager,
    apply_cleaning_pipeline,
    execute_recipe,
    suggest_cleaning_recipes,
)
from core.ingestion import load_dataset
from core.profiler import profile_dataset


@pytest.fixture
def dirty_sample_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "sample" / "sales_data_dirty.csv"
    res = load_dataset(sample_path)
    assert res.dataframe is not None
    return res.dataframe


def test_deduplication():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": ["a", "b", "b", "c"],
    })
    recipe = CleaningRecipe(
        recipe_id="dedup",
        action_type="drop_duplicates",
        column=None,
        description="Drop duplicates",
    )
    cleaned, diff = execute_recipe(df, recipe)
    assert len(cleaned) == 3
    assert diff.affected_rows_count == 1
    assert len(df) == 4  # Non-destructive: original untouched


def test_date_standardization():
    df = pd.DataFrame({
        "txn_date": ["2024-01-15", "16/01/2024", "01-17-2024", "corrupted_date"]
    })
    recipe = CleaningRecipe(
        recipe_id="std_date",
        action_type="standardize_dates",
        column="txn_date",
        description="Standardize date",
        parameters={"drop_invalid_dates": True},
    )
    cleaned, diff = execute_recipe(df, recipe)
    assert len(cleaned) == 3
    assert diff.affected_rows_count == 1
    # Check that dates follow ISO YYYY-MM-DD
    assert all("-" in str(d) and len(str(d)) == 10 for d in cleaned["txn_date"])


def test_fix_numeric_invalid():
    df = pd.DataFrame({
        "price": [10.0, -25.5, "INVALID", 40.0]
    })
    recipe = CleaningRecipe(
        recipe_id="fix_num",
        action_type="fix_numeric_invalid",
        column="price",
        description="Fix invalid price",
        parameters={"rectify_negative": "abs"},
    )
    cleaned, diff = execute_recipe(df, recipe)
    assert cleaned["price"].iloc[1] == 25.5  # Rectified negative
    assert np.isnan(cleaned["price"].iloc[2])  # Coerced string to NaN
    assert diff.affected_rows_count == 2


def test_imputation_median():
    df = pd.DataFrame({
        "score": [10.0, 20.0, np.nan, 30.0, 100.0]
    })
    recipe = CleaningRecipe(
        recipe_id="impute_score",
        action_type="impute_missing",
        column="score",
        description="Impute score",
        parameters={"strategy": "median"},
    )
    cleaned, diff = execute_recipe(df, recipe)
    assert cleaned["score"].isna().sum() == 0
    # Median of [10, 20, 30, 100] is 25.0
    assert cleaned["score"].iloc[2] == 25.0


def test_dataset_versioning_and_health_improvement(dirty_sample_df: pd.DataFrame):
    # 1. Initialize Version Manager
    manager = DatasetVersionManager(dirty_sample_df, "Dirty Sales v1")
    v1 = manager.get_active_version()
    assert v1.version_id == "v1_raw"
    initial_score = v1.health_score
    assert initial_score is not None

    # 2. Get suggested recipes
    report_v1 = profile_dataset(v1.df, "Dirty Sales v1")
    recipes = suggest_cleaning_recipes(report_v1)
    assert len(recipes) >= 4

    # 3. Apply cleaning pipeline
    cleaned_df, diffs = apply_cleaning_pipeline(v1.df, recipes)
    assert len(cleaned_df) < len(v1.df)  # Duplicates and corrupted dates dropped
    assert cleaned_df["price"].isna().sum() == 0  # Imputed after rectification

    # 4. Create new version
    v2 = manager.create_new_version(cleaned_df, "Cleaned Sales v2", recipes, diffs)
    assert v2.version_id == "v2_cleaned"
    assert v2.parent_version_id == "v1_raw"
    assert v2.health_score is not None
    # Health score must improve significantly
    assert v2.health_score > initial_score
    assert v2.health_score >= 92.5
    assert v2.df.isna().sum().sum() == 0
    assert v2.df.duplicated().sum() == 0

    # 5. Verify original raw dataset remained unchanged
    assert len(v1.df) == len(dirty_sample_df)
    assert v1.df.duplicated().sum() == dirty_sample_df.duplicated().sum()
