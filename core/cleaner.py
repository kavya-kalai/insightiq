"""InsightIQ Data Cleaning & Versioning Engine.

Provides non-destructive, previewable cleaning transformations:
1. Deduplication (removing exact or subset duplicate rows).
2. Date standardization (unifying mixed date formats to ISO YYYY-MM-DD).
3. Invalid value remediation (fixing negative prices, non-numeric strings).
4. Missing value imputation (mean, median, mode, constant, or row drop).
5. Outlier handling (IQR clipping or removal).
6. Non-destructive dataset versioning (v1_raw -> v2_cleaned) with before/after diffs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import numpy as np
import pandas as pd
from core.profiler import DataHealthReport, profile_dataset


@dataclass
class CleaningRecipe:
    """A proposed cleaning action that can be previewed and approved."""
    recipe_id: str
    action_type: Literal[
        "drop_duplicates",
        "standardize_dates",
        "fix_numeric_invalid",
        "impute_missing",
        "handle_outliers",
    ]
    column: str | None
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    estimated_impact: str = ""
    is_approved: bool = True


@dataclass
class CleaningDiff:
    """Before vs After difference summary for an executed cleaning action."""
    action_type: str
    column: str | None
    rows_before: int
    rows_after: int
    affected_rows_count: int
    nulls_before: int
    nulls_after: int
    details: str
    sample_before: list[Any] = field(default_factory=list)
    sample_after: list[Any] = field(default_factory=list)


@dataclass
class DatasetVersion:
    """Immutable snapshot of a dataset version."""
    version_id: str  # e.g., 'v1_raw', 'v2_cleaned'
    name: str
    df: pd.DataFrame
    created_at: str
    parent_version_id: str | None
    applied_recipes: list[CleaningRecipe] = field(default_factory=list)
    diffs: list[CleaningDiff] = field(default_factory=list)
    health_score: float | None = None


class DatasetVersionManager:
    """Manages non-destructive dataset lineage and versions."""

    def __init__(self, initial_df: pd.DataFrame, initial_name: str = "Raw Dataset"):
        self.versions: dict[str, DatasetVersion] = {}
        # Deep copy to ensure original cannot be mutated
        initial_copy = initial_df.copy(deep=True)
        report = profile_dataset(initial_copy, initial_name)
        
        root_version = DatasetVersion(
            version_id="v1_raw",
            name=initial_name,
            df=initial_copy,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            parent_version_id=None,
            health_score=report.score.overall_health_score,
        )
        self.versions["v1_raw"] = root_version
        self.active_version_id = "v1_raw"

    def get_active_version(self) -> DatasetVersion:
        """Return the current active dataset version."""
        return self.versions[self.active_version_id]

    def get_version(self, version_id: str) -> DatasetVersion | None:
        """Retrieve a specific dataset version by ID."""
        return self.versions.get(version_id)

    def list_versions(self) -> list[dict[str, Any]]:
        """List summary of all dataset versions."""
        return [
            {
                "version_id": v.version_id,
                "name": v.name,
                "rows": len(v.df),
                "cols": len(v.df.columns),
                "created_at": v.created_at,
                "health_score": v.health_score,
                "parent": v.parent_version_id,
            }
            for v in self.versions.values()
        ]

    def create_new_version(
        self,
        new_df: pd.DataFrame,
        name: str,
        applied_recipes: list[CleaningRecipe],
        diffs: list[CleaningDiff],
    ) -> DatasetVersion:
        """Create and register a new dataset version derived from active version."""
        new_version_num = len(self.versions) + 1
        new_version_id = f"v{new_version_num}_cleaned"
        
        # Deep copy ensures no cross-version reference mutation
        snapshot = new_df.copy(deep=True)
        report = profile_dataset(snapshot, name)

        new_version = DatasetVersion(
            version_id=new_version_id,
            name=name,
            df=snapshot,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            parent_version_id=self.active_version_id,
            applied_recipes=applied_recipes,
            diffs=diffs,
            health_score=report.score.overall_health_score,
        )
        self.versions[new_version_id] = new_version
        self.active_version_id = new_version_id
        return new_version


def suggest_cleaning_recipes(report: DataHealthReport) -> list[CleaningRecipe]:
    """Analyze a DataHealthReport and generate intelligent cleaning recommendations."""
    recipes: list[CleaningRecipe] = []

    # 1. Deduplication suggestion
    if report.total_duplicate_rows > 0:
        recipes.append(
            CleaningRecipe(
                recipe_id="dedup_exact",
                action_type="drop_duplicates",
                column=None,
                description=f"Remove {report.total_duplicate_rows} exact duplicate row(s)",
                parameters={"subset": None, "keep": "first"},
                estimated_impact=f"Will remove {report.total_duplicate_rows} redundant rows.",
                is_approved=True,
            )
        )

    # 2. Column-level suggestions
    for col_name, cp in report.columns.items():
        # Date standardization
        if cp.semantic_type == "datetime":
            recipes.append(
                CleaningRecipe(
                    recipe_id=f"standardize_date_{col_name}",
                    action_type="standardize_dates",
                    column=col_name,
                    description=f"Standardize '{col_name}' to ISO YYYY-MM-DD (drop unparseable)",
                    parameters={"errors": "coerce", "drop_invalid_dates": True},
                    estimated_impact=f"Standardizes mixed formats. Removes {cp.invalid_count} corrupted date rows.",
                    is_approved=True,
                )
            )

        # Numeric invalid values (negative prices, string tokens)
        if cp.semantic_type == "numeric":
            if cp.invalid_count > 0:
                recipes.append(
                    CleaningRecipe(
                        recipe_id=f"fix_invalid_{col_name}",
                        action_type="fix_numeric_invalid",
                        column=col_name,
                        description=f"Sanitize '{col_name}': coerce non-numeric tokens and rectify negative values",
                        parameters={"rectify_negative": "abs", "coerce_strings": True},
                        estimated_impact=f"Fixes {cp.invalid_count} invalid numeric values by taking absolute values and coercing tokens.",
                        is_approved=True,
                    )
                )

            # Missing value imputation for numerics
            if cp.null_count > 0:
                # Use median for skewed distributions (iqr > 0), mean otherwise
                impute_strategy = "median" if cp.outlier_count > 0 else "mean"
                recipes.append(
                    CleaningRecipe(
                        recipe_id=f"impute_num_{col_name}",
                        action_type="impute_missing",
                        column=col_name,
                        description=f"Impute {cp.null_count} missing values in '{col_name}' using {impute_strategy}",
                        parameters={"strategy": impute_strategy},
                        estimated_impact=f"Replaces {cp.null_count} NaNs with column {impute_strategy}.",
                        is_approved=True,
                    )
                )

            # Outlier handling (optional / default to IQR capping)
            if cp.outlier_count > 0:
                recipes.append(
                    CleaningRecipe(
                        recipe_id=f"cap_outliers_{col_name}",
                        action_type="handle_outliers",
                        column=col_name,
                        description=f"Cap extreme {cp.outlier_count} outliers in '{col_name}' at IQR boundaries [{cp.lower_bound}, {cp.upper_bound}]",
                        parameters={"strategy": "clip", "lower": cp.lower_bound, "upper": cp.upper_bound},
                        estimated_impact=f"Caps {cp.outlier_count} extreme values to prevent distribution distortion.",
                        is_approved=False,  # Unchecked by default so user controls whether to cap
                    )
                )

        # Categorical / Identifier missing values
        elif cp.semantic_type in {"categorical", "identifier/text"} and cp.null_count > 0:
            recipes.append(
                CleaningRecipe(
                    recipe_id=f"impute_cat_{col_name}",
                    action_type="impute_missing",
                    column=col_name,
                    description=f"Impute {cp.null_count} missing values in '{col_name}' with 'Unknown' / Mode",
                    parameters={"strategy": "constant", "fill_value": "Unknown"},
                    estimated_impact=f"Replaces {cp.null_count} NaNs with 'Unknown'.",
                    is_approved=True,
                )
            )

    return recipes


def execute_recipe(df: pd.DataFrame, recipe: CleaningRecipe) -> tuple[pd.DataFrame, CleaningDiff]:
    """Execute a single approved cleaning recipe non-destructively on a DataFrame copy."""
    # Always work on a fresh copy
    working_df = df.copy(deep=True)
    rows_before = len(working_df)
    col = recipe.column

    nulls_before = int(working_df[col].isna().sum()) if col and col in working_df else int(working_df.isna().sum().sum())
    action = recipe.action_type
    params = recipe.parameters

    diff_details = ""
    sample_before: list[Any] = []
    sample_after: list[Any] = []
    affected_count = 0

    if action == "drop_duplicates":
        subset = params.get("subset")
        keep = params.get("keep", "first")
        dup_mask = working_df.duplicated(subset=subset, keep=keep)
        affected_count = int(dup_mask.sum())
        working_df = working_df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        diff_details = f"Removed {affected_count} duplicate rows."

    elif action == "standardize_dates" and col:
        initial_series = working_df[col].copy()
        parsed_dates = pd.to_datetime(working_df[col], errors="coerce", format="mixed")
        
        # Capture samples of what failed
        unparseable_mask = initial_series.notna() & parsed_dates.isna()
        sample_before = initial_series[unparseable_mask].head(3).tolist()
        
        if params.get("drop_invalid_dates", True):
            # Drop rows with corrupted dates or missing dates
            drop_mask = unparseable_mask | working_df[col].isna()
            affected_count = int(drop_mask.sum())
            working_df = working_df[~drop_mask].reset_index(drop=True)
            # Re-parse valid rows and format as ISO YYYY-MM-DD string
            valid_dates = pd.to_datetime(working_df[col], errors="coerce", format="mixed")
            working_df[col] = valid_dates.dt.strftime("%Y-%m-%d")
            sample_after = working_df[col].head(3).tolist()
            diff_details = f"Formatted dates to ISO YYYY-MM-DD. Dropped {affected_count} invalid/missing date rows."
        else:
            working_df[col] = parsed_dates.dt.strftime("%Y-%m-%d")
            affected_count = int(unparseable_mask.sum())
            diff_details = f"Formatted dates to ISO YYYY-MM-DD (unparseable set to NaT)."

    elif action == "fix_numeric_invalid" and col:
        initial_vals = working_df[col].copy()
        num_converted = pd.to_numeric(working_df[col], errors="coerce")
        
        # 1. Coerce bad string tokens
        bad_token_mask = initial_vals.notna() & num_converted.isna()
        sample_before = initial_vals[bad_token_mask].head(3).tolist()
        
        # 2. Rectify negative numbers
        neg_mask = num_converted < 0
        neg_count = int(neg_mask.sum())
        
        if params.get("rectify_negative") == "abs":
            num_converted = num_converted.abs()
        
        working_df[col] = num_converted
        affected_count = int(bad_token_mask.sum()) + neg_count
        sample_after = working_df[col].dropna().head(3).tolist()
        diff_details = f"Coerced {int(bad_token_mask.sum())} non-numeric token(s) and rectified {neg_count} negative value(s)."

    elif action == "impute_missing" and col:
        missing_mask = working_df[col].isna()
        affected_count = int(missing_mask.sum())
        strategy = params.get("strategy", "median")
        
        if affected_count > 0:
            if strategy == "median":
                fill_val = round(float(pd.to_numeric(working_df[col], errors="coerce").median()), 2)
                working_df[col] = working_df[col].fillna(fill_val)
                diff_details = f"Imputed {affected_count} missing values with median ({fill_val})."
            elif strategy == "mean":
                fill_val = round(float(pd.to_numeric(working_df[col], errors="coerce").mean()), 2)
                working_df[col] = working_df[col].fillna(fill_val)
                diff_details = f"Imputed {affected_count} missing values with mean ({fill_val})."
            elif strategy == "mode":
                mode_val = working_df[col].mode().iloc[0] if not working_df[col].mode().empty else "Unknown"
                working_df[col] = working_df[col].fillna(mode_val)
                diff_details = f"Imputed {affected_count} missing values with mode ('{mode_val}')."
            elif strategy == "constant":
                const_val = params.get("fill_value", "Unknown")
                working_df[col] = working_df[col].fillna(const_val)
                diff_details = f"Imputed {affected_count} missing values with constant ('{const_val}')."
            elif strategy == "drop":
                working_df = working_df.dropna(subset=[col]).reset_index(drop=True)
                diff_details = f"Dropped {affected_count} rows with missing '{col}'."

    elif action == "handle_outliers" and col:
        num_s = pd.to_numeric(working_df[col], errors="coerce")
        lower = params.get("lower")
        upper = params.get("upper")
        strategy = params.get("strategy", "clip")
        
        if lower is not None and upper is not None:
            outlier_mask = (num_s < lower) | (num_s > upper)
            affected_count = int(outlier_mask.sum())
            if strategy == "clip" and affected_count > 0:
                working_df[col] = num_s.clip(lower=lower, upper=upper)
                diff_details = f"Clipped {affected_count} outlier values to range [{lower}, {upper}]."
            elif strategy == "drop" and affected_count > 0:
                working_df = working_df[~outlier_mask].reset_index(drop=True)
                diff_details = f"Dropped {affected_count} rows exceeding IQR bounds."

    rows_after = len(working_df)
    nulls_after = int(working_df[col].isna().sum()) if col and col in working_df else int(working_df.isna().sum().sum())

    diff = CleaningDiff(
        action_type=action,
        column=col,
        rows_before=rows_before,
        rows_after=rows_after,
        affected_rows_count=affected_count,
        nulls_before=nulls_before,
        nulls_after=nulls_after,
        details=diff_details,
        sample_before=sample_before,
        sample_after=sample_after,
    )
    return working_df, diff


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    recipes: list[CleaningRecipe],
) -> tuple[pd.DataFrame, list[CleaningDiff]]:
    """Execute all approved recipes sequentially on a clean copy of the dataset."""
    working_df = df.copy(deep=True)
    diffs: list[CleaningDiff] = []

    for recipe in recipes:
        if not recipe.is_approved:
            continue
        working_df, diff = execute_recipe(working_df, recipe)
        diffs.append(diff)

    return working_df, diffs
