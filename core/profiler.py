"""InsightIQ Data Profiling & Health Scoring Engine.

Analyzes raw datasets across 4 core data-quality dimensions:
1. Completeness (missing values rate)
2. Uniqueness (duplicate rows rate)
3. Validity (format violations, negative prices, unparseable dates)
4. Consistency & Outliers (Interquartile Range - IQR detection)

Computes a deterministic, transparent 0-100 Data Health Score with full
audit breakdown and generates an executive Data Health Report.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Detailed profile of a single dataset column."""
    name: str
    physical_dtype: str
    semantic_type: str  # 'numeric', 'categorical', 'datetime', 'identifier/text'
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: list[Any]
    # Numeric / Outlier specific
    mean: float | None = None
    std: float | None = None
    min_value: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max_value: float | None = None
    iqr: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    # Validity specific
    invalid_count: int = 0
    invalid_samples: list[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Transparent mathematical breakdown of the 0-100 Health Score."""
    completeness_score: float  # Weight 30%
    uniqueness_score: float    # Weight 25%
    validity_score: float      # Weight 25%
    outlier_score: float       # Weight 20%
    overall_health_score: float
    grade: str                 # A, B, C, D, F
    deductions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DataHealthReport:
    """Comprehensive dataset health audit report."""
    dataset_name: str
    total_rows: int
    total_columns: int
    total_cells: int
    total_missing_cells: int
    total_duplicate_rows: int
    columns: dict[str, ColumnProfile]
    score: ScoreBreakdown
    top_issues: list[str] = field(default_factory=list)

    def to_markdown_report(self) -> str:
        """Render a clean, human-readable markdown report for the user or UI."""
        sb = self.score
        lines = [
            f"# 📋 InsightIQ Data Health Report: {self.dataset_name}",
            "",
            "## 1. Overall Health Score",
            f"**Health Score:** `{sb.overall_health_score:.1f} / 100` (Grade: **{sb.grade}**)",
            "",
            "| Dimension | Weight | Dimension Score | Weighted Contribution |",
            "|---|:---:|:---:|:---:|",
            f"| **Completeness** | 30% | {sb.completeness_score:.1f}% | {sb.completeness_score * 0.30:.1f} pts |",
            f"| **Uniqueness** | 25% | {sb.uniqueness_score:.1f}% | {sb.uniqueness_score * 0.25:.1f} pts |",
            f"| **Validity** | 25% | {sb.validity_score:.1f}% | {sb.validity_score * 0.25:.1f} pts |",
            f"| **Statistical Uniformity (IQR)** | 20% | {sb.outlier_score:.1f}% | {sb.outlier_score * 0.20:.1f} pts |",
            f"| **TOTAL** | **100%** | — | **{sb.overall_health_score:.1f} / 100** |",
            "",
            "## 2. Dataset Overview",
            f"- **Rows:** {self.total_rows:,} | **Columns:** {self.total_columns}",
            f"- **Total Missing Cells:** {self.total_missing_cells:,} ({(self.total_missing_cells / max(1, self.total_cells) * 100):.2f}% of all cells)",
            f"- **Exact Duplicate Rows:** {self.total_duplicate_rows:,} ({(self.total_duplicate_rows / max(1, self.total_rows) * 100):.2f}% of rows)",
            "",
            "## 3. Key Issues Identified",
        ]
        if self.top_issues:
            for issue in self.top_issues:
                lines.append(f"- ⚠️ {issue}")
        else:
            lines.append("- ✅ No major data health anomalies detected.")

        lines.extend([
            "",
            "## 4. Column Diagnostics",
            "| Column | Inferred Type | Nulls (%) | Unique | Outliers (IQR) | Validity Warnings |",
            "|---|---|:---:|:---:|:---:|---|",
        ])
        for col_name, cp in self.columns.items():
            warn_str = f"{cp.invalid_count} invalid" if cp.invalid_count > 0 else "None"
            outlier_str = f"{cp.outlier_count} ({cp.outlier_percentage:.1f}%)" if cp.semantic_type == "numeric" else "N/A"
            lines.append(
                f"| `{col_name}` | {cp.semantic_type} | {cp.null_count} ({cp.null_percentage:.1f}%) | "
                f"{cp.unique_count} | {outlier_str} | {warn_str} |"
            )

        return "\n".join(lines)


def infer_semantic_type(series: pd.Series) -> str:
    """Infer the underlying semantic data type beyond the raw pandas dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # For object/string columns, attempt safe parsing to detect hidden numbers or dates
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return "categorical"

    # Test if candidate numeric
    numeric_converted = pd.to_numeric(non_null, errors="coerce")
    valid_numeric_ratio = numeric_converted.notna().mean()
    if valid_numeric_ratio >= 0.70:
        return "numeric"

    # Test if candidate datetime
    # Sample up to 100 non-null values for speed
    sample = non_null.sample(min(len(non_null), 100), random_state=42)
    parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed_dates.notna().mean() >= 0.70:
        return "datetime"

    # Distinguish low-cardinality categorical from high-cardinality text / ID
    unique_ratio = series.nunique() / max(1, len(series))
    if unique_ratio < 0.20 or series.nunique() <= 30:
        return "categorical"

    return "identifier/text"


def detect_iqr_outliers(series: pd.Series) -> tuple[float, float, float, float, float, int, float, list[float]]:
    """Calculate IQR bounds and identify statistical outlier count and samples.
    
    Returns:
        (Q1, median, Q3, lower_bound, upper_bound, outlier_count, outlier_pct, sample_outliers)
    """
    clean_num = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_num) < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, []

    q1 = float(np.percentile(clean_num, 25))
    median = float(np.percentile(clean_num, 50))
    q3 = float(np.percentile(clean_num, 75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (clean_num < lower_bound) | (clean_num > upper_bound)
    outlier_values = clean_num[outlier_mask].tolist()
    outlier_count = len(outlier_values)
    outlier_pct = round((outlier_count / len(clean_num)) * 100, 2)
    sample_outliers = [round(float(v), 2) for v in outlier_values[:5]]

    return (
        round(q1, 2),
        round(median, 2),
        round(q3, 2),
        round(lower_bound, 2),
        round(upper_bound, 2),
        outlier_count,
        outlier_pct,
        sample_outliers,
    )


def calculate_health_score(
    total_cells: int,
    total_missing_cells: int,
    total_rows: int,
    duplicate_rows: int,
    total_invalid_values: int,
    total_numeric_values: int,
    total_outliers: int,
) -> ScoreBreakdown:
    """Calculate transparent 0-100 data health score and dimension points."""
    deductions: list[dict[str, Any]] = []

    # 1. Completeness Score (Weight: 30%)
    missing_ratio = total_missing_cells / max(1, total_cells)
    completeness_score = max(0.0, min(100.0, (1.0 - missing_ratio) * 100.0))
    if missing_ratio > 0:
        deductions.append({
            "dimension": "Completeness",
            "penalty": round((100.0 - completeness_score) * 0.30, 2),
            "reason": f"{total_missing_cells:,} missing cells ({missing_ratio * 100:.1f}% missing rate)",
        })

    # 2. Uniqueness Score (Weight: 25%)
    duplicate_ratio = duplicate_rows / max(1, total_rows)
    uniqueness_score = max(0.0, min(100.0, (1.0 - duplicate_ratio) * 100.0))
    if duplicate_rows > 0:
        deductions.append({
            "dimension": "Uniqueness",
            "penalty": round((100.0 - uniqueness_score) * 0.25, 2),
            "reason": f"{duplicate_rows:,} duplicate rows ({duplicate_ratio * 100:.1f}% duplicate rate)",
        })

    # 3. Validity Score (Weight: 25%)
    # Ratio of invalid values against total rows (capped at 100% penalty)
    invalid_ratio = min(1.0, total_invalid_values / max(1, total_rows))
    validity_score = max(0.0, min(100.0, (1.0 - invalid_ratio) * 100.0))
    if total_invalid_values > 0:
        deductions.append({
            "dimension": "Validity",
            "penalty": round((100.0 - validity_score) * 0.25, 2),
            "reason": f"{total_invalid_values:,} format violations (invalid dates, negative prices, corrupted tokens)",
        })

    # 4. Outlier / Statistical Consistency Score (Weight: 20%)
    outlier_ratio = total_outliers / max(1, total_numeric_values)
    # A standard normal distribution has ~0.7% IQR outliers naturally; only penalize excess outliers (> 2%)
    excess_outlier_ratio = max(0.0, outlier_ratio - 0.02)
    outlier_score = max(0.0, min(100.0, (1.0 - (excess_outlier_ratio * 5.0)) * 100.0))
    if total_outliers > 0:
        deductions.append({
            "dimension": "Statistical Uniformity",
            "penalty": round((100.0 - outlier_score) * 0.20, 2),
            "reason": f"{total_outliers:,} extreme outliers detected via IQR ({outlier_ratio * 100:.1f}%)",
        })

    # Weighted Overall Score
    overall = (
        (completeness_score * 0.30)
        + (uniqueness_score * 0.25)
        + (validity_score * 0.25)
        + (outlier_score * 0.20)
    )
    overall = round(max(0.0, min(100.0, overall)), 1)

    # Assign Grade
    if overall >= 90.0:
        grade = "A (Excellent)"
    elif overall >= 80.0:
        grade = "B (Good)"
    elif overall >= 70.0:
        grade = "C (Fair - Attention Needed)"
    elif overall >= 60.0:
        grade = "D (Poor - Cleaning Required)"
    else:
        grade = "F (Critical - Unsafe for Analytics)"

    return ScoreBreakdown(
        completeness_score=round(completeness_score, 1),
        uniqueness_score=round(uniqueness_score, 1),
        validity_score=round(validity_score, 1),
        outlier_score=round(outlier_score, 1),
        overall_health_score=overall,
        grade=grade,
        deductions=deductions,
    )


def profile_dataset(df: pd.DataFrame, dataset_name: str = "Dataset") -> DataHealthReport:
    """Perform comprehensive data profiling and compute the Data Health Report."""
    total_rows = len(df)
    total_columns = len(df.columns)
    total_cells = total_rows * total_columns
    total_missing_cells = int(df.isna().sum().sum())
    total_duplicate_rows = int(df.duplicated().sum())

    columns_profile: dict[str, ColumnProfile] = {}
    top_issues: list[str] = []

    if total_duplicate_rows > 0:
        dup_pct = (total_duplicate_rows / total_rows) * 100
        top_issues.append(f"Found {total_duplicate_rows} duplicate rows ({dup_pct:.1f}% of dataset).")

    total_invalid_values = 0
    total_numeric_values = 0
    total_outliers = 0

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2)
        unique_count = int(series.nunique())
        unique_pct = round((unique_count / total_rows) * 100, 2)
        sample_vals = series.dropna().head(4).tolist()

        sem_type = infer_semantic_type(series)

        # Profile per semantic type
        cp = ColumnProfile(
            name=col,
            physical_dtype=str(series.dtype),
            semantic_type=sem_type,
            total_count=total_rows,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            sample_values=sample_vals,
        )

        if null_pct > 5.0:
            top_issues.append(f"Column '{col}' has high missingness ({null_pct:.1f}% missing).")

        # Numeric Diagnostics & Outliers
        if sem_type == "numeric":
            num_series = pd.to_numeric(series, errors="coerce")
            valid_num = num_series.dropna()
            total_numeric_values += len(valid_num)

            # Check invalid strings in numeric column
            invalid_strings_mask = series.notna() & num_series.isna()
            invalid_str_count = int(invalid_strings_mask.sum())
            if invalid_str_count > 0:
                cp.invalid_count += invalid_str_count
                cp.invalid_samples.extend(series[invalid_strings_mask].astype(str).unique().tolist()[:3])
                total_invalid_values += invalid_str_count
                top_issues.append(
                    f"Column '{col}' contains {invalid_str_count} non-numeric token(s) (e.g. {cp.invalid_samples})."
                )

            # Check domain violations (e.g. negative prices or negative quantities)
            if col.lower() in {"price", "quantity", "revenue", "marketing_spend"}:
                neg_mask = valid_num < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    cp.invalid_count += neg_count
                    total_invalid_values += neg_count
                    top_issues.append(f"Column '{col}' has {neg_count} negative value(s), violating domain rules.")

            if len(valid_num) > 0:
                cp.mean = round(float(valid_num.mean()), 2)
                cp.std = round(float(valid_num.std()), 2) if len(valid_num) > 1 else 0.0
                cp.min_value = round(float(valid_num.min()), 2)
                cp.max_value = round(float(valid_num.max()), 2)

                (
                    q1, median, q3, lower_b, upper_b,
                    outlier_cnt, outlier_pct, samples
                ) = detect_iqr_outliers(valid_num)

                cp.q1 = q1
                cp.median = median
                cp.q3 = q3
                cp.iqr = round(q3 - q1, 2)
                cp.lower_bound = lower_b
                cp.upper_bound = upper_b
                cp.outlier_count = outlier_cnt
                cp.outlier_percentage = outlier_pct
                total_outliers += outlier_cnt

                if outlier_pct > 1.0:
                    top_issues.append(
                        f"Column '{col}' has {outlier_cnt} statistical outliers via IQR (>{upper_b} or <{lower_b})."
                    )

        # Datetime Diagnostics
        elif sem_type == "datetime":
            # Test parsing consistency
            parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
            unparseable_mask = series.notna() & parsed_dates.isna()
            unparseable_count = int(unparseable_mask.sum())
            if unparseable_count > 0:
                cp.invalid_count += unparseable_count
                cp.invalid_samples.extend(series[unparseable_mask].astype(str).unique().tolist()[:3])
                total_invalid_values += unparseable_count
                top_issues.append(
                    f"Column '{col}' contains {unparseable_count} unparseable date values (e.g. {cp.invalid_samples})."
                )

        columns_profile[col] = cp

    # Calculate overall health score
    score_breakdown = calculate_health_score(
        total_cells=total_cells,
        total_missing_cells=total_missing_cells,
        total_rows=total_rows,
        duplicate_rows=total_duplicate_rows,
        total_invalid_values=total_invalid_values,
        total_numeric_values=total_numeric_values,
        total_outliers=total_outliers,
    )

    return DataHealthReport(
        dataset_name=dataset_name,
        total_rows=total_rows,
        total_columns=total_columns,
        total_cells=total_cells,
        total_missing_cells=total_missing_cells,
        total_duplicate_rows=total_duplicate_rows,
        columns=columns_profile,
        score=score_breakdown,
        top_issues=top_issues,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.ingestion import load_dataset

    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "sales_data_dirty.csv"
    ingest_res = load_dataset(sample_path)
    if ingest_res.dataframe is not None:
        report = profile_dataset(ingest_res.dataframe, dataset_name="Dirty Sales Sample")
        print(report.to_markdown_report())
