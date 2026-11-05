"""Unit tests for the Data Ingestion & Validation Module."""

import io
from pathlib import Path
import pandas as pd
import pytest
from core.ingestion import (
    load_dataset,
    validate_file_metadata,
    DEFAULT_MAX_FILE_SIZE_BYTES,
)


@pytest.fixture
def sample_csv_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "sample" / "sales_data_dirty.csv"
    assert path.exists(), f"Sample dataset missing at {path}"
    return path


def test_load_sample_csv_success(sample_csv_path: Path):
    result = load_dataset(sample_csv_path)
    assert result.is_valid is True
    assert result.dataframe is not None
    assert result.row_count > 1900
    assert result.col_count == 8
    assert len(result.errors) == 0


def test_reject_unsupported_file_extension():
    content = io.BytesIO(b"fake binary content")
    result = load_dataset(content, filename="malicious.exe")
    assert result.is_valid is False
    assert any("Unsupported file type" in err for err in result.errors)


def test_reject_empty_file():
    content = io.BytesIO(b"")
    result = load_dataset(content, filename="empty.csv")
    assert result.is_valid is False
    assert any("empty (0 bytes)" in err for err in result.errors)


def test_reject_file_exceeding_max_size():
    # Test with a low max_size limit (e.g. 50 bytes)
    content = io.BytesIO(b"col1,col2\n" + b"1,2\n" * 50)
    result = load_dataset(content, filename="large.csv", max_size_bytes=50)
    assert result.is_valid is False
    assert any("exceeds maximum allowed limit" in err for err in result.errors)


def test_reject_zero_rows_csv():
    # Only header, no rows
    content = io.BytesIO(b"col1,col2,col3\n")
    result = load_dataset(content, filename="header_only.csv")
    assert result.is_valid is False
    assert any("0 rows" in err for err in result.errors)


def test_latin1_encoding_fallback():
    # Text with latin-1 character not valid in plain ascii
    latin1_bytes = "name,city\nRené,Montréal\n".encode("latin-1")
    content = io.BytesIO(latin1_bytes)
    result = load_dataset(content, filename="latin1_sample.csv")
    assert result.is_valid is True
    assert result.row_count == 1
    assert result.dataframe is not None
    assert "René" in result.dataframe["name"].values

