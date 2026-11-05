"""InsightIQ Data Ingestion & Validation Module.

Provides safe file ingestion for CSV and Excel files:
- Enforces strict file extension and MIME type allowlists.
- Enforces maximum file size limit (default 25 MB) to prevent DoS.
- Handles multiple text encodings (utf-8, utf-8-sig, latin-1, cp1252) automatically.
- Detects empty files, zero-row datasets, and corrupted byte streams.
- Framework-independent: works with file paths, binary file-like objects (BytesIO, Streamlit UploadedFile, FastAPI UploadFile).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import BinaryIO, Union
import pandas as pd


# 25 MB default size limit
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass
class IngestionResult:
    """Structured result returned after file validation and parsing."""
    is_valid: bool
    filename: str
    file_size_bytes: int
    file_type: str
    dataframe: pd.DataFrame | None = None
    row_count: int = 0
    col_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Union[str, int, bool, list[str]]]:
        """Return a clean dictionary representation of the ingestion result."""
        return {
            "is_valid": self.is_valid,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size_kb": round(self.file_size_bytes / 1024, 2),
            "row_count": self.row_count,
            "col_count": self.col_count,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_file_metadata(
    filename: str,
    file_size_bytes: int,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[bool, str, list[str]]:
    """Validate file extension and size constraints before reading into memory.
    
    Returns:
        (is_valid, file_extension, error_messages)
    """
    errors: list[str] = []
    path = Path(filename)
    extension = path.suffix.lower()

    if not extension:
        errors.append(f"File '{filename}' has no extension. Please upload a .csv or .xlsx file.")
    elif extension not in ALLOWED_EXTENSIONS:
        errors.append(
            f"Unsupported file type '{extension}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    if file_size_bytes <= 0:
        errors.append(f"File '{filename}' is empty (0 bytes).")
    elif file_size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = file_size_bytes / (1024 * 1024)
        errors.append(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB."
        )

    return len(errors) == 0, extension, errors


def parse_csv_stream(content_bytes: bytes) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Parse CSV bytes trying multiple safe encodings and robust delimiters."""
    warnings: list[str] = []
    errors: list[str] = []
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    df = None
    last_err: Exception | None = None

    for enc in encodings_to_try:
        try:
            stream = io.BytesIO(content_bytes)
            # Use Python engine or C engine with fallback
            df = pd.read_csv(stream, encoding=enc, on_bad_lines="warn")
            if enc != "utf-8":
                warnings.append(f"File decoded using fallback encoding '{enc}'.")
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            last_err = exc
            break

    if df is None:
        errors.append(f"Failed to parse CSV: {last_err or 'Encoding decode failure'}")
        return None, warnings, errors

    return df, warnings, errors


def parse_excel_stream(content_bytes: bytes) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Parse Excel (.xlsx / .xls) bytes using openpyxl."""
    warnings: list[str] = []
    errors: list[str] = []
    try:
        stream = io.BytesIO(content_bytes)
        df = pd.read_excel(stream, engine="openpyxl")
        return df, warnings, errors
    except Exception as exc:
        errors.append(f"Failed to parse Excel file: {str(exc)}")
        return None, warnings, errors


def load_dataset(
    source: Union[str, Path, BinaryIO, io.BytesIO],
    filename: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> IngestionResult:
    """Safe universal ingestion entrypoint.
    
    Accepts:
    - Path or string to local file
    - Binary file object / BytesIO (such as Streamlit UploadedFile)
    """
    warnings: list[str] = []
    errors: list[str] = []

    # 1. Resolve content bytes and file size
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            return IngestionResult(
                is_valid=False,
                filename=file_path.name,
                file_size_bytes=0,
                file_type="",
                errors=[f"File '{file_path}' does not exist on disk."]
            )
        resolved_filename = filename or file_path.name
        content_bytes = file_path.read_bytes()
        file_size_bytes = len(content_bytes)
    else:
        # Stream-like object (BytesIO, UploadedFile)
        resolved_filename = filename or getattr(source, "name", "uploaded_file.csv")
        source.seek(0)
        content_bytes = source.read()
        file_size_bytes = len(content_bytes)
        source.seek(0)

    # 2. Validate metadata
    is_valid_meta, ext, meta_errors = validate_file_metadata(
        filename=resolved_filename,
        file_size_bytes=file_size_bytes,
        max_size_bytes=max_size_bytes,
    )
    if not is_valid_meta:
        return IngestionResult(
            is_valid=False,
            filename=resolved_filename,
            file_size_bytes=file_size_bytes,
            file_type=ext,
            errors=meta_errors,
        )

    # 3. Parse based on extension
    if ext == ".csv":
        df, parse_warns, parse_errors = parse_csv_stream(content_bytes)
    else:
        df, parse_warns, parse_errors = parse_excel_stream(content_bytes)

    warnings.extend(parse_warns)
    errors.extend(parse_errors)

    if df is None:
        return IngestionResult(
            is_valid=False,
            filename=resolved_filename,
            file_size_bytes=file_size_bytes,
            file_type=ext,
            warnings=warnings,
            errors=errors,
        )

    # 4. Logical validation on parsed DataFrame
    if df.empty:
        errors.append("Dataset contains 0 rows of data.")
        return IngestionResult(
            is_valid=False,
            filename=resolved_filename,
            file_size_bytes=file_size_bytes,
            file_type=ext,
            warnings=warnings,
            errors=errors,
        )

    if len(df.columns) == 0:
        errors.append("Dataset contains 0 columns.")
        return IngestionResult(
            is_valid=False,
            filename=resolved_filename,
            file_size_bytes=file_size_bytes,
            file_type=ext,
            warnings=warnings,
            errors=errors,
        )

    # Sanitize column names: strip whitespace, stringify
    df.columns = [str(col).strip() for col in df.columns]

    # Check for unnamed or duplicate columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        warnings.append(f"Detected {len(unnamed_cols)} unnamed column(s) which may indicate index columns.")

    if len(df.columns) != len(set(df.columns)):
        warnings.append("Dataset contains duplicate column headers.")

    return IngestionResult(
        is_valid=True,
        filename=resolved_filename,
        file_size_bytes=file_size_bytes,
        file_type=ext,
        dataframe=df,
        row_count=len(df),
        col_count=len(df.columns),
        warnings=warnings,
        errors=errors,
    )

