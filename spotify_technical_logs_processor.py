#!/usr/bin/env python3
"""
Process selected Spotify Technical Log Information JSON files into clean,
analysis-ready tables.

This script intentionally focuses only on the most useful technical-log files:

- RawCoreStream.json
- RawCoreStream_1.json
- ConnectActiveDeviceChanged.json
- ConnectionInfo.json
- SessionCreation.json
- ShuffleSequenceEvent.json

It keeps technical-log outputs separate from streaming-history outputs while
preserving timestamps, session IDs, content identifiers, platform fields, and
device/context fields for later merging.

Usage:
    python3 spotify_technical_logs_processor.py \
        --input-dir "/path/to/Spotify Technical Log Information" \
        --output-dir "spotify_technical_log_outputs"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_TABLES = {
    "raw_core_stream_df": {
        "output_name": "raw_core_stream",
        "source_files": ["RawCoreStream.json", "RawCoreStream_1.json"],
        "daily_count_column": "number_of_raw_core_events",
    },
    "device_change_df": {
        "output_name": "device_change",
        "source_files": ["ConnectActiveDeviceChanged.json"],
        "daily_count_column": "number_of_device_changes",
    },
    "connection_info_df": {
        "output_name": "connection_info",
        "source_files": ["ConnectionInfo.json"],
        "daily_count_column": "number_of_connection_events",
    },
    "session_creation_df": {
        "output_name": "session_creation",
        "source_files": ["SessionCreation.json"],
        "daily_count_column": "number_of_sessions_created",
    },
    "shuffle_events_df": {
        "output_name": "shuffle_events",
        "source_files": ["ShuffleSequenceEvent.json"],
        "daily_count_column": "number_of_shuffle_events",
    },
}

TIMESTAMP_NAME_PATTERNS = [
    "timestamp",
    "event_time",
    "eventtime",
    "created",
    "created_at",
    "time",
    "date",
    "ts",
]

MAIN_TIMESTAMP_PRIORITY = [
    "timestamp_utc",
    "event_timestamp",
    "event_time",
    "eventtime",
    "message_event_time",
    "message_created_at",
    "created_at",
    "context_time",
    "context_receiver_service_timestamp",
]


@dataclass
class TableResult:
    """One parsed table plus metadata needed for summaries and exports."""

    table_name: str
    output_name: str
    dataframe: pd.DataFrame
    source_file: str
    parse_status: str
    timestamp_column_used: str | None


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process selected Spotify technical-log JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing Spotify Technical Log Information JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("spotify_technical_log_outputs"),
        type=Path,
        help="Folder where output CSV and parquet files will be saved.",
    )
    return parser.parse_args()


def pyarrow_available() -> bool:
    """Return True if pyarrow is installed and parquet output can be attempted."""
    return importlib.util.find_spec("pyarrow") is not None


def safe_load_json(file_path: Path) -> tuple[Any | None, str]:
    """
    Safely load a JSON file.

    Returns:
        A tuple of (loaded_payload, parse_status).
    """
    if not file_path.exists():
        return None, "missing_file"

    try:
        raw_text = file_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, f"read_error: {exc}"

    if not raw_text:
        return None, "empty_file"

    try:
        return json.loads(raw_text), "parsed"
    except json.JSONDecodeError as exc:
        return None, f"malformed_json: {exc}"


def detect_json_structure(payload: Any) -> str:
    """Detect whether the loaded JSON is a list, dict, scalar, or missing."""
    if payload is None:
        return "none"
    if isinstance(payload, list):
        return "list"
    if isinstance(payload, dict):
        return "dict"
    return type(payload).__name__


def normalize_json_payload(payload: Any) -> pd.DataFrame:
    """
    Normalize JSON payload into a DataFrame.

    Uses pandas.json_normalize for dictionaries, lists of dictionaries, and
    nested structures. Scalar lists are retained as a simple one-column table.
    """
    structure = detect_json_structure(payload)

    if structure == "list":
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in payload):
            return pd.json_normalize(payload, sep="_")
        return pd.DataFrame({"value": payload})

    if structure == "dict":
        return pd.json_normalize(payload, sep="_")

    if structure == "none":
        return pd.DataFrame()

    return pd.DataFrame({"value": [payload]})


def to_snake_case(column_name: Any) -> str:
    """Convert a column name to snake_case."""
    name = str(column_name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower() or "unnamed_column"


def make_unique_column_names(column_names: list[str]) -> list[str]:
    """Ensure column names remain unique after standardization."""
    seen: dict[str, int] = {}
    unique_names = []

    for column_name in column_names:
        count = seen.get(column_name, 0)
        if count == 0:
            unique_names.append(column_name)
        else:
            unique_names.append(f"{column_name}_{count + 1}")
        seen[column_name] = count + 1

    return unique_names


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize all column names to snake_case."""
    standardized = dataframe.copy()
    snake_case_names = [to_snake_case(column) for column in standardized.columns]
    standardized.columns = make_unique_column_names(snake_case_names)
    return standardized


def is_likely_timestamp_column(column_name: str) -> bool:
    """Identify likely timestamp columns using common timestamp-related terms."""
    name = column_name.lower()

    if name in {"date", "time", "timestamp", "ts", "created", "created_at"}:
        return True

    if name.endswith(("_time", "_timestamp", "_date", "_ts", "_created_at")):
        return True

    return any(pattern in name for pattern in TIMESTAMP_NAME_PATTERNS)


def infer_epoch_unit(numeric_values: pd.Series) -> str | None:
    """
    Infer epoch unit from numeric timestamp magnitudes.

    Current epoch-like values are typically:
    - seconds:      ~1e9
    - milliseconds: ~1e12
    - microseconds: ~1e15
    - nanoseconds:  ~1e18
    """
    non_null = numeric_values.dropna()
    if non_null.empty:
        return None

    median_abs = non_null.abs().median()

    if median_abs >= 1e17:
        return "ns"
    if median_abs >= 1e14:
        return "us"
    if median_abs >= 1e11:
        return "ms"
    if median_abs >= 1e9:
        return "s"

    return None


def convert_series_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a likely timestamp series to datetime safely.

    Numeric epoch timestamps are converted using an inferred unit. String
    timestamps are parsed with pandas after removing bracketed timezone suffixes.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    numeric_values = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_values.notna().mean() if len(series) else 0

    if numeric_ratio >= 0.75:
        unit = infer_epoch_unit(numeric_values)
        if unit:
            return pd.to_datetime(numeric_values, errors="coerce", unit=unit, utc=True)

    cleaned_strings = (
        series.astype("string")
        .str.replace(r"\[.*\]$", "", regex=True)
        .str.strip()
    )
    return pd.to_datetime(cleaned_strings, errors="coerce", utc=True)


def convert_likely_timestamp_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert all likely timestamp columns to datetime.

    Returns:
        (dataframe_with_converted_columns, successfully_converted_column_names)
    """
    converted = dataframe.copy()
    timestamp_columns = []

    for column in converted.columns:
        if not is_likely_timestamp_column(column):
            continue

        parsed = convert_series_to_datetime(converted[column])
        original_non_null = converted[column].notna().sum()
        parsed_non_null = parsed.notna().sum()

        if parsed_non_null > 0 and (original_non_null == 0 or parsed_non_null / original_non_null >= 0.25):
            converted[column] = parsed
            timestamp_columns.append(column)

    return converted, timestamp_columns


def choose_main_timestamp_column(
    dataframe: pd.DataFrame,
    timestamp_columns: list[str],
) -> str | None:
    """Choose the best available timestamp column for sorting and date features."""
    usable_columns = [
        column for column in timestamp_columns if column in dataframe.columns and dataframe[column].notna().any()
    ]

    if not usable_columns:
        return None

    for preferred_column in MAIN_TIMESTAMP_PRIORITY:
        if preferred_column in usable_columns:
            return preferred_column

    return max(usable_columns, key=lambda column: dataframe[column].notna().sum())


def add_time_features(
    dataframe: pd.DataFrame,
    timestamp_column: str | None,
) -> pd.DataFrame:
    """Add consistent event timestamp and calendar features."""
    enriched = dataframe.copy()

    if not timestamp_column or timestamp_column not in enriched.columns:
        return enriched

    enriched["event_timestamp"] = enriched[timestamp_column]
    enriched["date"] = enriched["event_timestamp"].dt.date
    enriched["year"] = enriched["event_timestamp"].dt.year
    enriched["month"] = enriched["event_timestamp"].dt.month
    enriched["day"] = enriched["event_timestamp"].dt.day
    enriched["hour"] = enriched["event_timestamp"].dt.hour
    enriched["weekday"] = enriched["event_timestamp"].dt.day_name()

    return enriched


def serialize_complex_value(value: Any) -> Any:
    """Serialize complex cell values so CSV/parquet exports are stable."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def serialize_complex_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert complex object values to JSON strings."""
    exported = dataframe.copy()

    for column in exported.columns:
        if exported[column].dtype == "object":
            exported[column] = exported[column].map(serialize_complex_value)

    return exported


def make_hashable_for_dedupe(value: Any) -> Any:
    """Create a stable comparable representation for duplicate detection."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def drop_exact_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, including rows with list or dict values."""
    if dataframe.empty:
        return dataframe

    comparable = dataframe.apply(lambda column: column.map(make_hashable_for_dedupe))
    return dataframe.loc[~comparable.duplicated()].copy()


def clean_dataframe(dataframe: pd.DataFrame, timestamp_column: str | None) -> pd.DataFrame:
    """
    Apply conservative cleaning.

    Drops fully empty rows, drops fully empty columns, drops exact duplicates, and
    sorts by the main timestamp when one is available.
    """
    cleaned = dataframe.copy()
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = drop_exact_duplicates(cleaned)

    if timestamp_column and timestamp_column in cleaned.columns:
        cleaned = cleaned.sort_values(timestamp_column, kind="mergesort")

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def reorder_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Move common merge/context columns to the front while preserving all columns."""
    preferred_columns = [
        "source_file",
        "event_timestamp",
        "timestamp_utc",
        "context_time",
        "message_created_at",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "message_session_id",
        "session_id",
        "message_content_uri",
        "message_displayed_content_uri",
        "message_audio_id",
        "context_device_manufacturer",
        "context_device_model",
        "context_device_type",
        "context_os_name",
        "context_os_version",
        "context_application_version",
        "context_conn_country",
    ]

    front_columns = [column for column in preferred_columns if column in dataframe.columns]
    remaining_columns = [column for column in dataframe.columns if column not in front_columns]
    return dataframe.loc[:, front_columns + remaining_columns]


def parse_table_group(
    input_dir: Path,
    table_name: str,
    table_config: dict[str, Any],
) -> TableResult:
    """Parse one logical output table from one or more source files."""
    frames = []
    status_parts = []
    source_files_used = []

    for source_file_name in table_config["source_files"]:
        file_path = input_dir / source_file_name
        payload, status = safe_load_json(file_path)
        status_parts.append(f"{source_file_name}:{status}")

        if payload is None:
            continue

        frame = normalize_json_payload(payload)
        frame["source_file"] = source_file_name
        frame["json_structure"] = detect_json_structure(payload)
        frames.append(frame)
        source_files_used.append(source_file_name)

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=["source_file", "json_structure"])

    combined = standardize_column_names(combined)
    combined, converted_timestamp_columns = convert_likely_timestamp_columns(combined)
    main_timestamp_column = choose_main_timestamp_column(combined, converted_timestamp_columns)
    combined = add_time_features(combined, main_timestamp_column)
    combined = clean_dataframe(combined, main_timestamp_column or "event_timestamp")
    combined = reorder_columns(combined)

    return TableResult(
        table_name=table_name,
        output_name=table_config["output_name"],
        dataframe=combined,
        source_file=",".join(source_files_used) if source_files_used else ",".join(table_config["source_files"]),
        parse_status="; ".join(status_parts),
        timestamp_column_used=main_timestamp_column,
    )


def save_dataframe(dataframe: pd.DataFrame, output_dir: Path, output_name: str) -> None:
    """Save a DataFrame as CSV and parquet when pyarrow is available."""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_df = serialize_complex_columns(dataframe)
    export_df.to_csv(output_dir / f"{output_name}.csv", index=False)

    if pyarrow_available():
        try:
            export_df.to_parquet(output_dir / f"{output_name}.parquet", index=False)
        except Exception as exc:
            print(f"Warning: could not save parquet for {output_name}: {exc}")


def get_timestamp_bounds(
    dataframe: pd.DataFrame,
    timestamp_column: str | None,
) -> tuple[Any, Any]:
    """Return min and max timestamp values for metadata."""
    if not timestamp_column or timestamp_column not in dataframe.columns:
        if "event_timestamp" not in dataframe.columns:
            return pd.NA, pd.NA
        timestamp_column = "event_timestamp"

    values = dataframe[timestamp_column].dropna()
    if values.empty:
        return pd.NA, pd.NA

    return values.min(), values.max()


def build_metadata_summary(table_results: list[TableResult]) -> pd.DataFrame:
    """Build metadata summary table for all parsed technical-log tables."""
    rows = []

    for result in table_results:
        min_timestamp, max_timestamp = get_timestamp_bounds(
            result.dataframe,
            result.timestamp_column_used,
        )

        rows.append(
            {
                "source_file": result.source_file,
                "output_table_name": result.table_name,
                "rows": len(result.dataframe),
                "columns": len(result.dataframe.columns),
                "column_names": list(result.dataframe.columns),
                "parse_status": result.parse_status,
                "timestamp_column_used": result.timestamp_column_used,
                "min_timestamp": min_timestamp,
                "max_timestamp": max_timestamp,
            }
        )

    return pd.DataFrame(rows)


def build_daily_context_summary(table_results: list[TableResult]) -> pd.DataFrame:
    """
    Build one daily summary table intended for later merging with streaming history.

    Each target table contributes one count column if it has usable date features.
    """
    daily_frames = []

    for result in table_results:
        config = TARGET_TABLES.get(result.table_name)
        if not config:
            continue

        dataframe = result.dataframe
        if dataframe.empty or "date" not in dataframe.columns:
            continue

        dates = dataframe["date"].dropna()
        if dates.empty:
            continue

        daily_counts = (
            dataframe.dropna(subset=["date"])
            .groupby("date")
            .size()
            .reset_index(name=config["daily_count_column"])
        )
        daily_frames.append(daily_counts)

    if not daily_frames:
        return pd.DataFrame(
            columns=[
                "date",
                "number_of_raw_core_events",
                "number_of_device_changes",
                "number_of_connection_events",
                "number_of_sessions_created",
                "number_of_shuffle_events",
            ]
        )

    daily_summary = daily_frames[0]
    for daily_frame in daily_frames[1:]:
        daily_summary = daily_summary.merge(daily_frame, on="date", how="outer")

    count_columns = [column for column in daily_summary.columns if column != "date"]
    daily_summary[count_columns] = daily_summary[count_columns].fillna(0).astype("int64")
    daily_summary["date"] = pd.to_datetime(daily_summary["date"], errors="coerce")
    daily_summary = daily_summary.sort_values("date").reset_index(drop=True)
    daily_summary["date"] = daily_summary["date"].dt.date

    return daily_summary


def print_dataframe_summary(result: TableResult) -> None:
    """Print concise summary for one parsed DataFrame."""
    first_columns = list(result.dataframe.columns[:10])
    print(f"\nDataFrame: {result.table_name}")
    print(f"Rows: {len(result.dataframe)}")
    print(f"Columns: {len(result.dataframe.columns)}")
    print(f"Timestamp column used: {result.timestamp_column_used}")
    print(f"First columns: {first_columns}")


def print_interpretation(table_results: list[TableResult], daily_summary: pd.DataFrame) -> None:
    """Print a short interpretation of useful downstream analysis roles."""
    non_empty_tables = {
        result.table_name
        for result in table_results
        if result.dataframe is not None and not result.dataframe.empty
    }

    print("\nInterpretation for later analysis:")
    if {"device_change_df", "connection_info_df"} & non_empty_tables:
        print(
            "- Device/context analysis: device_change_df and connection_info_df are best for studying "
            "how platform, device type, and network context vary across normal days versus special days."
        )
    if "session_creation_df" in non_empty_tables:
        print(
            "- Session analysis: session_creation_df is useful for measuring when listening sessions begin "
            "and can later be joined to streaming sessions using timestamps or session identifiers."
        )
    if {"raw_core_stream_df", "shuffle_events_df"} & non_empty_tables:
        print(
            "- Playback context analysis: raw_core_stream_df and shuffle_events_df preserve playback reasons, "
            "content URIs, shuffle state, and device context that can enrich streaming-history behavior."
        )
    if not daily_summary.empty:
        print(
            "- Merging with extended streaming history: daily_context_summary_df is the easiest table to join "
            "with daily streaming aggregates by date, while detailed tables can be merged by timestamp, URI, "
            "session ID, device, or platform fields."
        )


def process_technical_logs(input_dir: Path) -> tuple[list[TableResult], pd.DataFrame, pd.DataFrame]:
    """Run the full technical-log processing pipeline."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    table_results = [
        parse_table_group(input_dir, table_name, table_config)
        for table_name, table_config in TARGET_TABLES.items()
    ]

    metadata_summary = build_metadata_summary(table_results)
    daily_context_summary = build_daily_context_summary(table_results)

    return table_results, metadata_summary, daily_context_summary


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    table_results, metadata_summary, daily_context_summary = process_technical_logs(
        args.input_dir
    )

    for result in table_results:
        print_dataframe_summary(result)
        save_dataframe(result.dataframe, args.output_dir, result.output_name)

    metadata_result = TableResult(
        table_name="technical_logs_metadata_df",
        output_name="technical_logs_metadata",
        dataframe=metadata_summary,
        source_file="multiple",
        parse_status="generated",
        timestamp_column_used=None,
    )
    print_dataframe_summary(metadata_result)
    save_dataframe(metadata_summary, args.output_dir, "technical_logs_metadata")

    daily_result = TableResult(
        table_name="daily_context_summary_df",
        output_name="daily_context_summary",
        dataframe=daily_context_summary,
        source_file="multiple",
        parse_status="generated",
        timestamp_column_used="date" if not daily_context_summary.empty else None,
    )
    print_dataframe_summary(daily_result)
    save_dataframe(daily_context_summary, args.output_dir, "daily_context_summary")

    print_interpretation(table_results, daily_context_summary)
    print(f"\nSaved technical-log outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
