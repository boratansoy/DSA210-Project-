#!/usr/bin/env python3
"""
Process Spotify Extended Streaming History JSON exports into a cleaned CSV file.

Usage:
    python3 spotify_history_processor.py \
        --input-dir "Spotify Extended Streaming History" \
        --output-file "spotify_cleaned.csv"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


# Canonical columns we want to keep in the final dataset.
IMPORTANT_COLUMNS = [
    "ts",
    "date",
    "hour",
    "weekday",
    "ms_played",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "session_id",
]

# Common aliases in case column names vary slightly across exports.
COLUMN_ALIASES = {
    "timestamp": "ts",
    "time_stamp": "ts",
    "played_at": "ts",
    "msPlayed": "ms_played",
    "track_name": "master_metadata_track_name",
    "artist_name": "master_metadata_album_artist_name",
    "artist": "master_metadata_album_artist_name",
    "track": "master_metadata_track_name",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and output locations."""
    parser = argparse.ArgumentParser(
        description="Clean and combine Spotify Extended Streaming History JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing Spotify Extended Streaming History JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default=Path("spotify_cleaned.csv"),
        type=Path,
        help="Path for the cleaned CSV output. Default: spotify_cleaned.csv",
    )
    return parser.parse_args()


def find_json_files(input_dir: Path) -> list[Path]:
    """Return all JSON files in the given directory, sorted by filename."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")
    return json_files


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name and map common aliases to canonical names."""
    normalized = column_name.strip()
    return COLUMN_ALIASES.get(normalized, normalized)


def load_json_file(file_path: Path) -> pd.DataFrame | None:
    """
    Load one Spotify JSON file into a DataFrame.

    Returns None if the file is corrupted or does not contain a list of records.
    """
    try:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to read {file_path.name}: {exc}")
        return None

    if not isinstance(payload, list):
        print(f"Warning: skipped {file_path.name} because it does not contain a list.")
        return None

    try:
        dataframe = pd.DataFrame(payload)
    except ValueError as exc:
        print(f"Warning: failed to convert {file_path.name} into a DataFrame: {exc}")
        return None

    dataframe = dataframe.rename(columns=normalize_column_name)
    dataframe["source_file"] = file_path.name
    return dataframe


def align_dataframes(dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Align schemas across all files by creating the union of columns.

    Missing columns are added automatically as NaN before concatenation.
    """
    frames = list(dataframes)
    if not frames:
        return pd.DataFrame()

    all_columns: list[str] = sorted({column for df in frames for column in df.columns})
    aligned_frames = [df.reindex(columns=all_columns) for df in frames]
    return pd.concat(aligned_frames, ignore_index=True)


def add_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert the timestamp column and create additional calendar features."""
    if "ts" not in dataframe.columns:
        raise KeyError("The required timestamp column 'ts' was not found.")

    dataframe["ts"] = pd.to_datetime(dataframe["ts"], errors="coerce", utc=True)
    dataframe = dataframe.dropna(subset=["ts"]).copy()

    dataframe["date"] = dataframe["ts"].dt.date
    dataframe["year"] = dataframe["ts"].dt.year
    dataframe["month"] = dataframe["ts"].dt.month
    dataframe["day"] = dataframe["ts"].dt.day
    dataframe["hour"] = dataframe["ts"].dt.hour
    dataframe["weekday"] = dataframe["ts"].dt.day_name()
    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows, deduplicate, and handle missing track or artist names."""
    cleaned = dataframe.copy()

    if "ms_played" in cleaned.columns:
        cleaned["ms_played"] = pd.to_numeric(cleaned["ms_played"], errors="coerce")
        cleaned = cleaned[cleaned["ms_played"].fillna(0) > 0]

    # Use episode metadata as a fallback when track or artist values are missing.
    if "master_metadata_track_name" in cleaned.columns:
        if "episode_name" in cleaned.columns:
            cleaned["master_metadata_track_name"] = cleaned[
                "master_metadata_track_name"
            ].fillna(cleaned["episode_name"])
        cleaned["master_metadata_track_name"] = cleaned[
            "master_metadata_track_name"
        ].fillna("Unknown Track")

    if "master_metadata_album_artist_name" in cleaned.columns:
        if "episode_show_name" in cleaned.columns:
            cleaned["master_metadata_album_artist_name"] = cleaned[
                "master_metadata_album_artist_name"
            ].fillna(cleaned["episode_show_name"])
        cleaned["master_metadata_album_artist_name"] = cleaned[
            "master_metadata_album_artist_name"
        ].fillna("Unknown Artist")

    cleaned = cleaned.drop_duplicates().copy()
    return cleaned


def add_session_ids(dataframe: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    Create a session_id column.

    A new session starts when the gap between consecutive rows is greater than
    the given threshold.
    """
    sessionized = dataframe.sort_values("ts").copy()
    time_gap = sessionized["ts"].diff()
    new_session = time_gap.gt(pd.Timedelta(minutes=gap_minutes)).fillna(True)
    sessionized["session_id"] = new_session.cumsum().astype("int64")
    return sessionized


def select_output_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep the important columns that are available in the processed dataset."""
    available_columns = [column for column in IMPORTANT_COLUMNS if column in dataframe.columns]
    return dataframe.loc[:, available_columns].copy()


def print_summary(dataframe: pd.DataFrame) -> None:
    """Print summary statistics for the cleaned dataset."""
    total_rows = len(dataframe)
    min_date = dataframe["date"].min() if "date" in dataframe.columns else None
    max_date = dataframe["date"].max() if "date" in dataframe.columns else None
    unique_artists = (
        dataframe["master_metadata_album_artist_name"].nunique(dropna=True)
        if "master_metadata_album_artist_name" in dataframe.columns
        else 0
    )
    unique_tracks = (
        dataframe["master_metadata_track_name"].nunique(dropna=True)
        if "master_metadata_track_name" in dataframe.columns
        else 0
    )

    print(f"Total rows: {total_rows}")
    print(f"Date range: {min_date} to {max_date}")
    print(f"Unique artists: {unique_artists}")
    print(f"Unique tracks: {unique_tracks}")


def process_spotify_history(input_dir: Path) -> pd.DataFrame:
    """Run the full processing pipeline for all JSON files in a directory."""
    json_files = find_json_files(input_dir)

    loaded_frames: list[pd.DataFrame] = []
    for file_path in json_files:
        dataframe = load_json_file(file_path)
        if dataframe is not None and not dataframe.empty:
            loaded_frames.append(dataframe)

    if not loaded_frames:
        raise ValueError("No valid JSON files could be loaded.")

    combined = align_dataframes(loaded_frames)
    enriched = add_datetime_features(combined)
    cleaned = clean_data(enriched)
    sessionized = add_session_ids(cleaned)
    final_dataframe = select_output_columns(sessionized)
    final_dataframe = final_dataframe.sort_values("ts").reset_index(drop=True)
    return final_dataframe


def save_dataframe(dataframe: pd.DataFrame, output_file: Path) -> None:
    """Save the cleaned DataFrame to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False)


def main() -> None:
    """Parse arguments, process Spotify history, print stats, and save CSV."""
    args = parse_arguments()

    cleaned_dataframe = process_spotify_history(args.input_dir)
    print_summary(cleaned_dataframe)
    save_dataframe(cleaned_dataframe, args.output_file)
    print(f"Saved cleaned data to: {args.output_file}")


if __name__ == "__main__":
    main()
