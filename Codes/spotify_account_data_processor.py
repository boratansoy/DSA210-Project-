#!/usr/bin/env python3
"""
Process Spotify account-data JSON exports into structured, analysis-ready tables.

This script intentionally skips the legacy 1-year streaming history and podcast
history files (for example, ``StreamingHistory_music_0.json``), because those
should remain separate from the extended streaming history pipeline.

Usage:
    python3 spotify_account_data_processor.py \
        --input-dir "/path/to/Spotify Account Data" \
        --output-dir "/path/to/output_tables"
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


@dataclass
class TableResult:
    """Container for one parsed output table and its metadata."""

    name: str
    dataframe: pd.DataFrame
    source_file: str
    parse_status: str


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Spotify account-data JSON files into separate tables."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing Spotify account-data JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("spotify_account_data_outputs"),
        type=Path,
        help="Folder where CSV and parquet tables will be saved.",
    )
    return parser.parse_args()


def is_streaming_history_file(file_path: Path) -> bool:
    """Return True for legacy streaming-history files that should be ignored."""
    return file_path.name.startswith("StreamingHistory_")


def list_account_json_files(input_dir: Path) -> list[Path]:
    """List JSON files in the input directory, excluding streaming-history files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    json_files = sorted(
        path for path in input_dir.glob("*.json") if not is_streaming_history_file(path)
    )
    if not json_files:
        raise FileNotFoundError(
            "No account-data JSON files were found after excluding StreamingHistory_* files."
        )
    return json_files


def safe_load_json(file_path: Path) -> tuple[Any | None, str]:
    """
    Load JSON content with strong error handling.

    Returns:
        (data, status)
    """
    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, f"error: could not read file ({exc})"

    if not text:
        return None, "empty_file"

    try:
        return json.loads(text), "parsed"
    except json.JSONDecodeError as exc:
        return None, f"error: malformed json ({exc})"


def detect_json_structure(data: Any) -> str:
    """Detect the broad JSON structure type for a loaded payload."""
    if data is None:
        return "none"
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        return "dict"
    return type(data).__name__


def ensure_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure all expected columns exist."""
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe


def parse_datetime_series(series: pd.Series, utc: bool = True) -> pd.Series:
    """
    Parse timestamps robustly.

    Spotify sometimes appends timezone annotations such as ``[UTC]``.
    """
    cleaned = series.astype("string").str.replace(r"\[.*\]$", "", regex=True).str.strip()
    return pd.to_datetime(cleaned, errors="coerce", utc=utc)


def add_date_parts(dataframe: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Add common calendar features derived from a datetime column."""
    dataframe = dataframe.copy()
    dataframe["date"] = dataframe[datetime_column].dt.date
    dataframe["year"] = dataframe[datetime_column].dt.year
    dataframe["month"] = dataframe[datetime_column].dt.month
    dataframe["day"] = dataframe[datetime_column].dt.day
    dataframe["hour"] = dataframe[datetime_column].dt.hour
    dataframe["weekday"] = dataframe[datetime_column].dt.day_name()
    return dataframe


def flatten_scalar_summary(
    payload: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """
    Flatten only scalar summary values from nested dictionaries.

    Lists are summarized using a simple count so the one-row summary stays compact.
    """
    flattened: dict[str, Any] = {}

    for key, value in payload.items():
        current_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_scalar_summary(value, current_key))
        elif isinstance(value, list):
            flattened[f"{current_key}_count"] = len(value)
        else:
            flattened[current_key] = value

    return flattened


def infer_inference_type(label: str | None) -> str:
    """Derive a coarse inference type from the label prefix or naming pattern."""
    if not label:
        return "other"

    normalized = label.lower()
    if normalized.startswith("demographic_"):
        return "demographic"
    if normalized.startswith("interest_"):
        return "interest"
    if normalized.startswith("content_"):
        return "content"
    if normalized.startswith(("1p_", "2p_", "3p_")) or "custom" in normalized:
        return "custom"
    return "other"


def make_ranked_uri_table(
    values: list[Any],
    column_name: str,
    source_file: str,
) -> pd.DataFrame:
    """Create a ranked one-column table from a list of scalar values."""
    if not values:
        return pd.DataFrame(columns=["rank", column_name, "source_file"])

    dataframe = pd.DataFrame({column_name: values})
    dataframe.insert(0, "rank", range(1, len(dataframe) + 1))
    dataframe["source_file"] = source_file
    return dataframe


def serialize_complex_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert lists and dictionaries to JSON strings before export."""
    exported = dataframe.copy()

    for column in exported.columns:
        if exported[column].dtype == "object":
            exported[column] = exported[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict, tuple, set))
                else value
            )

    return exported


def print_dataset_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Print dataset-level summary details."""
    print(f"\nDataset: {name}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Column names: {list(dataframe.columns)}")


def parquet_is_available() -> bool:
    """Check whether pyarrow is installed for parquet output."""
    return importlib.util.find_spec("pyarrow") is not None


def save_table(dataframe: pd.DataFrame, output_dir: Path, table_name: str) -> None:
    """Save one DataFrame as CSV and parquet when available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    export_df = serialize_complex_values(dataframe)

    csv_path = output_dir / f"{table_name}.csv"
    export_df.to_csv(csv_path, index=False)

    if parquet_is_available():
        parquet_path = output_dir / f"{table_name}.parquet"
        try:
            export_df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            print(f"Warning: failed to save parquet for {table_name}: {exc}")


def parse_search_queries(file_path: Path) -> list[TableResult]:
    """Parse SearchQueries.json into a search-behavior table."""
    data, status = safe_load_json(file_path)
    empty_columns = [
        "platform",
        "searchTime",
        "searchQuery",
        "searchInteractionURIs",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "query_length",
        "num_clicked_results",
        "source_file",
    ]

    if data is None:
        return [
            TableResult(
                name="search_queries_df",
                dataframe=pd.DataFrame(columns=empty_columns),
                source_file=file_path.name,
                parse_status=status,
            )
        ]

    structure = detect_json_structure(data)
    if structure == "list":
        dataframe = pd.json_normalize(data, sep="_")
    elif structure == "dict":
        candidate = next(
            (
                value
                for value in data.values()
                if isinstance(value, list) and value and isinstance(value[0], dict)
            ),
            [],
        )
        dataframe = pd.json_normalize(candidate, sep="_")
    else:
        dataframe = pd.DataFrame()

    dataframe = ensure_columns(
        dataframe,
        ["platform", "searchTime", "searchQuery", "searchInteractionURIs"],
    )
    dataframe["searchTime"] = parse_datetime_series(dataframe["searchTime"])
    dataframe = add_date_parts(dataframe, "searchTime")
    dataframe["query_length"] = dataframe["searchQuery"].astype("string").str.len()
    dataframe["num_clicked_results"] = dataframe["searchInteractionURIs"].map(
        lambda value: len(value) if isinstance(value, list) else 0
    )
    dataframe["source_file"] = file_path.name

    ordered_columns = [
        "platform",
        "searchTime",
        "searchQuery",
        "searchInteractionURIs",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "query_length",
        "num_clicked_results",
        "source_file",
    ]
    dataframe = dataframe.loc[:, ordered_columns]

    return [
        TableResult(
            name="search_queries_df",
            dataframe=dataframe,
            source_file=file_path.name,
            parse_status=status,
        )
    ]


def parse_playlists(file_paths: list[Path]) -> list[TableResult]:
    """Parse one or more Playlist*.json files into one flattened item-level table."""
    playlist_frames: list[pd.DataFrame] = []
    source_names: list[str] = []
    parse_statuses: list[str] = []

    for file_path in file_paths:
        data, status = safe_load_json(file_path)
        source_names.append(file_path.name)
        parse_statuses.append(f"{file_path.name}:{status}")

        if data is None:
            continue

        playlists = []
        if isinstance(data, dict):
            playlists = data.get("playlists", [])
        elif isinstance(data, list):
            playlists = data

        if not playlists:
            continue

        flattened = pd.json_normalize(
            playlists,
            record_path="items",
            meta=["name", "lastModifiedDate", "numberOfFollowers"],
            errors="ignore",
            sep="_",
        )

        if flattened.empty:
            continue

        flattened = flattened.rename(
            columns={
                "name": "playlist_name",
                "lastModifiedDate": "playlist_last_modified_date",
                "addedDate": "item_added_date",
                "track_trackName": "track_name",
                "track_artistName": "artist_name",
                "track_albumName": "album_name",
                "track_trackUri": "track_uri",
                "episode_episodeName": "episode_name",
                "episode_showName": "show_name",
                "episode_episodeUri": "episode_uri",
            }
        )
        flattened["item_type"] = pd.NA
        flattened.loc[flattened["track_uri"].notna(), "item_type"] = "track"
        flattened.loc[flattened["episode_uri"].notna(), "item_type"] = "episode"

        flattened["playlist_last_modified_date"] = pd.to_datetime(
            flattened["playlist_last_modified_date"], errors="coerce"
        )
        flattened["item_added_date"] = pd.to_datetime(
            flattened["item_added_date"], errors="coerce"
        )
        flattened["source_file"] = file_path.name

        ordered_columns = [
            "playlist_name",
            "playlist_last_modified_date",
            "item_added_date",
            "item_type",
            "track_name",
            "artist_name",
            "album_name",
            "track_uri",
            "episode_name",
            "show_name",
            "episode_uri",
            "numberOfFollowers",
            "source_file",
        ]
        flattened = ensure_columns(flattened, ordered_columns)
        playlist_frames.append(flattened.loc[:, ordered_columns])

    if playlist_frames:
        combined = pd.concat(playlist_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "playlist_name",
                "playlist_last_modified_date",
                "item_added_date",
                "item_type",
                "track_name",
                "artist_name",
                "album_name",
                "track_uri",
                "episode_name",
                "show_name",
                "episode_uri",
                "numberOfFollowers",
                "source_file",
            ]
        )

    return [
        TableResult(
            name="playlists_df",
            dataframe=combined,
            source_file=",".join(source_names),
            parse_status="; ".join(parse_statuses),
        )
    ]


def parse_library_tracks(file_path: Path) -> list[TableResult]:
    """Parse the saved tracks portion of YourLibrary.json."""
    data, status = safe_load_json(file_path)
    columns = ["artist", "album", "track", "uri", "source_file"]

    if data is None:
        return [
            TableResult(
                name="library_tracks_df",
                dataframe=pd.DataFrame(columns=columns),
                source_file=file_path.name,
                parse_status=status,
            )
        ]

    track_records = data.get("tracks", []) if isinstance(data, dict) else []
    dataframe = pd.json_normalize(track_records, sep="_")
    dataframe = ensure_columns(dataframe, ["artist", "album", "track", "uri"])
    dataframe["source_file"] = file_path.name
    dataframe = dataframe.loc[:, columns]

    return [
        TableResult(
            name="library_tracks_df",
            dataframe=dataframe,
            source_file=file_path.name,
            parse_status=status,
        )
    ]


def parse_wrapped(file_path: Path) -> list[TableResult]:
    """Parse Wrapped2025.json into summary and nested ranking tables."""
    data, status = safe_load_json(file_path)
    results: list[TableResult] = []

    if data is None or not isinstance(data, dict):
        empty_tables = {
            "wrapped_summary_df": pd.DataFrame(),
            "wrapped_top_tracks_df": pd.DataFrame(),
            "wrapped_top_artists_df": pd.DataFrame(),
            "wrapped_top_genres_df": pd.DataFrame(),
            "wrapped_top_albums_df": pd.DataFrame(),
        }
        for table_name, dataframe in empty_tables.items():
            results.append(
                TableResult(
                    name=table_name,
                    dataframe=dataframe,
                    source_file=file_path.name,
                    parse_status=status,
                )
            )
        return results

    summary_row = flatten_scalar_summary(data)
    wrapped_summary_df = pd.DataFrame([summary_row])
    wrapped_summary_df["source_file"] = file_path.name

    top_tracks = data.get("topTracks", {}).get("topTracks", [])
    wrapped_top_tracks_df = pd.json_normalize(top_tracks, sep="_")
    if wrapped_top_tracks_df.empty:
        wrapped_top_tracks_df = pd.DataFrame(columns=["rank", "trackUri", "count", "msPlayed"])
    wrapped_top_tracks_df.insert(0, "rank", range(1, len(wrapped_top_tracks_df) + 1))
    wrapped_top_tracks_df = wrapped_top_tracks_df.rename(
        columns={"trackUri": "track_uri", "msPlayed": "ms_played"}
    )
    wrapped_top_tracks_df["source_file"] = file_path.name

    wrapped_top_artists_df = make_ranked_uri_table(
        data.get("topArtists", {}).get("topArtistUris", []),
        "artist_uri",
        file_path.name,
    )
    wrapped_top_genres_df = make_ranked_uri_table(
        data.get("topGenres", {}).get("topGenres", []),
        "genre_uri",
        file_path.name,
    )
    wrapped_top_albums_df = make_ranked_uri_table(
        data.get("topAlbums", {}).get("topAlbums", []),
        "album_uri",
        file_path.name,
    )

    for table_name, dataframe in {
        "wrapped_summary_df": wrapped_summary_df,
        "wrapped_top_tracks_df": wrapped_top_tracks_df,
        "wrapped_top_artists_df": wrapped_top_artists_df,
        "wrapped_top_genres_df": wrapped_top_genres_df,
        "wrapped_top_albums_df": wrapped_top_albums_df,
    }.items():
        results.append(
            TableResult(
                name=table_name,
                dataframe=dataframe,
                source_file=file_path.name,
                parse_status=status,
            )
        )

    return results


def parse_sound_capsule(file_path: Path) -> list[TableResult]:
    """Parse YourSoundCapsule.json into daily and nested top-entity tables."""
    data, status = safe_load_json(file_path)
    results: list[TableResult] = []

    sound_capsule_df = pd.DataFrame(
        columns=["date", "streamCount", "secondsPlayed", "source_file"]
    )
    sound_capsule_top_tracks_df = pd.DataFrame(
        columns=["stat_date", "rank", "name", "streamCount", "secondsPlayed", "source_file"]
    )
    sound_capsule_top_artists_df = pd.DataFrame(
        columns=["stat_date", "rank", "name", "streamCount", "secondsPlayed", "source_file"]
    )

    if data is not None and isinstance(data, dict):
        stats = data.get("stats", [])
        stats_df = pd.json_normalize(stats, sep="_")
        if not stats_df.empty:
            stats_df = ensure_columns(stats_df, ["date", "streamCount", "secondsPlayed"])
            stats_df["date"] = pd.to_datetime(stats_df["date"], errors="coerce")
            stats_df["source_file"] = file_path.name
            sound_capsule_df = stats_df.loc[:, ["date", "streamCount", "secondsPlayed", "source_file"]]

        top_track_rows: list[dict[str, Any]] = []
        top_artist_rows: list[dict[str, Any]] = []

        for stat in stats:
            stat_date = stat.get("date")

            for index, track in enumerate(stat.get("topTracks", []), start=1):
                top_track_rows.append(
                    {
                        "stat_date": stat_date,
                        "rank": index,
                        "name": track.get("name"),
                        "streamCount": track.get("streamCount"),
                        "secondsPlayed": track.get("secondsPlayed"),
                        "source_file": file_path.name,
                    }
                )

            for index, artist in enumerate(stat.get("topArtists", []), start=1):
                top_artist_rows.append(
                    {
                        "stat_date": stat_date,
                        "rank": index,
                        "name": artist.get("name"),
                        "streamCount": artist.get("streamCount"),
                        "secondsPlayed": artist.get("secondsPlayed"),
                        "source_file": file_path.name,
                    }
                )

        sound_capsule_top_tracks_df = pd.DataFrame(top_track_rows)
        sound_capsule_top_artists_df = pd.DataFrame(top_artist_rows)

        if not sound_capsule_top_tracks_df.empty:
            sound_capsule_top_tracks_df["stat_date"] = pd.to_datetime(
                sound_capsule_top_tracks_df["stat_date"], errors="coerce"
            )
        else:
            sound_capsule_top_tracks_df = pd.DataFrame(
                columns=["stat_date", "rank", "name", "streamCount", "secondsPlayed", "source_file"]
            )

        if not sound_capsule_top_artists_df.empty:
            sound_capsule_top_artists_df["stat_date"] = pd.to_datetime(
                sound_capsule_top_artists_df["stat_date"], errors="coerce"
            )
        else:
            sound_capsule_top_artists_df = pd.DataFrame(
                columns=["stat_date", "rank", "name", "streamCount", "secondsPlayed", "source_file"]
            )

    for table_name, dataframe in {
        "sound_capsule_df": sound_capsule_df,
        "sound_capsule_top_tracks_df": sound_capsule_top_tracks_df,
        "sound_capsule_top_artists_df": sound_capsule_top_artists_df,
    }.items():
        results.append(
            TableResult(
                name=table_name,
                dataframe=dataframe,
                source_file=file_path.name,
                parse_status=status,
            )
        )

    return results


def parse_inferences(file_path: Path) -> list[TableResult]:
    """Parse inference labels into one row per label."""
    data, status = safe_load_json(file_path)
    columns = ["inference_label", "inference_type", "source_file"]

    if data is None:
        return [
            TableResult(
                name="inferences_df",
                dataframe=pd.DataFrame(columns=columns),
                source_file=file_path.name,
                parse_status=status,
            )
        ]

    labels = data.get("inferences", []) if isinstance(data, dict) else []
    dataframe = pd.DataFrame({"inference_label": labels})
    dataframe["inference_type"] = dataframe["inference_label"].map(infer_inference_type)
    dataframe["source_file"] = file_path.name
    dataframe = dataframe.loc[:, columns]

    return [
        TableResult(
            name="inferences_df",
            dataframe=dataframe,
            source_file=file_path.name,
            parse_status=status,
        )
    ]


def parse_follow(file_path: Path) -> list[TableResult]:
    """Parse Follow.json into following, followers, and blocking tables."""
    data, status = safe_load_json(file_path)
    relationship_map = {
        "following_df": "userIsFollowing",
        "followers_df": "userIsFollowedBy",
        "blocking_df": "userIsBlocking",
    }
    results: list[TableResult] = []

    for table_name, field_name in relationship_map.items():
        if data is not None and isinstance(data, dict):
            values = data.get(field_name, [])
        else:
            values = []

        dataframe = pd.DataFrame({"account_name": values})
        dataframe["source_file"] = file_path.name
        results.append(
            TableResult(
                name=table_name,
                dataframe=dataframe,
                source_file=file_path.name,
                parse_status=status,
            )
        )

    return results


def extract_first_value(data: Any, key: str) -> Any:
    """Extract a field from either a dictionary or a list of dictionaries."""
    if isinstance(data, dict):
        return data.get(key)
    if isinstance(data, list):
        values = [item.get(key) for item in data if isinstance(item, dict) and key in item]
        values = [value for value in values if value not in (None, "", [])]
        if not values:
            return pd.NA
        return "; ".join(map(str, values))
    return pd.NA


def has_meaningful_value(value: Any) -> bool:
    """Return True when a value should be treated as present."""
    if value is pd.NA or value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return False
    return True


def coalesce_values(*values: Any) -> Any:
    """Return the first meaningful value from a list of candidates."""
    for value in values:
        if has_meaningful_value(value):
            return value
    return pd.NA


def parse_profile(
    identity_path: Path | None,
    user_attributes_path: Path | None,
    identifiers_path: Path | None,
    payments_path: Path | None,
) -> list[TableResult]:
    """Create a compact one-row profile table from several account files."""
    sources: list[str] = []
    statuses: list[str] = []
    payloads: dict[str, Any] = {}

    file_map = {
        "identity": identity_path,
        "user_attributes": user_attributes_path,
        "identifiers": identifiers_path,
        "payments": payments_path,
    }

    for key, file_path in file_map.items():
        if file_path is None:
            payloads[key] = None
            continue
        data, status = safe_load_json(file_path)
        payloads[key] = data
        sources.append(file_path.name)
        statuses.append(f"{file_path.name}:{status}")

    profile_row = {
        "displayName": extract_first_value(payloads["identity"], "displayName"),
        "username": extract_first_value(payloads["user_attributes"], "username"),
        "email": extract_first_value(payloads["user_attributes"], "email"),
        "country": coalesce_values(
            extract_first_value(payloads["user_attributes"], "country"),
            extract_first_value(payloads["payments"], "country"),
        ),
        "birthdate": extract_first_value(payloads["user_attributes"], "birthdate"),
        "gender": extract_first_value(payloads["user_attributes"], "gender"),
        "creationTime": extract_first_value(payloads["user_attributes"], "creationTime"),
        "identifierType": extract_first_value(payloads["identifiers"], "identifierType"),
        "identifierValue": extract_first_value(payloads["identifiers"], "identifierValue"),
        "payment_method": extract_first_value(payloads["payments"], "payment_method"),
        "creation_date": extract_first_value(payloads["payments"], "creation_date"),
        "source_file": ",".join(sources),
    }

    dataframe = pd.DataFrame([profile_row])
    dataframe["birthdate"] = pd.to_datetime(dataframe["birthdate"], errors="coerce")
    dataframe["creationTime"] = parse_datetime_series(dataframe["creationTime"], utc=True)
    dataframe["creation_date"] = pd.to_datetime(dataframe["creation_date"], errors="coerce")

    return [
        TableResult(
            name="profile_df",
            dataframe=dataframe,
            source_file=",".join(sources),
            parse_status="; ".join(statuses) if statuses else "missing",
        )
    ]


def parse_customer_service(
    customer_service_paths: list[Path],
    agent_gateway_path: Path | None,
) -> list[TableResult]:
    """Parse customer-service and agent-gateway files safely."""
    results: list[TableResult] = []
    message_frames: list[pd.DataFrame] = []
    statuses: list[str] = []
    source_names: list[str] = []

    for file_path in customer_service_paths:
        data, status = safe_load_json(file_path)
        statuses.append(f"{file_path.name}:{status}")
        source_names.append(file_path.name)

        if data is None:
            continue

        if isinstance(data, dict):
            dataframe = pd.json_normalize(data, sep="_")
        elif isinstance(data, list):
            dataframe = pd.json_normalize(data, sep="_")
        else:
            dataframe = pd.DataFrame({"value": [data]})

        dataframe["source_file"] = file_path.name
        if "messageDate" in dataframe.columns:
            dataframe["messageDate"] = parse_datetime_series(dataframe["messageDate"], utc=True)
        message_frames.append(dataframe)

    if message_frames:
        customer_service_messages_df = pd.concat(message_frames, ignore_index=True)
    else:
        customer_service_messages_df = pd.DataFrame(
            columns=[
                "channel",
                "messageDate",
                "messageSource",
                "subject",
                "message",
                "source_file",
            ]
        )

    results.append(
        TableResult(
            name="customer_service_messages_df",
            dataframe=customer_service_messages_df,
            source_file=",".join(source_names),
            parse_status="; ".join(statuses) if statuses else "missing",
        )
    )

    agent_status = "missing"
    agent_source = agent_gateway_path.name if agent_gateway_path else ""
    agent_gateway_conversations_df = pd.DataFrame()

    if agent_gateway_path is not None:
        data, agent_status = safe_load_json(agent_gateway_path)
        if data is not None and isinstance(data, dict):
            conversations = data.get("conversations", [])
            agent_gateway_conversations_df = pd.json_normalize(conversations, sep="_")
            agent_gateway_conversations_df["source_file"] = agent_gateway_path.name
        else:
            agent_gateway_conversations_df = pd.DataFrame(columns=["source_file"])

    results.append(
        TableResult(
            name="agent_gateway_conversations_df",
            dataframe=agent_gateway_conversations_df,
            source_file=agent_source,
            parse_status=agent_status,
        )
    )

    return results


def parse_generic_json(file_path: Path) -> list[TableResult]:
    """
    Generic parser for account-data files that are not covered by a custom parser.

    This keeps the pipeline extensible for files such as Marquee.json or future
    Spotify exports with similar structures.
    """
    data, status = safe_load_json(file_path)
    table_name = f"{re.sub(r'[^0-9a-zA-Z]+', '_', file_path.stem).strip('_').lower()}_df"

    if data is None:
        return [
            TableResult(
                name=table_name,
                dataframe=pd.DataFrame(),
                source_file=file_path.name,
                parse_status=status,
            )
        ]

    structure = detect_json_structure(data)
    if structure == "list":
        if data and all(isinstance(item, dict) for item in data):
            dataframe = pd.json_normalize(data, sep="_")
        else:
            dataframe = pd.DataFrame({"value": data})
    elif structure == "dict":
        top_level_lists = [
            (key, value)
            for key, value in data.items()
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value)
        ]
        if len(top_level_lists) == 1:
            _, records = top_level_lists[0]
            dataframe = pd.json_normalize(records, sep="_")
        else:
            dataframe = pd.json_normalize(data, sep="_")
    else:
        dataframe = pd.DataFrame({"value": [data]})

    dataframe["source_file"] = file_path.name

    return [
        TableResult(
            name=table_name,
            dataframe=dataframe,
            source_file=file_path.name,
            parse_status=status,
        )
    ]


def identify_music_behavior_tables(table_results: list[TableResult]) -> dict[str, list[str]]:
    """Highlight datasets most relevant to music-behavior analysis."""
    available = {
        result.name
        for result in table_results
        if result.dataframe is not None and not result.dataframe.empty
    }

    categories = {
        "search_behavior": ["search_queries_df"],
        "playlist_library_preference": ["playlists_df", "library_tracks_df"],
        "inferred_interests": ["inferences_df"],
        "yearly_summary_metrics": [
            "wrapped_summary_df",
            "wrapped_top_tracks_df",
            "wrapped_top_artists_df",
            "sound_capsule_df",
            "sound_capsule_top_tracks_df",
            "sound_capsule_top_artists_df",
        ],
    }

    return {
        category: [table for table in names if table in available]
        for category, names in categories.items()
    }


def print_project_interpretation(table_results: list[TableResult]) -> None:
    """Print a short interpretation for a term project on music listening and special days."""
    relevant = identify_music_behavior_tables(table_results)

    print("\nMost useful datasets for a term project on music listening and special days:")
    if relevant["search_behavior"]:
        print(
            "- Search behavior tables are useful for measuring intent and curiosity on specific dates,"
            " especially when you later join them with streaming history by date."
        )
    if relevant["playlist_library_preference"]:
        print(
            "- Playlist and library tables capture preference formation and saved-music behavior,"
            " which can be compared with special days, trips, exams, holidays, or routines."
        )
    if relevant["inferred_interests"]:
        print(
            "- Inference labels provide a higher-level profile of genre, content, and interest segments,"
            " which can help explain why certain day-level patterns appear."
        )
    if relevant["yearly_summary_metrics"]:
        print(
            "- Wrapped and Sound Capsule tables are strong summary layers for annual and short-window"
            " context, and they are easy to merge later using date fields or Spotify URIs."
        )


def build_metadata_summary(table_results: list[TableResult]) -> pd.DataFrame:
    """Build a metadata summary table for all exported outputs."""
    rows = []
    for result in table_results:
        rows.append(
            {
                "source_file": result.source_file,
                "output_table_name": result.name,
                "rows": len(result.dataframe),
                "columns": len(result.dataframe.columns),
                "parse_status": result.parse_status,
            }
        )
    return pd.DataFrame(rows)


def process_account_data(input_dir: Path) -> list[TableResult]:
    """Run the end-to-end account-data parsing pipeline."""
    account_files = list_account_json_files(input_dir)
    available_by_name = {path.name: path for path in account_files}
    processed_files: set[str] = set()
    results: list[TableResult] = []

    search_queries_path = available_by_name.get("SearchQueries.json")
    if search_queries_path:
        results.extend(parse_search_queries(search_queries_path))
        processed_files.add(search_queries_path.name)

    playlist_paths = sorted(input_dir.glob("Playlist*.json"))
    playlist_paths = [path for path in playlist_paths if path.name in available_by_name]
    if playlist_paths:
        results.extend(parse_playlists(playlist_paths))
        processed_files.update(path.name for path in playlist_paths)

    your_library_path = available_by_name.get("YourLibrary.json")
    if your_library_path:
        results.extend(parse_library_tracks(your_library_path))
        processed_files.add(your_library_path.name)

    wrapped_path = available_by_name.get("Wrapped2025.json")
    if wrapped_path:
        results.extend(parse_wrapped(wrapped_path))
        processed_files.add(wrapped_path.name)

    sound_capsule_path = available_by_name.get("YourSoundCapsule.json")
    if sound_capsule_path:
        results.extend(parse_sound_capsule(sound_capsule_path))
        processed_files.add(sound_capsule_path.name)

    inferences_path = available_by_name.get("Inferences.json")
    if inferences_path:
        results.extend(parse_inferences(inferences_path))
        processed_files.add(inferences_path.name)

    follow_path = available_by_name.get("Follow.json")
    if follow_path:
        results.extend(parse_follow(follow_path))
        processed_files.add(follow_path.name)

    profile_source_paths = [
        available_by_name.get("Identity.json"),
        available_by_name.get("UserAttributes.json"),
        available_by_name.get("Identifiers.json"),
        available_by_name.get("Payments.json"),
    ]
    if any(path is not None for path in profile_source_paths):
        results.extend(
            parse_profile(
                identity_path=available_by_name.get("Identity.json"),
                user_attributes_path=available_by_name.get("UserAttributes.json"),
                identifiers_path=available_by_name.get("Identifiers.json"),
                payments_path=available_by_name.get("Payments.json"),
            )
        )
        processed_files.update(
            path.name for path in profile_source_paths if path is not None
        )

    customer_service_paths = sorted(input_dir.glob("CustomerServiceHistoryAndSurveyData*.json"))
    customer_service_paths = [
        path for path in customer_service_paths if path.name in available_by_name
    ]
    agent_gateway_path = available_by_name.get("AgentGateway.json")
    if customer_service_paths or agent_gateway_path:
        results.extend(
            parse_customer_service(
                customer_service_paths=customer_service_paths,
                agent_gateway_path=agent_gateway_path,
            )
        )
        processed_files.update(path.name for path in customer_service_paths)
        if agent_gateway_path is not None:
            processed_files.add(agent_gateway_path.name)

    for file_path in account_files:
        if file_path.name not in processed_files:
            results.extend(parse_generic_json(file_path))

    metadata_summary_df = build_metadata_summary(results)
    results.append(
        TableResult(
            name="metadata_summary_df",
            dataframe=metadata_summary_df,
            source_file="multiple",
            parse_status="generated",
        )
    )

    return results


def main() -> None:
    """Entry point for the account-data processing pipeline."""
    args = parse_arguments()
    table_results = process_account_data(args.input_dir)

    for result in table_results:
        print_dataset_summary(result.name, result.dataframe)
        save_table(result.dataframe, args.output_dir, result.name)

    print_project_interpretation(table_results)
    print(f"\nSaved output tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
