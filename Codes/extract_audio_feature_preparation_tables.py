from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
RAW_STREAMING_DIR = (
    Path.home()
    / "Desktop"
    / "DSA210 TERM PROJECT DATA&REPORTS"
    / "Spotify Extended Streaming History"
)
OUTPUT_DIR = BASE_DIR / "extended_streaming_history_data"

EVENTS_OUTPUT_PATH = OUTPUT_DIR / "spotify_events_with_track_uri.csv"
TRACKS_OUTPUT_PATH = OUTPUT_DIR / "unique_tracks_for_audio_features.csv"
AUDIO_TEMPLATE_OUTPUT_PATH = OUTPUT_DIR / "track_audio_features_template.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "audio_feature_preparation_summary.csv"

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "key",
    "mode",
]


def iter_audio_files(folder: Path) -> Iterable[Path]:
    """Yield only audio streaming history JSON files in sorted order."""
    for path in sorted(folder.glob("Streaming_History_Audio_*.json")):
        if path.is_file():
            yield path


def safe_read_json(path: Path) -> list[dict]:
    """Read a Spotify history JSON file and return a list of dict records."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Skipping unreadable file {path.name}: {exc}")
        return []

    if not isinstance(data, list):
        print(f"Skipping non-list JSON file {path.name}")
        return []

    rows = [row for row in data if isinstance(row, dict)]
    return rows


def parse_track_id(uri: object) -> str | None:
    """Extract the Spotify track ID from a spotify:track:... URI."""
    if uri is None:
        return None
    text = str(uri).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("spotify:track:"):
        return text.split(":")[-1]
    return None


def build_event_level_table(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-level streaming table with track URI and a per-file summary."""
    event_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    selected_columns = [
        "ts",
        "platform",
        "ms_played",
        "conn_country",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "spotify_track_uri",
        "reason_start",
        "reason_end",
        "shuffle",
        "skipped",
        "offline",
        "incognito_mode",
    ]

    for path in iter_audio_files(raw_dir):
        rows = safe_read_json(path)
        if not rows:
            summary_rows.append(
                {
                    "source_file": path.name,
                    "rows_total": 0,
                    "rows_with_track_uri": 0,
                    "uri_coverage": 0.0,
                    "parse_status": "empty_or_unreadable",
                }
            )
            continue

        frame = pd.DataFrame(rows)
        for column in selected_columns:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame = frame[selected_columns].copy()
        frame["source_file"] = path.name
        frame["track_id"] = frame["spotify_track_uri"].map(parse_track_id)
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
        frame["date"] = pd.to_datetime(frame["ts"].dt.date, errors="coerce")
        frame["ms_played"] = pd.to_numeric(frame["ms_played"], errors="coerce")
        frame["shuffle"] = frame["shuffle"].astype("string")
        frame["skipped"] = frame["skipped"].astype("string")

        rows_total = len(frame)
        rows_with_track_uri = int(frame["spotify_track_uri"].notna().sum())
        uri_coverage = rows_with_track_uri / rows_total if rows_total else 0.0

        summary_rows.append(
            {
                "source_file": path.name,
                "rows_total": rows_total,
                "rows_with_track_uri": rows_with_track_uri,
                "uri_coverage": round(uri_coverage, 6),
                "parse_status": "ok",
            }
        )
        event_frames.append(frame)

    if event_frames:
        events_df = pd.concat(event_frames, ignore_index=True)
        events_df = events_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
        events_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)
    return events_df, summary_df


def build_unique_track_tables(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a unique track catalog and a template for audio-feature enrichment."""
    if events_df.empty:
        track_catalog = pd.DataFrame(
            columns=[
                "spotify_track_uri",
                "track_id",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "master_metadata_album_album_name",
                "first_played_at",
                "last_played_at",
                "stream_count",
                "total_ms_played",
            ]
        )
        audio_template = track_catalog.copy()
        for column in AUDIO_FEATURE_COLUMNS:
            audio_template[column] = pd.NA
        return track_catalog, audio_template

    music_events = events_df.loc[
        events_df["spotify_track_uri"].notna()
        & events_df["master_metadata_track_name"].notna()
    ].copy()

    group_cols = [
        "spotify_track_uri",
        "track_id",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
    ]

    track_catalog = (
        music_events.groupby(group_cols, dropna=False)
        .agg(
            first_played_at=("ts", "min"),
            last_played_at=("ts", "max"),
            stream_count=("spotify_track_uri", "size"),
            total_ms_played=("ms_played", "sum"),
        )
        .reset_index()
        .sort_values(["stream_count", "total_ms_played"], ascending=[False, False])
        .reset_index(drop=True)
    )

    audio_template = track_catalog.copy()
    for column in AUDIO_FEATURE_COLUMNS:
        audio_template[column] = pd.NA

    return track_catalog, audio_template


def print_summary(events_df: pd.DataFrame, track_catalog: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Print concise extraction summary."""
    print(f"Event-level rows: {len(events_df):,}")
    if not events_df.empty:
        print(
            "Event date range:",
            events_df["ts"].min(),
            "to",
            events_df["ts"].max(),
        )
        print(
            "Track URI coverage:",
            f"{events_df['spotify_track_uri'].notna().mean():.2%}",
        )
    print(f"Unique tracks: {len(track_catalog):,}")
    if not track_catalog.empty:
        print("Top 5 tracks by stream count:")
        print(
            track_catalog[
                [
                    "master_metadata_track_name",
                    "master_metadata_album_artist_name",
                    "stream_count",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )
    print("Per-file coverage summary:")
    print(summary_df.to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events_df, summary_df = build_event_level_table(RAW_STREAMING_DIR)
    track_catalog, audio_template = build_unique_track_tables(events_df)

    events_df.to_csv(EVENTS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    track_catalog.to_csv(TRACKS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    audio_template.to_csv(AUDIO_TEMPLATE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print_summary(events_df, track_catalog, summary_df)
    print(f"\nSaved event-level table to: {EVENTS_OUTPUT_PATH}")
    print(f"Saved track catalog to: {TRACKS_OUTPUT_PATH}")
    print(f"Saved audio-feature template to: {AUDIO_TEMPLATE_OUTPUT_PATH}")
    print(f"Saved extraction summary to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
