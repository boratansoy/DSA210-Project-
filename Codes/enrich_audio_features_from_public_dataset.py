from __future__ import annotations

import ast
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
OUTPUT_DIR = BASE_DIR / "extended_streaming_history_data"
EXTERNAL_DATASET_DIR = BASE_DIR / "external_audio_features"
EXTERNAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

UNIQUE_TRACKS_PATH = OUTPUT_DIR / "unique_tracks_for_audio_features.csv"
EVENTS_PATH = OUTPUT_DIR / "spotify_events_with_track_uri.csv"
PUBLIC_DATASET_PATH = EXTERNAL_DATASET_DIR / "public_spotify_tracks_dataset.csv"
LARGE_PUBLIC_DATASET_PATH = EXTERNAL_DATASET_DIR / "recsystum_tracks_features.csv"

ENRICHED_TRACKS_PATH = OUTPUT_DIR / "track_audio_features_enriched.csv"
ENRICHED_EVENTS_PATH = OUTPUT_DIR / "spotify_events_with_audio_features.csv"
DAILY_AUDIO_PATH = OUTPUT_DIR / "spotify_daily_audio_features.csv"
ENRICHMENT_SUMMARY_PATH = OUTPUT_DIR / "audio_feature_enrichment_summary.csv"

PUBLIC_DATASET_URL = "https://huggingface.co/datasets/Faizasb/spotify-tracks-dataset/resolve/main/dataset.csv"
LARGE_PUBLIC_DATASET_URL = "https://huggingface.co/datasets/RecSysTUM/Million_Song_Dataset/resolve/main/tracks_features.csv"

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


def download_public_dataset(url: str, output_path: Path) -> None:
    """Download the external public audio-feature dataset if missing."""
    if output_path.exists():
        print(f"Using cached public dataset: {output_path}")
        return

    print(f"Downloading public audio-feature dataset from: {url}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as target:
        target.write(response.read())
    print(f"Saved public dataset to: {output_path}")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_primary_artist(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return normalize_text(parsed[0])
    except Exception:
        pass

    # fall back to separators commonly seen in artist strings
    for separator in [",", ";", "&", " feat.", " feat ", " ft.", " ft "]:
        if separator in text.lower():
            text = re.split(separator, text, flags=re.IGNORECASE)[0]
            break
    return normalize_text(text)


def choose_best_duplicate(group: pd.DataFrame) -> pd.Series:
    """Choose a single representative row from duplicate title-artist groups."""
    sort_cols = [column for column in ["popularity", "duration_ms"] if column in group.columns]
    if sort_cols:
        ordered = group.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        return ordered.iloc[0]
    return group.iloc[0]


def load_public_audio_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare the external track-level audio-feature dataset."""
    df = pd.read_csv(path)

    required_columns = [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        *AUDIO_FEATURE_COLUMNS,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Public audio-feature dataset is missing required columns: {missing}")

    keep_cols = [column for column in required_columns if column in df.columns]
    optional_cols = [column for column in ["popularity", "duration_ms", "explicit", "track_genre"] if column in df.columns]
    df = df[keep_cols + optional_cols].copy()

    for column in AUDIO_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["track_id"] = df["track_id"].astype(str).str.strip()
    df["norm_track_name"] = df["track_name"].map(normalize_text)
    df["norm_primary_artist"] = df["artists"].map(extract_primary_artist)
    df["norm_album_name"] = df["album_name"].map(normalize_text)

    by_track_id = df.drop_duplicates(subset=["track_id"], keep="first").copy()

    grouped = (
        df.groupby(["norm_track_name", "norm_primary_artist"], dropna=False, as_index=False)
        .apply(choose_best_duplicate, include_groups=False)
        .reset_index(drop=True)
    )

    return by_track_id, grouped


def load_large_public_audio_dataset(path: Path, unique_tracks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a larger public dataset in chunks and keep only relevant rows for the user's tracks."""
    user_track_ids = {
        str(value).strip()
        for value in unique_tracks["track_id"].dropna().astype(str)
        if str(value).strip()
    }
    user_pairs = {
        (
            normalize_text(track_name),
            normalize_text(artist_name),
        )
        for track_name, artist_name in zip(
            unique_tracks["master_metadata_track_name"].fillna(""),
            unique_tracks["master_metadata_album_artist_name"].fillna(""),
        )
        if normalize_text(track_name) and normalize_text(artist_name)
    }

    usecols = [
        "track_id",
        "name",
        "album",
        "artists",
        *AUDIO_FEATURE_COLUMNS,
    ]
    optional_cols = ["duration_ms", "explicit", "year", "release_date"]

    matched_chunks: list[pd.DataFrame] = []
    chunk_iter = pd.read_csv(path, usecols=lambda c: c in usecols + optional_cols, chunksize=100_000)

    for chunk in chunk_iter:
        chunk["track_id"] = chunk["track_id"].astype(str).str.strip()
        chunk["norm_track_name"] = chunk["name"].map(normalize_text)
        chunk["norm_primary_artist"] = chunk["artists"].map(extract_primary_artist)
        pairs = list(zip(chunk["norm_track_name"], chunk["norm_primary_artist"]))
        pair_mask = [pair in user_pairs for pair in pairs]
        id_mask = chunk["track_id"].isin(user_track_ids)
        matched = chunk.loc[id_mask | pair_mask].copy()
        if not matched.empty:
            matched_chunks.append(matched)

    if not matched_chunks:
        return (
            pd.DataFrame(columns=["track_id", "track_name", "album_name", "artists"] + AUDIO_FEATURE_COLUMNS),
            pd.DataFrame(columns=["norm_track_name", "norm_primary_artist"] + AUDIO_FEATURE_COLUMNS),
        )

    df = pd.concat(matched_chunks, ignore_index=True)
    for column in AUDIO_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["track_name"] = df["name"]
    df["album_name"] = df["album"]
    df["norm_album_name"] = df["album_name"].map(normalize_text)

    by_track_id = df.drop_duplicates(subset=["track_id"], keep="first").copy()
    grouped = (
        df.groupby(["norm_track_name", "norm_primary_artist"], dropna=False, as_index=False)
        .apply(choose_best_duplicate, include_groups=False)
        .reset_index(drop=True)
    )
    return by_track_id, grouped


def enrich_track_catalog(unique_tracks: pd.DataFrame, by_track_id: pd.DataFrame, by_name_artist: pd.DataFrame) -> pd.DataFrame:
    """Enrich the unique user track catalog with audio features."""
    tracks = unique_tracks.copy()
    tracks["track_id"] = tracks["track_id"].astype(str).str.strip()
    tracks["norm_track_name"] = tracks["master_metadata_track_name"].map(normalize_text)
    tracks["norm_primary_artist"] = tracks["master_metadata_album_artist_name"].map(normalize_text)
    tracks["norm_album_name"] = tracks["master_metadata_album_album_name"].map(normalize_text)

    id_merge_cols = ["track_id"] + AUDIO_FEATURE_COLUMNS
    enriched = tracks.merge(
        by_track_id[id_merge_cols].drop_duplicates(subset=["track_id"]),
        on="track_id",
        how="left",
    )
    enriched["audio_match_source"] = np.where(
        enriched[AUDIO_FEATURE_COLUMNS].notna().any(axis=1),
        "track_id",
        "unmatched",
    )

    unmatched_mask = ~enriched[AUDIO_FEATURE_COLUMNS].notna().any(axis=1)
    if unmatched_mask.any():
        fallback = enriched.loc[unmatched_mask, ["norm_track_name", "norm_primary_artist"]].merge(
            by_name_artist[["norm_track_name", "norm_primary_artist"] + AUDIO_FEATURE_COLUMNS].drop_duplicates(
                subset=["norm_track_name", "norm_primary_artist"]
            ),
            on=["norm_track_name", "norm_primary_artist"],
            how="left",
        )

        for column in AUDIO_FEATURE_COLUMNS:
            enriched.loc[unmatched_mask, column] = fallback[column].values

        matched_by_name = enriched.loc[unmatched_mask, AUDIO_FEATURE_COLUMNS].notna().any(axis=1)
        enriched.loc[unmatched_mask, "audio_match_source"] = np.where(
            matched_by_name,
            "track_name_artist",
            "unmatched",
        )

    return enriched


def attach_audio_to_events(events_df: pd.DataFrame, enriched_tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge track-level audio features back to event-level listening rows."""
    keep_cols = ["track_id", "spotify_track_uri", "audio_match_source"] + AUDIO_FEATURE_COLUMNS
    track_audio = enriched_tracks_df[keep_cols].drop_duplicates(subset=["track_id"])
    events = events_df.copy()
    events["track_id"] = events["track_id"].astype(str).str.strip()
    enriched_events = events.merge(track_audio, on="track_id", how="left")
    return enriched_events


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_mode(values: pd.Series, weights: pd.Series) -> float:
    working = pd.DataFrame({"value": values, "weight": weights})
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working["weight"] = pd.to_numeric(working["weight"], errors="coerce")
    working = working.dropna(subset=["value", "weight"])
    working = working.loc[working["weight"] > 0]
    if working.empty:
        return np.nan
    weighted_counts = working.groupby("value")["weight"].sum()
    return weighted_counts.idxmax()


def build_daily_audio_features(enriched_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level audio features to daily rows using ms_played as weight."""
    music_events = enriched_events.loc[
        enriched_events["date"].notna()
        & enriched_events["ms_played"].notna()
        & enriched_events[AUDIO_FEATURE_COLUMNS].notna().any(axis=1)
    ].copy()

    if music_events.empty:
        return pd.DataFrame(columns=["date"] + AUDIO_FEATURE_COLUMNS)

    daily_rows = []
    continuous_cols = [
        "danceability",
        "energy",
        "valence",
        "tempo",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "loudness",
    ]

    for date_value, group in music_events.groupby("date"):
        row = {"date": date_value}
        for column in continuous_cols:
            row[column] = weighted_average(group[column], group["ms_played"])
        row["key"] = weighted_mode(group["key"], group["ms_played"])
        row["mode"] = weighted_mode(group["mode"], group["ms_played"])
        row["audio_feature_event_count"] = int(group[AUDIO_FEATURE_COLUMNS].notna().any(axis=1).sum())
        row["audio_feature_total_ms"] = pd.to_numeric(group["ms_played"], errors="coerce").fillna(0).sum()
        daily_rows.append(row)

    return pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)


def build_summary(unique_tracks: pd.DataFrame, enriched_tracks: pd.DataFrame, events_df: pd.DataFrame, enriched_events: pd.DataFrame) -> pd.DataFrame:
    """Create enrichment coverage summary."""
    track_match_rate = enriched_tracks[AUDIO_FEATURE_COLUMNS].notna().any(axis=1).mean()
    weighted_track_match_rate = (
        enriched_tracks.loc[enriched_tracks[AUDIO_FEATURE_COLUMNS].notna().any(axis=1), "stream_count"].sum()
        / enriched_tracks["stream_count"].sum()
        if len(enriched_tracks)
        else np.nan
    )
    event_match_rate = enriched_events[AUDIO_FEATURE_COLUMNS].notna().any(axis=1).mean()
    weighted_event_match_rate = (
        enriched_events.loc[enriched_events[AUDIO_FEATURE_COLUMNS].notna().any(axis=1), "ms_played"].sum()
        / enriched_events["ms_played"].sum()
        if len(enriched_events)
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "unique_tracks_total",
                "value": len(unique_tracks),
            },
            {
                "metric": "unique_tracks_with_audio_features",
                "value": int(enriched_tracks[AUDIO_FEATURE_COLUMNS].notna().any(axis=1).sum()),
            },
            {
                "metric": "unique_track_match_rate",
                "value": round(float(track_match_rate), 6),
            },
            {
                "metric": "stream_weighted_track_match_rate",
                "value": round(float(weighted_track_match_rate), 6),
            },
            {
                "metric": "event_rows_total",
                "value": len(events_df),
            },
            {
                "metric": "event_rows_with_audio_features",
                "value": int(enriched_events[AUDIO_FEATURE_COLUMNS].notna().any(axis=1).sum()),
            },
            {
                "metric": "event_match_rate",
                "value": round(float(event_match_rate), 6),
            },
            {
                "metric": "ms_weighted_event_match_rate",
                "value": round(float(weighted_event_match_rate), 6),
            },
        ]
    )
    return summary


def print_summary(summary_df: pd.DataFrame, enriched_tracks: pd.DataFrame) -> None:
    """Print concise enrichment summary."""
    print("Audio enrichment summary:")
    print(summary_df.to_string(index=False))
    print("\nMatch sources:")
    print(enriched_tracks["audio_match_source"].value_counts(dropna=False).to_string())

    matched_top = enriched_tracks.loc[
        enriched_tracks[AUDIO_FEATURE_COLUMNS].notna().any(axis=1),
        [
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
            "stream_count",
            "audio_match_source",
            "danceability",
            "energy",
            "valence",
            "tempo",
        ],
    ].head(10)
    if not matched_top.empty:
        print("\nSample matched tracks:")
        print(matched_top.to_string(index=False))


def main() -> None:
    unique_tracks = pd.read_csv(UNIQUE_TRACKS_PATH)
    events_df = pd.read_csv(EVENTS_PATH)

    by_track_frames: list[pd.DataFrame] = []
    by_pair_frames: list[pd.DataFrame] = []
    source_names: list[str] = []

    try:
        download_public_dataset(PUBLIC_DATASET_URL, PUBLIC_DATASET_PATH)
        small_by_track, small_by_pair = load_public_audio_dataset(PUBLIC_DATASET_PATH)
        by_track_frames.append(small_by_track)
        by_pair_frames.append(small_by_pair)
        source_names.append("Faizasb/spotify-tracks-dataset")
    except Exception as exc:
        print(f"Small public dataset could not be loaded: {exc}")

    try:
        download_public_dataset(LARGE_PUBLIC_DATASET_URL, LARGE_PUBLIC_DATASET_PATH)
        large_by_track, large_by_pair = load_large_public_audio_dataset(LARGE_PUBLIC_DATASET_PATH, unique_tracks)
        by_track_frames.append(large_by_track)
        by_pair_frames.append(large_by_pair)
        source_names.append("RecSysTUM/Million_Song_Dataset")
    except Exception as exc:
        print(f"Large public dataset could not be loaded: {exc}")

    if not by_track_frames or not by_pair_frames:
        raise RuntimeError("No public audio-feature dataset could be loaded.")

    by_track_id = (
        pd.concat(by_track_frames, ignore_index=True)
        .drop_duplicates(subset=["track_id"], keep="first")
        .reset_index(drop=True)
    )
    combined_pairs = pd.concat(by_pair_frames, ignore_index=True)
    by_name_artist = (
        combined_pairs.groupby(["norm_track_name", "norm_primary_artist"], dropna=False, as_index=False)
        .apply(choose_best_duplicate, include_groups=False)
        .reset_index(drop=True)
    )
    source_name = " + ".join(source_names)

    enriched_tracks = enrich_track_catalog(unique_tracks, by_track_id, by_name_artist)
    enriched_events = attach_audio_to_events(events_df, enriched_tracks)
    daily_audio = build_daily_audio_features(enriched_events)
    summary_df = build_summary(unique_tracks, enriched_tracks, events_df, enriched_events)
    summary_df = pd.concat(
        [
            pd.DataFrame([{"metric": "source_dataset", "value": source_name}]),
            summary_df,
        ],
        ignore_index=True,
    )

    enriched_tracks.to_csv(ENRICHED_TRACKS_PATH, index=False, encoding="utf-8-sig")
    enriched_events.to_csv(ENRICHED_EVENTS_PATH, index=False, encoding="utf-8-sig")
    daily_audio.to_csv(DAILY_AUDIO_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(ENRICHMENT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print_summary(summary_df, enriched_tracks)
    print(f"\nSaved enriched track table to: {ENRICHED_TRACKS_PATH}")
    print(f"Saved enriched event table to: {ENRICHED_EVENTS_PATH}")
    print(f"Saved daily audio table to: {DAILY_AUDIO_PATH}")
    print(f"Saved enrichment summary to: {ENRICHMENT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
