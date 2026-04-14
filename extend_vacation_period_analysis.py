from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NOTEBOOK_PATH = Path.home() / "Desktop" / "DSA210 TERM PROJECT" / "advanced_behavioral_spotify_eda.ipynb"
EXTENSION_TAG = "vacation_travel_extension"
FINAL_FINDINGS_TAG = "overall_findings_limitations_extension"


def make_markdown_cell(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": tags or []},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def make_code_cell(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def replace_data_loading_cell(nb: dict[str, Any]) -> None:
    """Make the existing data-loading cell resilient to files stored in subfolders."""
    replacement_source = r'''
def resolve_project_file(filename, preferred_path):
    """Return the preferred path when present, otherwise search inside BASE_DIR."""
    preferred_path = Path(preferred_path)
    if preferred_path.exists():
        return preferred_path

    matches = sorted(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Missing {filename}. Expected it at {preferred_path} or somewhere under {BASE_DIR}."
    )


SPOTIFY_PATH = resolve_project_file("spotify_cleaned.csv", SPOTIFY_PATH)
SPECIAL_DATES_PATH = resolve_project_file("special_dates.csv", SPECIAL_DATES_PATH)

spotify_raw = pd.read_csv(SPOTIFY_PATH)
special_raw = pd.read_csv(SPECIAL_DATES_PATH)

print("Spotify data path:", SPOTIFY_PATH)
print("Special dates path:", SPECIAL_DATES_PATH)
print("spotify_raw shape:", spotify_raw.shape)
print("special_raw shape:", special_raw.shape)

display(spotify_raw.head())
display(special_raw.head())
'''

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if (
            cell.get("cell_type") == "code"
            and "spotify_raw = pd.read_csv(SPOTIFY_PATH)" in source
            and "special_raw = pd.read_csv(SPECIAL_DATES_PATH)" in source
        ):
            cell["source"] = replacement_source.strip("\n").splitlines(keepends=True)
            return


def build_extension_cells() -> list[dict[str, Any]]:
    tag = [EXTENSION_TAG]

    return [
        make_markdown_cell(
            r'''
# Vacation and Break Period Analysis

The earlier sections focused on academic pressure: exams, deadlines, and stress windows. This extension adds a second context layer: travel vacations, school breaks, holiday blocks, and long breaks. These periods are behaviorally important because they may change routine structure, available free time, location, and listening purpose.
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## A. Vacation/Break Overview

First, the interval-based travel and break file is expanded into daily labels. This creates one row per active date, making it compatible with the daily Spotify behavior table.
''',
            tag,
        ),
        make_code_cell(
            r'''
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
SPOTIFY_PATH = BASE_DIR / "spotify_cleaned.csv"
SPECIAL_DATES_PATH = BASE_DIR / "special_dates.csv"
TRAVEL_PERIODS_PATH = BASE_DIR / "travel_and_break_periods.csv"


def vacation_resolve_project_file(filename, preferred_path):
    """Resolve a project data file from the root folder or one of its subfolders."""
    preferred_path = Path(preferred_path)
    if preferred_path.exists():
        return preferred_path

    matches = sorted(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]

    return preferred_path


def vacation_first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def vacation_to_binary(series):
    if series.dtype == bool:
        return series.astype(int)

    normalized = series.astype(str).str.lower().str.strip()
    mapped = normalized.map({
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "nan": 0,
        "none": 0,
        "": 0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype(int)


def vacation_join_unique(values):
    clean_values = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
    return "; ".join(clean_values)


def vacation_expand_periods(periods_df):
    """Expand interval-level vacation/break rows into daily labels."""
    output_columns = [
        "date",
        "is_travel_vacation",
        "is_school_break",
        "is_public_holiday_block",
        "is_long_break",
        "active_period_names",
        "active_period_types",
    ]

    if periods_df is None or periods_df.empty:
        return pd.DataFrame(columns=output_columns)

    required = {"start_date", "end_date"}
    if not required.issubset(periods_df.columns):
        missing = sorted(required - set(periods_df.columns))
        print(f"Cannot expand vacation periods. Missing columns: {missing}")
        return pd.DataFrame(columns=output_columns)

    working = periods_df.copy()
    working["start_date"] = pd.to_datetime(working["start_date"], errors="coerce")
    working["end_date"] = pd.to_datetime(working["end_date"], errors="coerce")
    working = working.dropna(subset=["start_date", "end_date"]).copy()

    binary_columns = [
        "is_travel_vacation",
        "is_school_break",
        "is_public_holiday_block",
        "is_long_break",
    ]
    for column in binary_columns:
        if column not in working.columns:
            working[column] = 0
        working[column] = vacation_to_binary(working[column])

    if "period_name" not in working.columns:
        working["period_name"] = ""
    if "period_type" not in working.columns:
        working["period_type"] = ""

    rows = []
    for _, period in working.iterrows():
        start_date = min(period["start_date"], period["end_date"])
        end_date = max(period["start_date"], period["end_date"])
        for active_date in pd.date_range(start_date, end_date, freq="D"):
            rows.append({
                "date": pd.to_datetime(active_date.date()),
                "is_travel_vacation": int(period["is_travel_vacation"]),
                "is_school_break": int(period["is_school_break"]),
                "is_public_holiday_block": int(period["is_public_holiday_block"]),
                "is_long_break": int(period["is_long_break"]),
                "active_period_names": str(period.get("period_name", "") or ""),
                "active_period_types": str(period.get("period_type", "") or ""),
            })

    if not rows:
        return pd.DataFrame(columns=output_columns)

    daily = pd.DataFrame(rows)
    daily = (
        daily.groupby("date", as_index=False)
        .agg({
            "is_travel_vacation": "max",
            "is_school_break": "max",
            "is_public_holiday_block": "max",
            "is_long_break": "max",
            "active_period_names": vacation_join_unique,
            "active_period_types": vacation_join_unique,
        })
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily[output_columns]


def vacation_build_minimal_analysis_df():
    """Reuse analysis_df when available; otherwise reconstruct daily behavior from CSV files."""
    if "analysis_df" in globals() and isinstance(globals()["analysis_df"], pd.DataFrame):
        working = globals()["analysis_df"].copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        return working.dropna(subset=["date"]).copy()

    if "daily_df" in globals() and isinstance(globals()["daily_df"], pd.DataFrame):
        working = globals()["daily_df"].copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        return working.dropna(subset=["date"]).copy()

    spotify_path = vacation_resolve_project_file("spotify_cleaned.csv", SPOTIFY_PATH)
    special_path = vacation_resolve_project_file("special_dates.csv", SPECIAL_DATES_PATH)
    if not spotify_path.exists():
        raise FileNotFoundError("spotify_cleaned.csv could not be found in the project folder or subfolders.")

    spotify_raw_local = pd.read_csv(spotify_path)
    special_raw_local = pd.read_csv(special_path) if special_path.exists() else pd.DataFrame({"date": []})

    ts_column = vacation_first_existing(spotify_raw_local, ["ts", "timestamp", "played_at", "event_timestamp"])
    ms_column = vacation_first_existing(spotify_raw_local, ["ms_played", "msPlayed", "milliseconds_played"])
    artist_column = vacation_first_existing(
        spotify_raw_local,
        ["master_metadata_album_artist_name", "artist_name", "artist", "album_artist"],
    )
    track_column = vacation_first_existing(
        spotify_raw_local,
        ["master_metadata_track_name", "track_name", "track", "episode_name"],
    )
    skipped_column = vacation_first_existing(spotify_raw_local, ["skipped", "skip", "was_skipped"])
    shuffle_column = vacation_first_existing(spotify_raw_local, ["shuffle", "shuffle_state", "is_shuffle"])

    if ts_column is None or ms_column is None:
        raise ValueError("Cannot reconstruct daily analysis data without timestamp and ms_played columns.")

    events = spotify_raw_local.copy()
    events[ts_column] = pd.to_datetime(events[ts_column], errors="coerce", utc=True)
    events = events.dropna(subset=[ts_column]).copy()
    events["date"] = pd.to_datetime(events[ts_column].dt.date)
    events[ms_column] = pd.to_numeric(events[ms_column], errors="coerce").fillna(0)

    daily = (
        events.groupby("date")
        .agg(total_ms=(ms_column, "sum"), num_streams=(ms_column, "size"))
        .reset_index()
    )
    daily["total_minutes"] = daily["total_ms"] / 60000
    daily["unique_artists"] = events.groupby("date")[artist_column].nunique().values if artist_column else np.nan
    daily["unique_tracks"] = events.groupby("date")[track_column].nunique().values if track_column else np.nan
    daily["skip_rate"] = events.groupby("date")[skipped_column].apply(lambda x: vacation_to_binary(x).mean()).values if skipped_column else np.nan
    daily["shuffle_rate"] = events.groupby("date")[shuffle_column].apply(lambda x: vacation_to_binary(x).mean()).values if shuffle_column else np.nan

    if not special_raw_local.empty and "date" in special_raw_local.columns:
        special = special_raw_local.copy()
        special["date"] = pd.to_datetime(special["date"], errors="coerce")
        daily = daily.merge(special, on="date", how="left")

    return daily


travel_path = vacation_resolve_project_file("travel_and_break_periods.csv", TRAVEL_PERIODS_PATH)
if not travel_path.exists():
    print(f"travel_and_break_periods.csv was not found at {TRAVEL_PERIODS_PATH}. Vacation analysis will be skipped.")
    travel_periods_raw = pd.DataFrame()
else:
    travel_periods_raw = pd.read_csv(travel_path)

vacation_periods_daily = vacation_expand_periods(travel_periods_raw)

vacation_binary_columns = [
    "is_travel_vacation",
    "is_school_break",
    "is_public_holiday_block",
    "is_long_break",
]
vacation_text_columns = ["active_period_names", "active_period_types"]

analysis_base_df = vacation_build_minimal_analysis_df()
analysis_base_df["date"] = pd.to_datetime(analysis_base_df["date"], errors="coerce")
analysis_base_df = analysis_base_df.dropna(subset=["date"]).copy()

analysis_with_vacation_df = analysis_base_df.drop(
    columns=[column for column in vacation_binary_columns + vacation_text_columns if column in analysis_base_df.columns],
    errors="ignore",
).merge(vacation_periods_daily, on="date", how="left")

for column in vacation_binary_columns:
    if column not in analysis_with_vacation_df.columns:
        analysis_with_vacation_df[column] = 0
    analysis_with_vacation_df[column] = pd.to_numeric(analysis_with_vacation_df[column], errors="coerce").fillna(0).astype(int)

for column in vacation_text_columns:
    if column not in analysis_with_vacation_df.columns:
        analysis_with_vacation_df[column] = ""
    analysis_with_vacation_df[column] = analysis_with_vacation_df[column].fillna("").astype(str)

print("travel_periods_raw shape:", travel_periods_raw.shape)
print("vacation_periods_daily shape:", vacation_periods_daily.shape)
print("analysis_with_vacation_df shape:", analysis_with_vacation_df.shape)

vacation_overview = pd.DataFrame({
    "label": vacation_binary_columns,
    "number_of_days": [int(vacation_periods_daily[column].sum()) if column in vacation_periods_daily else 0 for column in vacation_binary_columns],
})
display(vacation_overview)
display(vacation_periods_daily.head(10))
''',
            tag,
        ),
        make_code_cell(
            r'''
if not vacation_periods_daily.empty:
    vacation_by_year = vacation_periods_daily.copy()
    vacation_by_year["year"] = vacation_by_year["date"].dt.year
    year_summary = (
        vacation_by_year.groupby("year")[vacation_binary_columns]
        .sum()
        .reset_index()
        .sort_values("year")
    )
    display(year_summary)

    timeline_df = vacation_periods_daily.copy()
    timeline_df["active_period_count"] = timeline_df[vacation_binary_columns].sum(axis=1)

    plt.figure(figsize=(12, 4))
    plt.bar(timeline_df["date"], timeline_df["active_period_count"])
    plt.title("Timeline of Vacation, Break, and Holiday Periods")
    plt.xlabel("Date")
    plt.ylabel("Number of active period labels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("No vacation or break periods were available to visualize."))
''',
            tag,
        ),
        make_code_cell(
            r'''
if vacation_periods_daily.empty:
    display(Markdown("**Insight:** No vacation-period labels were available, so the later vacation comparisons should be interpreted as unavailable rather than negative results."))
else:
    first_period = vacation_periods_daily["date"].min().date()
    last_period = vacation_periods_daily["date"].max().date()
    travel_days = int(vacation_periods_daily["is_travel_vacation"].sum())
    break_days = int(vacation_periods_daily["is_school_break"].sum())
    holiday_days = int(vacation_periods_daily["is_public_holiday_block"].sum())
    display(Markdown(
        f"**Insight:** Vacation and break labels cover **{first_period}** to **{last_period}**. "
        f"The data include **{travel_days} travel-vacation days**, **{break_days} school-break days**, "
        f"and **{holiday_days} public-holiday-block days**. This gives the analysis a non-academic context layer, "
        "allowing routine disruption and leisure periods to be compared against exam/deadline pressure."
    ))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## B. Listening Intensity and Frequency During Vacation

The next comparison asks whether travel vacations are associated with changes in listening volume. Two dimensions are used: total listening minutes and number of streams.
''',
            tag,
        ),
        make_code_cell(
            r'''
vacation_metrics = [metric for metric in ["total_minutes", "num_streams"] if metric in analysis_with_vacation_df.columns]

if not vacation_metrics:
    display(Markdown("No listening-intensity metrics are available for vacation comparison."))
else:
    mean_table = analysis_with_vacation_df.groupby("is_travel_vacation")[vacation_metrics].mean()
    median_table = analysis_with_vacation_df.groupby("is_travel_vacation")[vacation_metrics].median()

    print("Grouped means by travel-vacation status")
    display(mean_table.round(2))
    print("Grouped medians by travel-vacation status")
    display(median_table.round(2))

    for metric in vacation_metrics:
        non_vacation = analysis_with_vacation_df.loc[analysis_with_vacation_df["is_travel_vacation"] == 0, metric].dropna()
        vacation = analysis_with_vacation_df.loc[analysis_with_vacation_df["is_travel_vacation"] == 1, metric].dropna()

        if len(non_vacation) > 0 and len(vacation) > 0:
            plt.figure(figsize=(7, 5))
            plt.boxplot([non_vacation, vacation], labels=["Non-vacation", "Travel vacation"])
            plt.title(f"{metric}: Travel Vacation vs Non-vacation Days")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 5))
            plt.hist(non_vacation, bins=25, alpha=0.6, label="Non-vacation")
            plt.hist(vacation, bins=25, alpha=0.6, label="Travel vacation")
            plt.title(f"Distribution of {metric} by Travel-Vacation Status")
            plt.xlabel(metric)
            plt.ylabel("Number of days")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"Skipping plots for {metric}: one group has no observations.")
''',
            tag,
        ),
        make_code_cell(
            r'''
if vacation_metrics and {"total_minutes", "num_streams"}.issubset(analysis_with_vacation_df.columns):
    comparison = analysis_with_vacation_df.groupby("is_travel_vacation")[["total_minutes", "num_streams"]].mean()
    if {0, 1}.issubset(set(comparison.index)):
        minutes_change = comparison.loc[1, "total_minutes"] - comparison.loc[0, "total_minutes"]
        streams_change = comparison.loc[1, "num_streams"] - comparison.loc[0, "num_streams"]
        direction_minutes = "higher" if minutes_change > 0 else "lower"
        direction_streams = "higher" if streams_change > 0 else "lower"
        display(Markdown(
            f"**Insight:** Travel-vacation days show **{direction_minutes} listening time** by about "
            f"**{abs(minutes_change):.1f} minutes per day** and **{direction_streams} stream frequency** by about "
            f"**{abs(streams_change):.1f} streams per day**. Behaviorally, this comparison separates leisure/travel routine "
            "from ordinary academic routine: more streams can suggest greater availability or travel-based listening, while fewer streams can suggest disrupted access or less need for background music."
        ))
    else:
        display(Markdown("**Insight:** Only one vacation-status group is present, so vacation and non-vacation listening cannot be compared directly."))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## C. Listening Time-of-Day During Vacation

Daily totals cannot show when listening happens. This section returns to event-level Spotify records and classifies each play into a daypart: night, morning, afternoon, or evening.
''',
            tag,
        ),
        make_code_cell(
            r'''
def vacation_get_event_level_spotify():
    """Reuse the event-level Spotify dataframe when available, otherwise reload it safely."""
    for candidate in ["spotify", "spotify_raw"]:
        if candidate in globals() and isinstance(globals()[candidate], pd.DataFrame):
            return globals()[candidate].copy()

    spotify_path = vacation_resolve_project_file("spotify_cleaned.csv", SPOTIFY_PATH)
    if spotify_path.exists():
        return pd.read_csv(spotify_path)

    return pd.DataFrame()


def vacation_assign_daypart(hour):
    if pd.isna(hour):
        return np.nan
    hour = int(hour)
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


spotify_events_vacation = vacation_get_event_level_spotify()
event_vacation_df = pd.DataFrame()

if spotify_events_vacation.empty:
    display(Markdown("Event-level Spotify data is unavailable, so time-of-day vacation analysis cannot be performed."))
else:
    ts_column_vacation = vacation_first_existing(spotify_events_vacation, ["ts", "timestamp", "played_at", "event_timestamp"])
    if ts_column_vacation is None:
        display(Markdown("No timestamp column was found in the Spotify event data, so daypart analysis cannot be performed."))
    else:
        event_vacation_df = spotify_events_vacation.copy()
        event_vacation_df[ts_column_vacation] = pd.to_datetime(event_vacation_df[ts_column_vacation], errors="coerce", utc=True)
        event_vacation_df = event_vacation_df.dropna(subset=[ts_column_vacation]).copy()
        event_vacation_df["date"] = pd.to_datetime(event_vacation_df[ts_column_vacation].dt.date)
        event_vacation_df["hour"] = event_vacation_df[ts_column_vacation].dt.hour
        event_vacation_df["daypart"] = event_vacation_df["hour"].apply(vacation_assign_daypart)
        event_vacation_df = event_vacation_df.merge(
            vacation_periods_daily[["date", "is_travel_vacation"]],
            on="date",
            how="left",
        )
        event_vacation_df["is_travel_vacation"] = event_vacation_df["is_travel_vacation"].fillna(0).astype(int)
        event_vacation_df["vacation_status"] = np.where(event_vacation_df["is_travel_vacation"] == 1, "Travel vacation", "Non-vacation")

        daypart_order = ["night", "morning", "afternoon", "evening"]
        daypart_contingency = pd.crosstab(event_vacation_df["vacation_status"], event_vacation_df["daypart"]).reindex(columns=daypart_order, fill_value=0)
        display(daypart_contingency)

        plot_daypart = daypart_contingency.T
        x_positions = np.arange(len(plot_daypart.index))
        width = 0.35

        plt.figure(figsize=(8, 5))
        if "Non-vacation" in plot_daypart.columns:
            plt.bar(x_positions - width / 2, plot_daypart["Non-vacation"], width, label="Non-vacation")
        if "Travel vacation" in plot_daypart.columns:
            plt.bar(x_positions + width / 2, plot_daypart["Travel vacation"], width, label="Travel vacation")
        plt.title("Listening Activity by Daypart and Travel-Vacation Status")
        plt.xlabel("Daypart")
        plt.ylabel("Number of listening events")
        plt.xticks(x_positions, plot_daypart.index)
        plt.legend()
        plt.tight_layout()
        plt.show()
''',
            tag,
        ),
        make_code_cell(
            r'''
if "daypart_contingency" in globals() and isinstance(daypart_contingency, pd.DataFrame) and not daypart_contingency.empty:
    vacation_daypart_share = daypart_contingency.div(daypart_contingency.sum(axis=1), axis=0).fillna(0)
    display(vacation_daypart_share.round(3))
    if "Travel vacation" in vacation_daypart_share.index:
        top_daypart = vacation_daypart_share.loc["Travel vacation"].idxmax()
        display(Markdown(
            f"**Insight:** During travel vacations, the largest share of listening events occurs in the **{top_daypart}**. "
            "This is behaviorally meaningful because vacation periods can shift routine timing: later listening may reflect looser schedules, while daytime listening may reflect travel, walking, or leisure time."
        ))
else:
    display(Markdown("**Insight:** Daypart comparison was not available because event-level timestamps or vacation labels were missing."))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## D. Genre Analysis During Vacation

Genre composition is the most direct way to test whether musical taste changes during vacations. The cleaned Spotify export may or may not contain genre columns. If no real genre column exists, the notebook uses artist and track behavior only as a proxy, not as true genre evidence.
''',
            tag,
        ),
        make_code_cell(
            r'''
import ast
import re


def vacation_extract_genre_rows(df, genre_column):
    rows = []
    for _, row in df.dropna(subset=[genre_column]).iterrows():
        raw_value = row[genre_column]
        values = []

        if isinstance(raw_value, list):
            values = raw_value
        else:
            text_value = str(raw_value).strip()
            if not text_value:
                values = []
            else:
                try:
                    parsed = ast.literal_eval(text_value)
                    if isinstance(parsed, list):
                        values = parsed
                    else:
                        values = [text_value]
                except Exception:
                    if any(separator in text_value for separator in [";", "|", ","]):
                        values = re.split(r"[;|,]", text_value)
                    else:
                        values = [text_value]

        for value in values:
            genre = str(value).strip().lower()
            if genre:
                rows.append({
                    "is_travel_vacation": int(row.get("is_travel_vacation", 0)),
                    "vacation_status": row.get("vacation_status", "Non-vacation"),
                    "genre": genre,
                })

    return pd.DataFrame(rows)


genre_columns = []
if not event_vacation_df.empty:
    genre_columns = [
        column for column in event_vacation_df.columns
        if "genre" in column.lower() and event_vacation_df[column].notna().any()
    ]

genre_column = genre_columns[0] if genre_columns else None
genre_vacation_df = pd.DataFrame()
genre_contingency = pd.DataFrame()

if genre_column is not None:
    genre_vacation_df = vacation_extract_genre_rows(event_vacation_df, genre_column)
    if not genre_vacation_df.empty:
        genre_contingency = pd.crosstab(genre_vacation_df["vacation_status"], genre_vacation_df["genre"])
        top_vacation_genres = (
            genre_vacation_df.loc[genre_vacation_df["is_travel_vacation"] == 1, "genre"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_vacation_genres.columns = ["genre", "listening_events"]
        print(f"Using genre column: {genre_column}")
        display(top_vacation_genres)

        if not top_vacation_genres.empty:
            plt.figure(figsize=(9, 5))
            plt.bar(top_vacation_genres["genre"], top_vacation_genres["listening_events"])
            plt.title("Top Genres During Travel Vacations")
            plt.xlabel("Genre")
            plt.ylabel("Listening events")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
    else:
        display(Markdown("A genre-like column exists, but it does not contain usable genre values after normalization."))
else:
    display(Markdown(
        "**Genre availability note:** No direct genre column was found in the cleaned Spotify dataset. "
        "Therefore, formal genre-level vacation testing is not possible from the current file alone. "
        "The proxy analysis below uses top artists and tracks during travel vacations, but this should not be interpreted as true genre composition."
    ))

    artist_column_vacation = vacation_first_existing(
        event_vacation_df,
        ["master_metadata_album_artist_name", "artist_name", "artist", "album_artist"],
    ) if not event_vacation_df.empty else None
    track_column_vacation = vacation_first_existing(
        event_vacation_df,
        ["master_metadata_track_name", "track_name", "track", "episode_name"],
    ) if not event_vacation_df.empty else None

    if artist_column_vacation is not None:
        top_vacation_artists = (
            event_vacation_df.loc[event_vacation_df["is_travel_vacation"] == 1, artist_column_vacation]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_vacation_artists.columns = ["artist", "listening_events"]
        print("Proxy: top artists during travel vacations")
        display(top_vacation_artists)

        if not top_vacation_artists.empty:
            plt.figure(figsize=(9, 5))
            plt.bar(top_vacation_artists["artist"], top_vacation_artists["listening_events"])
            plt.title("Proxy Analysis: Top Artists During Travel Vacations")
            plt.xlabel("Artist")
            plt.ylabel("Listening events")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

    if track_column_vacation is not None:
        top_vacation_tracks = (
            event_vacation_df.loc[event_vacation_df["is_travel_vacation"] == 1, track_column_vacation]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_vacation_tracks.columns = ["track", "listening_events"]
        print("Proxy: top tracks during travel vacations")
        display(top_vacation_tracks)
''',
            tag,
        ),
        make_code_cell(
            r'''
if genre_column is not None and not genre_vacation_df.empty:
    display(Markdown(
        "**Insight:** A real genre field is available, so vacation genre composition can be evaluated directly. "
        "Differences in top genres would suggest that travel and leisure contexts influence not only how much I listen, but what type of music I choose."
    ))
else:
    display(Markdown(
        "**Insight:** Because direct genre metadata is unavailable, the analysis should avoid claiming genre shifts. "
        "Artist and track concentration can still reveal repetition versus exploration, but it remains a behavioral proxy rather than a genre result."
    ))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
# Additional Hypothesis Tests for Vacation Periods

The following tests extend the previous hypothesis-testing section by evaluating vacation-specific behavior. The tests use alpha = 0.05 and are designed to distinguish descriptive differences from statistically detectable differences.
''',
            tag,
        ),
        make_code_cell(
            r'''
from scipy.stats import chi2_contingency, ttest_ind


VACATION_ALPHA = 0.05
vacation_hypothesis_results = []


def vacation_decision(p_value, alpha=VACATION_ALPHA):
    if pd.isna(p_value):
        return "Test not run"
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def vacation_run_ttest(df, label_column, metric_column, hypothesis_name):
    if label_column not in df.columns or metric_column not in df.columns:
        print(f"{hypothesis_name}: required columns are missing.")
        return None

    group_1 = pd.to_numeric(df.loc[df[label_column] == 1, metric_column], errors="coerce").dropna()
    group_0 = pd.to_numeric(df.loc[df[label_column] == 0, metric_column], errors="coerce").dropna()

    print(f"{hypothesis_name}")
    print(f"n vacation/status=1: {len(group_1)}")
    print(f"n non-vacation/status=0: {len(group_0)}")

    if len(group_1) < 2 or len(group_0) < 2:
        print("Not enough observations in both groups to run a two-sample t-test.")
        vacation_hypothesis_results.append({
            "hypothesis": hypothesis_name,
            "test": "Welch two-sample t-test",
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    statistic, p_value = ttest_ind(group_1, group_0, equal_var=False, nan_policy="omit")
    decision = vacation_decision(p_value)

    print(f"t-statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Decision at alpha={VACATION_ALPHA}: {decision}")

    vacation_hypothesis_results.append({
        "hypothesis": hypothesis_name,
        "test": "Welch two-sample t-test",
        "statistic": statistic,
        "p_value": p_value,
        "decision": decision,
    })
    return statistic, p_value, decision, group_1.mean(), group_0.mean()


def vacation_run_chi_square(contingency_table, hypothesis_name):
    print(hypothesis_name)
    if contingency_table is None or contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print("Not enough contingency-table structure to run chi-square test.")
        vacation_hypothesis_results.append({
            "hypothesis": hypothesis_name,
            "test": "Chi-square test of independence",
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
    decision = vacation_decision(p_value)

    print("Observed counts:")
    display(contingency_table)
    print("Expected counts:")
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    display(expected_df.round(2))
    print(f"chi-square statistic: {chi2_statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"degrees of freedom: {dof}")
    print(f"Decision at alpha={VACATION_ALPHA}: {decision}")

    vacation_hypothesis_results.append({
        "hypothesis": hypothesis_name,
        "test": "Chi-square test of independence",
        "statistic": chi2_statistic,
        "p_value": p_value,
        "decision": decision,
    })
    return chi2_statistic, p_value, dof, expected_df, decision
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## H6 - Genre Distribution During Vacation

**H0:** The distribution of music genres is independent of vacation status.

**HA:** The distribution of music genres differs between vacation and non-vacation periods.

A chi-square test of independence is appropriate when both variables are categorical: vacation status and genre. This test is only valid here if real genre metadata exists.
''',
            tag,
        ),
        make_code_cell(
            r'''
if genre_column is not None and not genre_contingency.empty:
    h6_result = vacation_run_chi_square(genre_contingency, "H6: Genre distribution during vacation")
    if h6_result is not None:
        decision = h6_result[-1]
        display(Markdown(
            f"**H6 conclusion:** {decision}. "
            "If the null is rejected, vacation status is associated with a different genre mix, suggesting that context may shape music selection. "
            "If not rejected, the available genre data does not provide strong evidence of vacation-specific genre composition."
        ))
else:
    print("H6 cannot be directly tested because no real genre column is available.")
    display(Markdown(
        "**H6 conclusion:** This hypothesis cannot be directly tested with the current cleaned Spotify dataset because no genre column is available. "
        "The artist/track proxy analysis is useful for exploration, but it is not a formal genre-distribution test."
    ))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## H7 - Listening Time and Frequency During Vacation

**H0a:** Mean daily listening time is the same during vacation and non-vacation periods.

**HAa:** Mean daily listening time differs during vacation periods.

**H0b:** Mean daily stream count is the same during vacation and non-vacation periods.

**HAb:** Mean daily stream count differs during vacation periods.

Welch two-sample t-tests are used because vacation and non-vacation groups can have unequal sizes and unequal variance.
''',
            tag,
        ),
        make_code_cell(
            r'''
h7_minutes = vacation_run_ttest(
    analysis_with_vacation_df,
    "is_travel_vacation",
    "total_minutes",
    "H7a: Daily listening time during vacation",
)

h7_streams = vacation_run_ttest(
    analysis_with_vacation_df,
    "is_travel_vacation",
    "num_streams",
    "H7b: Daily stream count during vacation",
)

if h7_minutes is not None or h7_streams is not None:
    interpretation_parts = []
    if h7_minutes is not None:
        _, p_value, decision, vacation_mean, normal_mean = h7_minutes
        direction = "higher" if vacation_mean > normal_mean else "lower"
        interpretation_parts.append(
            f"daily listening time is {direction} during travel vacations on average "
            f"({vacation_mean:.1f} vs {normal_mean:.1f} minutes), with decision: {decision}"
        )
    if h7_streams is not None:
        _, p_value, decision, vacation_mean, normal_mean = h7_streams
        direction = "higher" if vacation_mean > normal_mean else "lower"
        interpretation_parts.append(
            f"stream count is {direction} during travel vacations on average "
            f"({vacation_mean:.1f} vs {normal_mean:.1f} streams), with decision: {decision}"
        )
    display(Markdown(
        "**H7 conclusion:** " + "; ".join(interpretation_parts) + ". "
        "Behaviorally, significant differences would suggest that travel changes the intensity or frequency of listening, possibly through leisure time, transportation, or routine disruption."
    ))
else:
    display(Markdown("**H7 conclusion:** The vacation listening-time and stream-count tests could not be run because the required data were unavailable or one group was too small."))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## H8 - Listening Time-of-Day Behavior During Vacation

**H0:** The distribution of listening activity across dayparts is independent of vacation status.

**HA:** The distribution of listening activity across dayparts differs during vacation periods.

A chi-square test is appropriate because both variables are categorical: vacation status and daypart.
''',
            tag,
        ),
        make_code_cell(
            r'''
if "daypart_contingency" in globals() and isinstance(daypart_contingency, pd.DataFrame) and not daypart_contingency.empty:
    h8_result = vacation_run_chi_square(daypart_contingency, "H8: Daypart distribution during vacation")
    if h8_result is not None:
        decision = h8_result[-1]
        display(Markdown(
            f"**H8 conclusion:** {decision}. "
            "A rejection would indicate that vacation periods shift when listening happens during the day, which would support the idea that travel and breaks change routine timing. "
            "Failing to reject would suggest that the overall daily rhythm of listening remains relatively stable across vacation and non-vacation contexts."
        ))
else:
    print("H8 cannot be tested because event-level daypart counts are unavailable.")
    display(Markdown("**H8 conclusion:** The daypart hypothesis could not be tested because event-level timestamp data or vacation labels were unavailable."))
''',
            tag,
        ),
        make_markdown_cell(
            r'''
## Vacation Extension Summary

This extension adds a non-academic context layer to the project. Exams and deadlines capture pressure, while vacations and breaks capture routine disruption, leisure, and location change. Together, these contexts make the project more behavioral: the analysis now asks not only whether listening changes under stress, but also whether it changes when time becomes less academically structured.
''',
            tag,
        ),
        make_code_cell(
            r'''
if vacation_hypothesis_results:
    vacation_hypothesis_results_df = pd.DataFrame(vacation_hypothesis_results)
    display(vacation_hypothesis_results_df)
else:
    display(Markdown("No vacation hypothesis tests were run."))
''',
            tag,
        ),
    ]


def insert_extension_cells(nb: dict[str, Any], extension_cells: list[dict[str, Any]]) -> None:
    nb["cells"] = [
        cell
        for cell in nb.get("cells", [])
        if EXTENSION_TAG not in cell.get("metadata", {}).get("tags", [])
    ]

    nb["cells"].extend(extension_cells)


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    replace_data_loading_cell(nb)
    insert_extension_cells(nb, build_extension_cells())
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")

    extension_count = sum(
        EXTENSION_TAG in cell.get("metadata", {}).get("tags", [])
        for cell in nb.get("cells", [])
    )
    print(f"Updated notebook saved to: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(nb.get('cells', []))}")
    print(f"Vacation extension cells: {extension_count}")


if __name__ == "__main__":
    main()
