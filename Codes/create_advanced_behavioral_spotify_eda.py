from pathlib import Path
import json


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
NOTEBOOK_PATH = BASE_DIR / "advanced_behavioral_spotify_eda.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


cells = [
    md(
        """
# Advanced Behavioral Spotify EDA

## Research question

**How does my music listening behavior change across different contexts, especially on special days such as exams, deadlines, and stress periods?**

This notebook treats Spotify listening history as behavioral event data. Each row is a listening event, and the goal is to transform those events into interpretable behavioral signals:

- **Intensity:** how much time I spend listening.
- **Frequency:** how many listening events happen.
- **Engagement:** whether I skip, shuffle, or end tracks early.
- **Diversity:** whether I repeat familiar music or explore more artists and tracks.
- **Context sensitivity:** whether exams, deadlines, and stress periods shift these behaviors.

The notebook is exploratory, but it is designed to produce research-level insights that can guide formal hypothesis testing later.
"""
    ),
    md(
        """
# Section 1 — Setup

This notebook uses only `pandas` and `matplotlib`. The code is written defensively because exported Spotify files may have slightly different column names.
"""
    ),
    code(
        """
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
SPOTIFY_PATH = BASE_DIR / "spotify_cleaned.csv"
SPECIAL_DATES_PATH = BASE_DIR / "special_dates.csv"

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 140)

print("Project folder:", BASE_DIR)
print("Spotify file exists:", SPOTIFY_PATH.exists())
print("Special dates file exists:", SPECIAL_DATES_PATH.exists())
"""
    ),
    md(
        """
# Section 2 — Data overview

The project combines event-level Spotify listening history with daily calendar labels. The first dataset captures what I listened to and how I interacted with it. The second dataset gives contextual labels such as exam days, deadlines, and stress periods.
"""
    ),
    code(
        """
if not SPOTIFY_PATH.exists():
    raise FileNotFoundError(f"Missing Spotify data: {SPOTIFY_PATH}")
if not SPECIAL_DATES_PATH.exists():
    raise FileNotFoundError(f"Missing special-date data: {SPECIAL_DATES_PATH}")

spotify_raw = pd.read_csv(SPOTIFY_PATH)
special_raw = pd.read_csv(SPECIAL_DATES_PATH)

print("spotify_raw shape:", spotify_raw.shape)
print("special_raw shape:", special_raw.shape)

display(spotify_raw.head())
display(special_raw.head())
"""
    ),
    code(
        """
print("Spotify columns")
display(pd.DataFrame({"column": spotify_raw.columns}))

print("Special-date columns")
display(pd.DataFrame({"column": special_raw.columns}))
"""
    ),
    code(
        """
def first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


ts_col = first_existing(spotify_raw, ["ts", "timestamp", "played_at", "event_timestamp"])
ms_col = first_existing(spotify_raw, ["ms_played", "msPlayed", "milliseconds_played"])
artist_col = first_existing(spotify_raw, ["master_metadata_album_artist_name", "artist_name", "artist", "album_artist"])
track_col = first_existing(spotify_raw, ["master_metadata_track_name", "track_name", "track"])
album_col = first_existing(spotify_raw, ["master_metadata_album_album_name", "album_name", "album"])
reason_start_col = first_existing(spotify_raw, ["reason_start", "message_reason_start"])
reason_end_col = first_existing(spotify_raw, ["reason_end", "message_reason_end"])
shuffle_col = first_existing(spotify_raw, ["shuffle", "is_shuffle", "message_shuffle"])
skipped_col = first_existing(spotify_raw, ["skipped", "skip", "was_skipped"])

detected_columns = pd.DataFrame(
    {
        "semantic_role": [
            "timestamp", "duration_ms", "artist", "track", "album",
            "reason_start", "reason_end", "shuffle", "skipped"
        ],
        "detected_column": [
            ts_col, ms_col, artist_col, track_col, album_col,
            reason_start_col, reason_end_col, shuffle_col, skipped_col
        ],
    }
)
display(detected_columns)

if ts_col is None:
    raise ValueError("No timestamp column found. A timestamp is required for this project.")
if ms_col is None:
    raise ValueError("No listening-duration column found. ms_played or equivalent is required.")
"""
    ),
    md(
        """
**Insight:** The available fields support a behavioral study rather than just a usage summary. Duration and stream counts measure volume, skip and end reasons measure engagement, and artist/track counts measure diversity.
"""
    ),
    md(
        """
# Core processing — event data to daily behavior

The Spotify dataset is event-based, while calendar context is daily. I therefore build a daily behavioral dataset where each row represents one day.
"""
    ),
    code(
        """
def to_binary(series):
    if series.dtype == bool:
        return series.astype(int)
    normalized = series.astype(str).str.lower().str.strip()
    mapped = normalized.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "yes": 1, "no": 0, "nan": 0, "none": 0
    })
    return mapped.fillna(pd.to_numeric(series, errors="coerce")).fillna(0)


spotify = spotify_raw.copy()
special = special_raw.copy()

spotify[ts_col] = pd.to_datetime(spotify[ts_col], errors="coerce", utc=True)
spotify = spotify.dropna(subset=[ts_col]).copy()
spotify["date"] = pd.to_datetime(spotify[ts_col].dt.date)
spotify["hour"] = spotify[ts_col].dt.hour
spotify[ms_col] = pd.to_numeric(spotify[ms_col], errors="coerce").fillna(0)
spotify["minutes_played"] = spotify[ms_col] / 60000

special["date"] = pd.to_datetime(special["date"], errors="coerce")
special = special.dropna(subset=["date"]).copy()

label_cols = ["is_exam", "is_deadline", "is_stress_period", "is_academic_event", "is_personal"]
for column in label_cols:
    if column not in special.columns:
        special[column] = 0
    special[column] = pd.to_numeric(special[column], errors="coerce").fillna(0).astype(int)

for column in ["source_events", "categories"]:
    if column not in special.columns:
        special[column] = ""
    special[column] = special[column].fillna("").astype(str)

if skipped_col is not None:
    spotify["skipped_numeric"] = to_binary(spotify[skipped_col])
else:
    spotify["skipped_numeric"] = pd.NA

if shuffle_col is not None:
    spotify["shuffle_numeric"] = to_binary(spotify[shuffle_col])
else:
    spotify["shuffle_numeric"] = pd.NA

agg_dict = {ms_col: "sum"}
if artist_col is not None:
    agg_dict[artist_col] = pd.Series.nunique
if track_col is not None:
    agg_dict[track_col] = pd.Series.nunique

daily_df = spotify.groupby("date").agg(agg_dict).reset_index()
daily_df = daily_df.rename(columns={ms_col: "total_ms"})
daily_df["total_minutes"] = daily_df["total_ms"] / 60000
daily_df["num_streams"] = spotify.groupby("date").size().reindex(daily_df["date"]).values

if artist_col is not None:
    daily_df = daily_df.rename(columns={artist_col: "unique_artists"})
else:
    daily_df["unique_artists"] = pd.NA

if track_col is not None:
    daily_df = daily_df.rename(columns={track_col: "unique_tracks"})
else:
    daily_df["unique_tracks"] = pd.NA

if skipped_col is not None:
    daily_df = daily_df.merge(
        spotify.groupby("date")["skipped_numeric"].mean().reset_index(name="skip_rate"),
        on="date",
        how="left",
    )
else:
    daily_df["skip_rate"] = pd.NA

if shuffle_col is not None:
    daily_df = daily_df.merge(
        spotify.groupby("date")["shuffle_numeric"].mean().reset_index(name="shuffle_rate"),
        on="date",
        how="left",
    )
else:
    daily_df["shuffle_rate"] = pd.NA

daily_df["avg_stream_length"] = daily_df["total_minutes"] / daily_df["num_streams"].replace(0, pd.NA)
daily_df["delta_total_minutes"] = daily_df["total_minutes"].diff()
daily_df["rolling_7"] = daily_df["total_minutes"].rolling(7, min_periods=3).mean()

analysis_df = daily_df.merge(special, on="date", how="left")
for column in label_cols:
    analysis_df[column] = pd.to_numeric(analysis_df[column], errors="coerce").fillna(0).astype(int)
for column in ["source_events", "categories"]:
    analysis_df[column] = analysis_df[column].fillna("").astype(str)

analysis_df["any_special_day"] = analysis_df[label_cols].max(axis=1)
analysis_df["weekday"] = analysis_df["date"].dt.day_name()
analysis_df["weekday_num"] = analysis_df["date"].dt.dayofweek
analysis_df["month"] = analysis_df["date"].dt.month
analysis_df["year"] = analysis_df["date"].dt.year

analysis_df = analysis_df.sort_values("date").reset_index(drop=True)

print("Daily behavioral dataset shape:", analysis_df.shape)
display(analysis_df.head())
"""
    ),
    code(
        """
behavior_metrics = [
    "total_minutes", "num_streams", "avg_stream_length",
    "unique_artists", "unique_tracks", "skip_rate", "shuffle_rate"
]
behavior_metrics = [column for column in behavior_metrics if column in analysis_df.columns]

display(analysis_df[behavior_metrics].describe().T.round(2))
display(analysis_df[label_cols + ["any_special_day"]].sum().to_frame("number_of_days"))
"""
    ),
    md(
        """
**Insight:** The daily table creates a behavioral fingerprint for each day. It allows the project to compare intensity, frequency, engagement, and diversity across normal and special contexts.
"""
    ),
    md(
        """
# Section 2 — Baseline behavior: Who am I as a listener?

Before studying exams or deadlines, I first establish my baseline listening identity. This section asks: Do I listen consistently, in bursts, with diverse artists, or with repeated routines?
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["total_minutes"])
plt.title("Baseline Listening Intensity: Daily Minutes")
plt.xlabel("Date")
plt.ylabel("Total minutes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
peak_day = analysis_df.loc[analysis_df["total_minutes"].idxmax()]
median_minutes = analysis_df["total_minutes"].median()
mean_minutes = analysis_df["total_minutes"].mean()
display(Markdown(
    f"**Insight:** My baseline listening is uneven rather than flat. The average day has **{mean_minutes:.1f} minutes**, while the median is **{median_minutes:.1f} minutes**, showing whether a few high-volume days pull the average upward. The peak day, **{peak_day['date'].date()}**, reached **{peak_day['total_minutes']:.1f} minutes**, suggesting that some days represent exceptional listening contexts rather than normal routine."
))
"""
    ),
    code(
        """
plt.figure(figsize=(8, 5))
plt.hist(analysis_df["total_minutes"].dropna(), bins=30)
plt.title("Distribution of Daily Listening Time")
plt.xlabel("Total minutes")
plt.ylabel("Number of days")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
skew_direction = "right-skewed" if analysis_df["total_minutes"].mean() > analysis_df["total_minutes"].median() else "not strongly right-skewed"
display(Markdown(
    f"**Insight:** The listening-time distribution is **{skew_direction}**. This means averages alone may overstate typical behavior. For special-day analysis, medians and boxplots are necessary because stress-related effects may appear as changes in variability rather than simple mean shifts."
))
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["num_streams"])
plt.title("Baseline Listening Frequency: Daily Stream Count")
plt.xlabel("Date")
plt.ylabel("Number of streams")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
stream_peak = analysis_df.loc[analysis_df["num_streams"].idxmax()]
display(Markdown(
    f"**Insight:** Stream frequency captures interaction with Spotify, not just time spent. The highest-frequency day was **{stream_peak['date'].date()}** with **{int(stream_peak['num_streams'])} streams**. If frequency spikes without similar listening-time growth, that suggests more fragmented or skip-heavy behavior."
))
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["unique_artists"])
plt.title("Baseline Exploration: Unique Artists per Day")
plt.xlabel("Date")
plt.ylabel("Unique artists")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
diversity_median = analysis_df["unique_artists"].median()
diversity_peak = analysis_df.loc[analysis_df["unique_artists"].idxmax()]
display(Markdown(
    f"**Insight:** Artist diversity shows exploration versus repetition. The median day has **{diversity_median:.0f} unique artists**, while the most diverse day reached **{diversity_peak['unique_artists']:.0f} artists**. This dimension is behaviorally important because stress might push me toward familiar comfort music or toward broader exploration for mood regulation."
))
"""
    ),
    md(
        """
# Section 3 — Listening intensity vs frequency

Listening for a long time is not the same as listening frequently. This section separates deep, sustained listening from fragmented, high-interaction listening.
"""
    ),
    code(
        """
plt.figure(figsize=(8, 5))
plt.scatter(analysis_df["num_streams"], analysis_df["total_minutes"], alpha=0.6)
plt.title("Listening Intensity vs Frequency")
plt.xlabel("Number of streams")
plt.ylabel("Total minutes")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
correlation = analysis_df[["num_streams", "total_minutes"]].corr().iloc[0, 1]
display(Markdown(
    f"**Insight:** The correlation between stream count and listening time is **{correlation:.2f}**. A high value suggests that frequent listening generally means longer listening, while gaps from the trend may reveal fragmented days with many short plays or focused days with fewer longer plays."
))
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["avg_stream_length"])
plt.title("Average Stream Length Over Time")
plt.xlabel("Date")
plt.ylabel("Average minutes per stream")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
avg_length_median = analysis_df["avg_stream_length"].median()
display(Markdown(
    f"**Insight:** Average stream length is a proxy for listening continuity. A median of **{avg_length_median:.2f} minutes per stream** indicates how complete or fragmented my typical listening is. Shorter average stream lengths during special contexts may suggest distraction, dissatisfaction, or rapid switching."
))
"""
    ),
    md(
        """
# Section 4 — Engagement analysis

Engagement is measured through skipping, shuffling, and track-ending behavior. These variables are behavioral signals: skipping can indicate dissatisfaction or distraction, while shuffle can indicate openness to randomness instead of controlled selection.
"""
    ),
    code(
        """
if analysis_df["skip_rate"].notna().any():
    plt.figure(figsize=(8, 5))
    plt.hist(analysis_df["skip_rate"].dropna(), bins=20)
    plt.title("Distribution of Daily Skip Rate")
    plt.xlabel("Skip rate")
    plt.ylabel("Number of days")
    plt.tight_layout()
    plt.show()
else:
    print("Skip-rate data is unavailable.")
"""
    ),
    code(
        """
if analysis_df["skip_rate"].notna().any():
    display(Markdown(
        f"**Insight:** The average daily skip rate is **{analysis_df['skip_rate'].mean():.2f}**. Higher skip-rate days may reflect lower attention, less music satisfaction, or more active searching for the right mood."
    ))
else:
    display(Markdown("**Insight:** Skip behavior is unavailable, so engagement must be inferred from stream frequency and average stream length."))
"""
    ),
    code(
        """
if analysis_df["shuffle_rate"].notna().any():
    plt.figure(figsize=(8, 5))
    plt.hist(analysis_df["shuffle_rate"].dropna(), bins=20)
    plt.title("Distribution of Daily Shuffle Rate")
    plt.xlabel("Shuffle rate")
    plt.ylabel("Number of days")
    plt.tight_layout()
    plt.show()
else:
    print("Shuffle-rate data is unavailable.")
"""
    ),
    code(
        """
if analysis_df["shuffle_rate"].notna().any():
    display(Markdown(
        f"**Insight:** The average shuffle rate is **{analysis_df['shuffle_rate'].mean():.2f}**. Higher shuffle use can indicate passive or exploratory listening, while lower shuffle use may indicate controlled, intentional song choice."
    ))
"""
    ),
    code(
        """
if reason_end_col is not None:
    reason_end_counts = spotify[reason_end_col].fillna("missing").value_counts().head(10)
    display(reason_end_counts.to_frame("count"))
    plt.figure(figsize=(9, 5))
    plt.bar(reason_end_counts.index.astype(str), reason_end_counts.values)
    plt.title("Top Track Ending Reasons")
    plt.xlabel("Reason end")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("reason_end column is unavailable.")
"""
    ),
    code(
        """
if reason_end_col is not None:
    dominant_reason = reason_end_counts.index[0]
    display(Markdown(
        f"**Insight:** The most common ending reason is **{dominant_reason}**. Ending reasons help distinguish passive completion from active interruption. This is useful for stress analysis because stressed listening may involve more active control, skipping, or interruption."
    ))
"""
    ),
    md(
        """
# Section 5 — Artist and track behavior

This section studies repetition versus exploration. Top artists and tracks show the stable core of my listening identity, while diversity over time shows how flexible that identity is across contexts.
"""
    ),
    code(
        """
if artist_col is not None:
    top_artists = spotify[artist_col].dropna().value_counts().head(10)
    display(top_artists.to_frame("streams"))
    plt.figure(figsize=(10, 5))
    plt.bar(top_artists.index.astype(str), top_artists.values)
    plt.title("Top 10 Artists by Stream Count")
    plt.xlabel("Artist")
    plt.ylabel("Streams")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("Artist column unavailable.")
"""
    ),
    code(
        """
if artist_col is not None:
    top_artist_share = top_artists.iloc[0] / len(spotify) * 100
    display(Markdown(
        f"**Insight:** The top artist accounts for **{top_artist_share:.1f}%** of all streams. A concentrated top-artist profile suggests comfort repetition, while a flatter profile would suggest exploratory behavior."
    ))
"""
    ),
    code(
        """
if track_col is not None:
    top_tracks = spotify[track_col].dropna().value_counts().head(10)
    display(top_tracks.to_frame("streams"))
    plt.figure(figsize=(10, 5))
    plt.bar(top_tracks.index.astype(str), top_tracks.values)
    plt.title("Top 10 Tracks by Stream Count")
    plt.xlabel("Track")
    plt.ylabel("Streams")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("Track column unavailable.")
"""
    ),
    code(
        """
if track_col is not None:
    display(Markdown(
        "**Insight:** Top tracks represent repeated emotional or situational anchors. If these tracks become more dominant during stress periods, that would suggest music is being used as a stabilizing routine rather than just background entertainment."
    ))
"""
    ),
    md(
        """
# Section 6 — Time-based behavior

Time patterns reveal routine. Hourly, weekday, and monthly listening patterns show when music fits into daily life and academic rhythms.
"""
    ),
    code(
        """
hourly = spotify.groupby("hour")["minutes_played"].sum().reset_index()
plt.figure(figsize=(9, 5))
plt.bar(hourly["hour"].astype(str), hourly["minutes_played"])
plt.title("Listening by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Total minutes")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
peak_hour = hourly.loc[hourly["minutes_played"].idxmax(), "hour"]
display(Markdown(
    f"**Insight:** The peak listening hour is **{int(peak_hour)}:00**. This indicates the part of the day when music is most embedded in routine, such as commuting, studying, relaxing, or late-night decompression."
))
"""
    ),
    code(
        """
weekday_summary = (
    analysis_df.groupby(["weekday_num", "weekday"])[["total_minutes", "num_streams", "unique_artists"]]
    .mean()
    .reset_index()
    .sort_values("weekday_num")
)
display(weekday_summary.round(2))

plt.figure(figsize=(8, 5))
plt.bar(weekday_summary["weekday"], weekday_summary["total_minutes"])
plt.title("Average Listening Time by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average minutes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
weekday_high = weekday_summary.loc[weekday_summary["total_minutes"].idxmax()]
weekday_low = weekday_summary.loc[weekday_summary["total_minutes"].idxmin()]
display(Markdown(
    f"**Insight:** Listening is highest on **{weekday_high['weekday']}** and lowest on **{weekday_low['weekday']}**. This baseline matters because special days may appear abnormal partly because they interrupt or intensify existing weekday routines."
))
"""
    ),
    code(
        """
monthly_summary = analysis_df.groupby("month")[["total_minutes", "num_streams", "unique_artists"]].mean().reset_index()
display(monthly_summary.round(2))

plt.figure(figsize=(8, 5))
plt.bar(monthly_summary["month"].astype(str), monthly_summary["total_minutes"])
plt.title("Average Listening Time by Month")
plt.xlabel("Month")
plt.ylabel("Average minutes")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
month_high = monthly_summary.loc[monthly_summary["total_minutes"].idxmax()]
display(Markdown(
    f"**Insight:** Month **{int(month_high['month'])}** has the highest average listening time. Monthly variation may reflect semester rhythm, holidays, exams, or lifestyle changes rather than isolated daily effects."
))
"""
    ),
    md(
        """
# Section 7 — Special day impact

This is the core context analysis. For exams, deadlines, and stress periods, I compare intensity, frequency, and diversity using tables, boxplots, and histograms.
"""
    ),
    code(
        """
core_metrics = ["total_minutes", "num_streams", "unique_artists"]


def comparison_table(df, label):
    summary = df.groupby(label)[core_metrics].agg(["count", "mean", "median", "std"]).round(2)
    return summary


def effect_size_table(df, label):
    rows = []
    for metric in core_metrics:
        normal = df.loc[df[label] == 0, metric].dropna()
        special_group = df.loc[df[label] == 1, metric].dropna()
        if normal.empty or special_group.empty:
            continue
        normal_mean = normal.mean()
        special_mean = special_group.mean()
        rows.append({
            "metric": metric,
            "normal_mean": normal_mean,
            "label_mean": special_mean,
            "mean_difference": special_mean - normal_mean,
            "percent_change": ((special_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else pd.NA,
            "normal_median": normal.median(),
            "label_median": special_group.median(),
            "normal_std": normal.std(),
            "label_std": special_group.std(),
        })
    return pd.DataFrame(rows).round(2)


def boxplot_metric(df, label, metric):
    normal = df.loc[df[label] == 0, metric].dropna()
    special_group = df.loc[df[label] == 1, metric].dropna()
    if normal.empty or special_group.empty:
        print(f"Skipping {metric} boxplot for {label}; one group has no data.")
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot([normal, special_group], labels=[f"{label}=0", f"{label}=1"])
    plt.title(f"{metric} by {label}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


def histogram_metric(df, label, metric):
    normal = df.loc[df[label] == 0, metric].dropna()
    special_group = df.loc[df[label] == 1, metric].dropna()
    if normal.empty or special_group.empty:
        print(f"Skipping {metric} histogram for {label}; one group has no data.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(normal, bins=25, alpha=0.6, label=f"{label}=0")
    plt.hist(special_group, bins=25, alpha=0.6, label=f"{label}=1")
    plt.title(f"{metric} distribution by {label}")
    plt.xlabel(metric)
    plt.ylabel("Days")
    plt.legend()
    plt.tight_layout()
    plt.show()


def contextual_insight(df, label, context_name):
    effects = effect_size_table(df, label)
    if effects.empty:
        display(Markdown(f"**Insight:** {context_name} cannot be interpreted because one group has no observations."))
        return
    indexed = effects.set_index("metric")
    minutes_change = indexed.loc["total_minutes", "percent_change"]
    streams_change = indexed.loc["num_streams", "percent_change"]
    diversity_change = indexed.loc["unique_artists", "percent_change"]
    variability = "more variable" if indexed.loc["total_minutes", "label_std"] > indexed.loc["total_minutes", "normal_std"] else "less variable"
    direction = "increases" if minutes_change > 0 else "decreases"
    display(Markdown(
        f"**Insight for {context_name}:** Listening time **{direction}** by about **{minutes_change:.1f}%**, stream frequency changes by **{streams_change:.1f}%**, and artist diversity changes by **{diversity_change:.1f}%**. The labeled-day listening distribution is **{variability}**, which is important because stress may show up as instability rather than a simple average shift."
    ))
"""
    ),
    md(
        """
## 7A — Exam impact

Exam days are acute pressure points. I test whether listening becomes reduced, intensified, or more variable on these days.
"""
    ),
    code(
        """
display(comparison_table(analysis_df, "is_exam"))
display(effect_size_table(analysis_df, "is_exam"))
for metric in core_metrics:
    boxplot_metric(analysis_df, "is_exam", metric)
    histogram_metric(analysis_df, "is_exam", metric)
contextual_insight(analysis_df, "is_exam", "exam days")
"""
    ),
    md(
        """
## 7B — Deadline impact

Deadline days may reflect focus pressure. Music could become background support for work or disappear because the day is too disrupted.
"""
    ),
    code(
        """
display(comparison_table(analysis_df, "is_deadline"))
display(effect_size_table(analysis_df, "is_deadline"))
for metric in core_metrics:
    boxplot_metric(analysis_df, "is_deadline", metric)
    histogram_metric(analysis_df, "is_deadline", metric)
contextual_insight(analysis_df, "is_deadline", "deadline days")
"""
    ),
    md(
        """
## 7C — Stress-period impact

Stress periods are anticipatory windows before major events. They may be more behaviorally revealing than the event day itself.
"""
    ),
    code(
        """
display(comparison_table(analysis_df, "is_stress_period"))
display(effect_size_table(analysis_df, "is_stress_period"))
for metric in core_metrics:
    boxplot_metric(analysis_df, "is_stress_period", metric)
    histogram_metric(analysis_df, "is_stress_period", metric)
contextual_insight(analysis_df, "is_stress_period", "stress periods")
"""
    ),
    md(
        """
# Section 8 — Behavior shift analysis

This section focuses on changes rather than levels. A day may not be extreme in absolute listening time but may still represent a sharp behavioral shift from the previous day.
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["delta_total_minutes"])
plt.title("Daily Change in Listening Time")
plt.xlabel("Date")
plt.ylabel("Change in total minutes from previous listening day")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
largest_increase = analysis_df.loc[analysis_df["delta_total_minutes"].idxmax()]
largest_drop = analysis_df.loc[analysis_df["delta_total_minutes"].idxmin()]
display(Markdown(
    f"**Insight:** The largest increase happened on **{largest_increase['date'].date()}**, while the sharpest drop happened on **{largest_drop['date'].date()}**. Sudden changes may reflect routine disruption, mood changes, travel, deadlines, or recovery after high-pressure periods."
))
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
plt.plot(analysis_df["date"], analysis_df["total_minutes"], alpha=0.45, label="daily minutes")
plt.plot(analysis_df["date"], analysis_df["rolling_7"], label="7-day rolling mean")
plt.title("Daily Listening Time with 7-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Total minutes")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
display(Markdown(
    "**Insight:** The rolling mean smooths out isolated spikes and reveals broader behavioral phases. If stress periods align with rolling increases or decreases, that suggests a sustained context effect rather than a one-day anomaly."
))
"""
    ),
    md(
        """
# Section 9 — Event-centered exam analysis

Instead of only comparing exam days to normal days, this analysis aligns listening around exam events from 3 days before to 3 days after. This can reveal anticipation and recovery patterns.
"""
    ),
    code(
        """
exam_dates = sorted(analysis_df.loc[analysis_df["is_exam"] == 1, "date"].dropna().unique())
window = 3
event_rows = []

for exam_date in exam_dates:
    exam_date = pd.Timestamp(exam_date)
    for relative_day in range(-window, window + 1):
        target_date = exam_date + pd.Timedelta(days=relative_day)
        row = analysis_df[analysis_df["date"] == target_date]
        if row.empty:
            event_rows.append({
                "exam_date": exam_date,
                "relative_day": relative_day,
                "total_minutes": 0,
                "num_streams": 0,
                "unique_artists": 0,
            })
        else:
            event_rows.append({
                "exam_date": exam_date,
                "relative_day": relative_day,
                "total_minutes": row["total_minutes"].iloc[0],
                "num_streams": row["num_streams"].iloc[0],
                "unique_artists": row["unique_artists"].iloc[0],
            })

exam_window_df = pd.DataFrame(event_rows)
print("Exam events used:", len(exam_dates))
display(exam_window_df.head())
"""
    ),
    code(
        """
if not exam_window_df.empty:
    exam_curve = exam_window_df.groupby("relative_day")[core_metrics].mean().reset_index()
    display(exam_curve.round(2))
else:
    exam_curve = pd.DataFrame()
    print("No exam-centered data available.")
"""
    ),
    code(
        """
if not exam_curve.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(exam_curve["relative_day"], exam_curve["total_minutes"])
    plt.axvline(0, linestyle="--")
    plt.title("Average Listening Time Around Exam Days")
    plt.xlabel("Days relative to exam")
    plt.ylabel("Average total minutes")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """
if not exam_curve.empty:
    before = exam_window_df[exam_window_df["relative_day"].between(-3, -1)]["total_minutes"].mean()
    exam_day = exam_window_df[exam_window_df["relative_day"] == 0]["total_minutes"].mean()
    after = exam_window_df[exam_window_df["relative_day"].between(1, 3)]["total_minutes"].mean()
    display(Markdown(
        f"**Insight:** Average listening is **{before:.1f} minutes** before exams, **{exam_day:.1f} minutes** on exam days, and **{after:.1f} minutes** after exams. Higher pre-exam listening could indicate coping or study background music, while higher post-exam listening could indicate recovery and decompression."
    ))
"""
    ),
    md(
        """
# Section 10 — Extreme behavior

Extreme days are not just outliers to remove. In behavioral data, extremes can be meaningful episodes of stress, recovery, travel, social activity, or unusual routine.
"""
    ),
    code(
        """
extreme_cols = [
    "date", "total_minutes", "num_streams", "unique_artists",
    "is_exam", "is_deadline", "is_stress_period", "source_events", "categories"
]
highest_days = analysis_df.sort_values("total_minutes", ascending=False).head(10)[extreme_cols]
lowest_days = analysis_df.sort_values("total_minutes", ascending=True).head(10)[extreme_cols]

print("Highest listening days")
display(highest_days)

print("Lowest listening days")
display(lowest_days)
"""
    ),
    code(
        """
plt.figure(figsize=(10, 5))
plt.bar(highest_days["date"].dt.strftime("%Y-%m-%d"), highest_days["total_minutes"])
plt.title("Top 10 Highest Listening Days")
plt.xlabel("Date")
plt.ylabel("Total minutes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
plt.figure(figsize=(10, 5))
plt.bar(lowest_days["date"].dt.strftime("%Y-%m-%d"), lowest_days["total_minutes"])
plt.title("Top 10 Lowest Listening Days")
plt.xlabel("Date")
plt.ylabel("Total minutes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
highest_overlap = highest_days[["is_exam", "is_deadline", "is_stress_period"]].sum().sum()
lowest_overlap = lowest_days[["is_exam", "is_deadline", "is_stress_period"]].sum().sum()
display(Markdown(
    f"**Insight:** The highest-listening days contain **{int(highest_overlap)}** special-context labels, while the lowest-listening days contain **{int(lowest_overlap)}**. This helps distinguish ordinary outliers from context-linked behavioral episodes."
))
"""
    ),
    md(
        """
# Section 11 — Combined conditions

Special contexts can overlap. Exam days inside stress periods may behave differently from stress days without exams. This section checks compound effects.
"""
    ),
    code(
        """
combined = analysis_df.copy()
combined["exam_and_stress"] = ((combined["is_exam"] == 1) & (combined["is_stress_period"] == 1)).astype(int)
combined["stress_no_exam"] = ((combined["is_exam"] == 0) & (combined["is_stress_period"] == 1)).astype(int)
combined["deadline_and_stress"] = ((combined["is_deadline"] == 1) & (combined["is_stress_period"] == 1)).astype(int)

compound_summary = pd.DataFrame({
    "condition": ["exam_and_stress", "stress_no_exam", "deadline_and_stress"],
    "days": [
        combined["exam_and_stress"].sum(),
        combined["stress_no_exam"].sum(),
        combined["deadline_and_stress"].sum(),
    ],
    "mean_total_minutes": [
        combined.loc[combined["exam_and_stress"] == 1, "total_minutes"].mean(),
        combined.loc[combined["stress_no_exam"] == 1, "total_minutes"].mean(),
        combined.loc[combined["deadline_and_stress"] == 1, "total_minutes"].mean(),
    ],
    "mean_num_streams": [
        combined.loc[combined["exam_and_stress"] == 1, "num_streams"].mean(),
        combined.loc[combined["stress_no_exam"] == 1, "num_streams"].mean(),
        combined.loc[combined["deadline_and_stress"] == 1, "num_streams"].mean(),
    ],
    "mean_unique_artists": [
        combined.loc[combined["exam_and_stress"] == 1, "unique_artists"].mean(),
        combined.loc[combined["stress_no_exam"] == 1, "unique_artists"].mean(),
        combined.loc[combined["deadline_and_stress"] == 1, "unique_artists"].mean(),
    ],
})
display(compound_summary.round(2))
"""
    ),
    code(
        """
plot_compound = compound_summary.dropna(subset=["mean_total_minutes"])
plt.figure(figsize=(8, 5))
plt.bar(plot_compound["condition"], plot_compound["mean_total_minutes"])
plt.title("Mean Listening Time Under Compound Conditions")
plt.xlabel("Condition")
plt.ylabel("Mean total minutes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
display(Markdown(
    "**Insight:** Compound contexts help separate acute events from background stress. If stress-without-exam differs from exam-related stress, then the behavioral mechanism may not be the event itself but the anticipation and workload surrounding it."
))
"""
    ),
    md(
        """
# Section 12 — Key findings

This section summarizes the strongest patterns from the notebook in behavioral terms.
"""
    ),
    code(
        """
finding_lines = []
for label in ["is_exam", "is_deadline", "is_stress_period"]:
    effects = effect_size_table(analysis_df, label)
    for _, row in effects.iterrows():
        finding_lines.append(
            f"- **{label}** changes **{row['metric']}** by approximately **{row['percent_change']:.1f}%**."
        )

display(Markdown(
    "## Behavioral findings\\n"
    + "\\n".join(finding_lines[:9])
    + "\\n\\nOverall, the most important question is not only whether listening increases or decreases, but which dimension changes: time investment, interaction frequency, engagement control, or diversity."
))
"""
    ),
    md(
        """
# Section 13 — Hypothesis generation

The EDA suggests the following hypotheses for formal testing:

- **H1:** Exam days reduce daily listening time compared with non-exam days.
- **H2:** Stress periods increase skipping, suggesting lower satisfaction or more fragmented attention.
- **H3:** Stress periods reduce artist diversity, suggesting comfort-based repetitive listening.
- **H4:** Deadlines change listening frequency, either because music supports focus or because work disrupts normal listening.
- **H5:** Listening changes before and after exams, indicating anticipation and recovery effects.

These hypotheses are grounded in the EDA because the notebook compares levels, distributions, variability, temporal patterns, and event-centered windows rather than relying on one summary statistic.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

BASE_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH) 
