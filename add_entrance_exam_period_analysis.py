from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
NOTEBOOK_PATHS = [
    BASE_DIR / "advanced_behavioral_spotify_eda.ipynb",
    BASE_DIR / "advanced_behavioral_spotify_eda_UPDATED_WITH_HYPOTHESIS.ipynb",
]
TAG = "entrance_exam_period_extension"


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [TAG]},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def build_cells() -> list[dict[str, Any]]:
    return [
        markdown_cell(
            """
# Entrance Exam Preparation Context: LGS and YKS

This project already studies exams, deadlines, stress periods, travel, and breaks. However, two major long-term academic contexts are especially important for interpreting listening behavior:

- **LGS preparation:** 2019-2020 education year, with the exam period in June 2020.
- **YKS preparation:** 2023-2024 education year, with the exam period in June 2024.

Because both exams take place in June, this section treats the final phase before June as a higher-stress preparation window. To keep the analysis transparent, the notebook uses rule-based period labels instead of pretending to know exact daily stress intensity.
"""
        ),
        markdown_cell(
            """
## Period Definition and Assumptions

The preparation periods are defined as full education-year windows from September through June. The high-stress phase is defined as March through June, capturing the final preparation months and the exam month itself.

This is a modeling assumption. It is useful because it creates a consistent way to compare ordinary preparation months against the final high-pressure phase for both LGS and YKS.
"""
        ),
        code_cell(
            r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from scipy.stats import ttest_ind


ENTRANCE_ALPHA = 0.05
BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"


def entrance_resolve_project_file(filename):
    matches = sorted(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]
    return BASE_DIR / filename


def entrance_first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def entrance_binary(series):
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
    return mapped.fillna(pd.to_numeric(series, errors="coerce")).fillna(0).astype(int)


def entrance_get_daily_analysis_df():
    # Reuse the richest daily dataframe available; reconstruct only if needed.
    for candidate in ["analysis_with_vacation_df", "analysis_coverage_base_df", "analysis_df", "daily_df"]:
        if candidate in globals() and isinstance(globals()[candidate], pd.DataFrame):
            daily = globals()[candidate].copy()
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
            return daily.dropna(subset=["date"]).copy()

    spotify_path = entrance_resolve_project_file("spotify_cleaned.csv")
    if not spotify_path.exists():
        raise FileNotFoundError("spotify_cleaned.csv could not be found in the project folder or subfolders.")

    spotify_raw_local = pd.read_csv(spotify_path)
    ts_column = entrance_first_existing(spotify_raw_local, ["ts", "timestamp", "played_at", "event_timestamp"])
    ms_column = entrance_first_existing(spotify_raw_local, ["ms_played", "msPlayed", "milliseconds_played"])
    artist_column = entrance_first_existing(
        spotify_raw_local,
        ["master_metadata_album_artist_name", "artist_name", "artist", "album_artist"],
    )
    track_column = entrance_first_existing(
        spotify_raw_local,
        ["master_metadata_track_name", "track_name", "track", "episode_name"],
    )
    skipped_column = entrance_first_existing(spotify_raw_local, ["skipped", "skip", "was_skipped"])
    shuffle_column = entrance_first_existing(spotify_raw_local, ["shuffle", "shuffle_state", "is_shuffle"])

    if ts_column is None or ms_column is None:
        raise ValueError("Cannot reconstruct daily behavior without timestamp and ms_played columns.")

    events = spotify_raw_local.copy()
    events[ts_column] = pd.to_datetime(events[ts_column], errors="coerce", utc=True)
    events = events.dropna(subset=[ts_column]).copy()
    events["date"] = pd.to_datetime(events[ts_column].dt.date)
    events[ms_column] = pd.to_numeric(events[ms_column], errors="coerce").fillna(0)

    daily = events.groupby("date").agg(total_ms=(ms_column, "sum"), num_streams=(ms_column, "size")).reset_index()
    daily["total_minutes"] = daily["total_ms"] / 60000
    daily["unique_artists"] = events.groupby("date")[artist_column].nunique().values if artist_column else np.nan
    daily["unique_tracks"] = events.groupby("date")[track_column].nunique().values if track_column else np.nan
    daily["skip_rate"] = events.groupby("date")[skipped_column].apply(lambda x: entrance_binary(x).mean()).values if skipped_column else np.nan
    daily["shuffle_rate"] = events.groupby("date")[shuffle_column].apply(lambda x: entrance_binary(x).mean()).values if shuffle_column else np.nan
    return daily


entrance_periods = pd.DataFrame([
    {
        "exam_type": "LGS",
        "exam_description": "High School Entrance Exam",
        "education_year": "2019-2020",
        "prep_start": "2019-09-01",
        "prep_end": "2020-06-30",
        "high_stress_start": "2020-03-01",
        "high_stress_end": "2020-06-30",
        "assumption": "Final preparation months plus June exam month",
    },
    {
        "exam_type": "YKS",
        "exam_description": "University Entrance Exam",
        "education_year": "2023-2024",
        "prep_start": "2023-09-01",
        "prep_end": "2024-06-30",
        "high_stress_start": "2024-03-01",
        "high_stress_end": "2024-06-30",
        "assumption": "Final preparation months plus June exam month",
    },
])

for column in ["prep_start", "prep_end", "high_stress_start", "high_stress_end"]:
    entrance_periods[column] = pd.to_datetime(entrance_periods[column])

display(entrance_periods)
"""
        ),
        code_cell(
            r"""
def expand_entrance_exam_periods(periods_df):
    rows = []
    for _, period in periods_df.iterrows():
        for active_date in pd.date_range(period["prep_start"], period["prep_end"], freq="D"):
            is_high_stress = int(period["high_stress_start"] <= active_date <= period["high_stress_end"])
            rows.append({
                "date": pd.to_datetime(active_date.date()),
                "entrance_exam_type": period["exam_type"],
                "entrance_exam_description": period["exam_description"],
                "entrance_education_year": period["education_year"],
                "is_lgs_prep": int(period["exam_type"] == "LGS"),
                "is_yks_prep": int(period["exam_type"] == "YKS"),
                "is_entrance_exam_prep": 1,
                "is_lgs_high_stress": int(period["exam_type"] == "LGS" and is_high_stress == 1),
                "is_yks_high_stress": int(period["exam_type"] == "YKS" and is_high_stress == 1),
                "is_entrance_exam_high_stress": is_high_stress,
                "entrance_exam_phase": "final_high_stress" if is_high_stress else "general_preparation",
            })

    return pd.DataFrame(rows)


entrance_exam_daily = expand_entrance_exam_periods(entrance_periods)
analysis_entrance_base_df = entrance_get_daily_analysis_df()
analysis_entrance_base_df["date"] = pd.to_datetime(analysis_entrance_base_df["date"], errors="coerce")

entrance_columns_to_drop = [
    "entrance_exam_type",
    "entrance_exam_description",
    "entrance_education_year",
    "is_lgs_prep",
    "is_yks_prep",
    "is_entrance_exam_prep",
    "is_lgs_high_stress",
    "is_yks_high_stress",
    "is_entrance_exam_high_stress",
    "entrance_exam_phase",
]

analysis_entrance_df = analysis_entrance_base_df.drop(columns=entrance_columns_to_drop, errors="ignore").merge(
    entrance_exam_daily,
    on="date",
    how="left",
)

entrance_binary_columns = [
    "is_lgs_prep",
    "is_yks_prep",
    "is_entrance_exam_prep",
    "is_lgs_high_stress",
    "is_yks_high_stress",
    "is_entrance_exam_high_stress",
]

for column in entrance_binary_columns:
    analysis_entrance_df[column] = pd.to_numeric(analysis_entrance_df[column], errors="coerce").fillna(0).astype(int)

for column in ["entrance_exam_type", "entrance_exam_description", "entrance_education_year", "entrance_exam_phase"]:
    analysis_entrance_df[column] = analysis_entrance_df[column].fillna("").astype(str)

entrance_overlap_summary = (
    analysis_entrance_df.loc[analysis_entrance_df["is_entrance_exam_prep"] == 1]
    .groupby(["entrance_exam_type", "entrance_exam_phase"])
    .agg(
        spotify_days=("date", "nunique"),
        mean_total_minutes=("total_minutes", "mean"),
        median_total_minutes=("total_minutes", "median"),
        mean_num_streams=("num_streams", "mean"),
        mean_unique_artists=("unique_artists", "mean"),
    )
    .reset_index()
)

print("Entrance exam daily labels:", entrance_exam_daily.shape)
print("Daily Spotify rows with entrance-exam labels:", int(analysis_entrance_df["is_entrance_exam_prep"].sum()))
display(entrance_overlap_summary.round(2))
"""
        ),
        code_cell(
            r"""
display(Markdown(
    "**Labeling insight:** The entrance-exam labels are intentionally period-based rather than single-day labels. "
    "This is appropriate because LGS and YKS preparation are long-term academic contexts: the behavioral question is not only whether music changes on the exam day, "
    "but whether listening shifts during the preparation cycle and especially during the final high-stress phase."
))
"""
        ),
        markdown_cell(
            """
## Listening Behavior Across LGS and YKS Preparation

The next view compares the full preparation periods and highlights whether the final high-stress phase differs from general preparation months.
"""
        ),
        code_cell(
            r"""
entrance_focus_df = analysis_entrance_df.loc[analysis_entrance_df["is_entrance_exam_prep"] == 1].copy()

if entrance_focus_df.empty:
    display(Markdown("No Spotify listening days overlap with the LGS/YKS preparation windows."))
else:
    plt.figure(figsize=(12, 5))
    plt.plot(entrance_focus_df["date"], entrance_focus_df["total_minutes"])
    plt.title("Daily Listening Time During LGS and YKS Preparation Periods")
    plt.xlabel("Date")
    plt.ylabel("Total minutes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    phase_summary = (
        entrance_focus_df.groupby(["entrance_exam_type", "entrance_exam_phase"])[
            ["total_minutes", "num_streams", "unique_artists", "skip_rate"]
        ]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    display(phase_summary)
"""
        ),
        code_cell(
            r"""
if not entrance_focus_df.empty:
    lgs_days = int((entrance_focus_df["is_lgs_prep"] == 1).sum())
    yks_days = int((entrance_focus_df["is_yks_prep"] == 1).sum())
    high_stress_days = int((entrance_focus_df["is_entrance_exam_high_stress"] == 1).sum())
    display(Markdown(
        f"**Behavioral insight:** The notebook can compare **{lgs_days} LGS-preparation listening days** and "
        f"**{yks_days} YKS-preparation listening days**, including **{high_stress_days} final high-stress days**. "
        "This creates a stronger long-term stress analysis than single exam-day labels because entrance exams affect routines for months."
    ))
"""
        ),
        markdown_cell(
            """
## High-Stress Entrance Exam Phase vs General Preparation

This comparison focuses only on LGS/YKS preparation days. The control group is not the entire Spotify history; it is the general preparation months from the same entrance-exam windows.
"""
        ),
        code_cell(
            r"""
entrance_metrics = [
    metric for metric in ["total_minutes", "num_streams", "unique_artists", "skip_rate", "shuffle_rate"]
    if metric in entrance_focus_df.columns
]

if entrance_focus_df.empty or not entrance_metrics:
    display(Markdown("Entrance-exam preparation comparison is unavailable because no overlapping metrics were found."))
else:
    entrance_mean_table = entrance_focus_df.groupby("is_entrance_exam_high_stress")[entrance_metrics].mean()
    entrance_median_table = entrance_focus_df.groupby("is_entrance_exam_high_stress")[entrance_metrics].median()

    print("Mean behavior: general preparation vs final high-stress phase")
    display(entrance_mean_table.round(3))
    print("Median behavior: general preparation vs final high-stress phase")
    display(entrance_median_table.round(3))

    for metric in ["total_minutes", "num_streams", "unique_artists"]:
        if metric not in entrance_focus_df.columns:
            continue

        general = pd.to_numeric(
            entrance_focus_df.loc[entrance_focus_df["is_entrance_exam_high_stress"] == 0, metric],
            errors="coerce",
        ).dropna()
        high_stress = pd.to_numeric(
            entrance_focus_df.loc[entrance_focus_df["is_entrance_exam_high_stress"] == 1, metric],
            errors="coerce",
        ).dropna()

        if len(general) > 0 and len(high_stress) > 0:
            plt.figure(figsize=(7, 5))
            plt.boxplot([general, high_stress], labels=["General prep", "Final high-stress"])
            plt.title(f"{metric}: Entrance Exam Preparation Phase Comparison")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 5))
            plt.hist(general, bins=20, alpha=0.6, label="General prep")
            plt.hist(high_stress, bins=20, alpha=0.6, label="Final high-stress")
            plt.title(f"Distribution of {metric}: General Prep vs Final High-Stress")
            plt.xlabel(metric)
            plt.ylabel("Number of days")
            plt.legend()
            plt.tight_layout()
            plt.show()
"""
        ),
        code_cell(
            r"""
if not entrance_focus_df.empty and {"total_minutes", "num_streams", "unique_artists"}.issubset(entrance_focus_df.columns):
    summary = entrance_focus_df.groupby("is_entrance_exam_high_stress")[["total_minutes", "num_streams", "unique_artists"]].mean()
    if {0, 1}.issubset(set(summary.index)):
        changes = summary.loc[1] - summary.loc[0]
        display(Markdown(
            f"**Insight:** During the final entrance-exam high-stress phase, average daily listening changes by "
            f"**{changes['total_minutes']:.1f} minutes**, **{changes['num_streams']:.1f} streams**, and "
            f"**{changes['unique_artists']:.1f} unique artists** relative to general preparation months. "
            "A decrease can suggest routine compression or reduced leisure time, while an increase can suggest coping, background listening, or greater reliance on music for emotional regulation."
        ))
"""
        ),
        markdown_cell(
            """
## LGS vs YKS Context Comparison

LGS and YKS occurred at different life stages. Comparing them is exploratory, but useful: it can reveal whether later university-entrance preparation produced a different listening pattern than earlier high-school-entrance preparation.
"""
        ),
        code_cell(
            r"""
if entrance_focus_df.empty:
    display(Markdown("LGS/YKS comparison is unavailable because no entrance-exam listening days overlap with Spotify data."))
else:
    exam_type_summary = (
        entrance_focus_df.groupby("entrance_exam_type")[["total_minutes", "num_streams", "unique_artists", "skip_rate"]]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    display(exam_type_summary)

    for metric in ["total_minutes", "num_streams", "unique_artists"]:
        lgs_values = pd.to_numeric(entrance_focus_df.loc[entrance_focus_df["entrance_exam_type"] == "LGS", metric], errors="coerce").dropna()
        yks_values = pd.to_numeric(entrance_focus_df.loc[entrance_focus_df["entrance_exam_type"] == "YKS", metric], errors="coerce").dropna()

        if len(lgs_values) > 0 and len(yks_values) > 0:
            plt.figure(figsize=(7, 5))
            plt.boxplot([lgs_values, yks_values], labels=["LGS prep", "YKS prep"])
            plt.title(f"{metric}: LGS vs YKS Preparation")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.show()
"""
        ),
        code_cell(
            r"""
if not entrance_focus_df.empty:
    display(Markdown(
        "**Context insight:** The LGS-YKS comparison should be interpreted carefully because age, school environment, technology habits, and Spotify usage maturity may differ between 2019-2020 and 2023-2024. "
        "Still, it is useful as a within-person comparison of two major academic preparation contexts."
    ))
"""
        ),
        markdown_cell(
            """
# Additional Hypothesis Tests for Entrance Exam Preparation

These tests are restricted to LGS/YKS preparation days. This makes the control group conceptually cleaner: final high-stress months are compared with general preparation months rather than with unrelated years.
"""
        ),
        code_cell(
            r"""
entrance_hypothesis_results = []


def entrance_decision(p_value, alpha=ENTRANCE_ALPHA):
    if pd.isna(p_value):
        return "Test not run"
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def entrance_run_ttest(df, group_column, metric_column, hypothesis_name, group_1_label="1", group_0_label="0"):
    if df.empty or group_column not in df.columns or metric_column not in df.columns:
        print(f"{hypothesis_name}: skipped because required data are missing.")
        entrance_hypothesis_results.append({
            "hypothesis": hypothesis_name,
            "test": "Welch two-sample t-test",
            "n_group_1": np.nan,
            "n_group_0": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    group_1 = pd.to_numeric(df.loc[df[group_column] == 1, metric_column], errors="coerce").dropna()
    group_0 = pd.to_numeric(df.loc[df[group_column] == 0, metric_column], errors="coerce").dropna()

    print(f"\n{hypothesis_name}")
    print(f"n {group_1_label}: {len(group_1)}")
    print(f"n {group_0_label}: {len(group_0)}")

    if len(group_1) < 2 or len(group_0) < 2:
        print("Not enough observations in both groups to run the test.")
        entrance_hypothesis_results.append({
            "hypothesis": hypothesis_name,
            "test": "Welch two-sample t-test",
            "n_group_1": len(group_1),
            "n_group_0": len(group_0),
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    statistic, p_value = ttest_ind(group_1, group_0, equal_var=False, nan_policy="omit")
    decision = entrance_decision(p_value)

    print(f"t-statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Decision at alpha={ENTRANCE_ALPHA}: {decision}")
    print(f"mean {group_1_label}: {group_1.mean():.3f}")
    print(f"mean {group_0_label}: {group_0.mean():.3f}")

    entrance_hypothesis_results.append({
        "hypothesis": hypothesis_name,
        "test": "Welch two-sample t-test",
        "n_group_1": len(group_1),
        "n_group_0": len(group_0),
        "statistic": statistic,
        "p_value": p_value,
        "decision": decision,
    })
    return statistic, p_value, decision, group_1.mean(), group_0.mean()
"""
        ),
        markdown_cell(
            """
## H9 - Listening Time During Final Entrance-Exam Stress

**H0:** Mean daily listening time is the same during final entrance-exam high-stress months and general preparation months.

**HA:** Mean daily listening time differs during final entrance-exam high-stress months.
"""
        ),
        code_cell(
            r"""
h9_result = entrance_run_ttest(
    entrance_focus_df,
    "is_entrance_exam_high_stress",
    "total_minutes",
    "H9: entrance-exam high-stress phase vs general preparation - listening time",
    "final high-stress",
    "general preparation",
)

if h9_result is not None:
    _, p_value, decision, high_mean, general_mean = h9_result
    direction = "higher" if high_mean > general_mean else "lower"
    display(Markdown(
        f"**H9 conclusion:** {decision}. Average listening time is **{direction}** during the final high-stress phase "
        f"({high_mean:.1f} vs {general_mean:.1f} minutes). This result speaks directly to whether long-term entrance-exam pressure changes daily listening intensity."
    ))
"""
        ),
        markdown_cell(
            """
## H10 - Artist Diversity During Final Entrance-Exam Stress

**H0:** Mean artist diversity is the same during final entrance-exam high-stress months and general preparation months.

**HA:** Mean artist diversity differs during final entrance-exam high-stress months.
"""
        ),
        code_cell(
            r"""
h10_result = entrance_run_ttest(
    entrance_focus_df,
    "is_entrance_exam_high_stress",
    "unique_artists",
    "H10: entrance-exam high-stress phase vs general preparation - artist diversity",
    "final high-stress",
    "general preparation",
)

if h10_result is not None:
    _, p_value, decision, high_mean, general_mean = h10_result
    direction = "higher" if high_mean > general_mean else "lower"
    display(Markdown(
        f"**H10 conclusion:** {decision}. Artist diversity is **{direction}** during the final high-stress phase "
        f"({high_mean:.1f} vs {general_mean:.1f} unique artists). Lower diversity may suggest comfort/repetition, while higher diversity may suggest exploratory coping."
    ))
"""
        ),
        markdown_cell(
            """
## H11 - Listening Frequency During Final Entrance-Exam Stress

**H0:** Mean daily stream count is the same during final entrance-exam high-stress months and general preparation months.

**HA:** Mean daily stream count differs during final entrance-exam high-stress months.
"""
        ),
        code_cell(
            r"""
h11_result = entrance_run_ttest(
    entrance_focus_df,
    "is_entrance_exam_high_stress",
    "num_streams",
    "H11: entrance-exam high-stress phase vs general preparation - stream count",
    "final high-stress",
    "general preparation",
)

if h11_result is not None:
    _, p_value, decision, high_mean, general_mean = h11_result
    direction = "higher" if high_mean > general_mean else "lower"
    display(Markdown(
        f"**H11 conclusion:** {decision}. Stream count is **{direction}** during the final high-stress phase "
        f"({high_mean:.1f} vs {general_mean:.1f} streams). This helps distinguish longer listening from more frequent, fragmented interaction."
    ))
"""
        ),
        markdown_cell(
            """
## H12 - LGS vs YKS Preparation Listening

**H0:** Mean daily listening behavior is the same during LGS and YKS preparation periods.

**HA:** Mean daily listening behavior differs between LGS and YKS preparation periods.

This test is exploratory because the two periods happen at different ages and life stages.
"""
        ),
        code_cell(
            r"""
lgs_yks_df = entrance_focus_df.loc[entrance_focus_df["entrance_exam_type"].isin(["LGS", "YKS"])].copy()
lgs_yks_df["is_yks_context"] = (lgs_yks_df["entrance_exam_type"] == "YKS").astype(int)

h12_result = entrance_run_ttest(
    lgs_yks_df,
    "is_yks_context",
    "total_minutes",
    "H12: YKS preparation vs LGS preparation - listening time",
    "YKS preparation",
    "LGS preparation",
)

if h12_result is not None:
    _, p_value, decision, yks_mean, lgs_mean = h12_result
    direction = "higher" if yks_mean > lgs_mean else "lower"
    display(Markdown(
        f"**H12 conclusion:** {decision}. Average listening time is **{direction}** during YKS preparation than LGS preparation "
        f"({yks_mean:.1f} vs {lgs_mean:.1f} minutes). This may reflect not only exam context, but also changes in age, habits, autonomy, and Spotify usage over time."
    ))
"""
        ),
        code_cell(
            r"""
entrance_hypothesis_results_df = pd.DataFrame(entrance_hypothesis_results)
display(entrance_hypothesis_results_df)

display(Markdown(
    "**Entrance-exam update summary:** Adding LGS and YKS creates a stronger long-term academic stress layer. "
    "The key methodological improvement is that final high-stress months are compared against general preparation months from the same entrance-exam periods, not against unrelated years. "
    "This makes the interpretation clearer: any detected shift is more plausibly tied to entrance-exam preparation context."
))
"""
        ),
    ]


def update_notebook(path: Path) -> None:
    if not path.exists():
        return

    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"] = [
        cell
        for cell in nb.get("cells", [])
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    nb["cells"].extend(build_cells())
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {path.name}: {len(nb['cells'])} cells")


def main() -> None:
    for path in NOTEBOOK_PATHS:
        update_notebook(path)


if __name__ == "__main__":
    main()
