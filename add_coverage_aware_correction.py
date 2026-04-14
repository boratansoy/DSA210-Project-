from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
MAIN_NOTEBOOK = BASE_DIR / "advanced_behavioral_spotify_eda.ipynb"
UPDATED_NOTEBOOK = BASE_DIR / "advanced_behavioral_spotify_eda_UPDATED_WITH_HYPOTHESIS.ipynb"
TAG = "coverage_aware_correction_extension"


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
# Coverage-Aware Correction for Limited Label Data

The Spotify streaming dataset covers a long period, but the context-label files do not necessarily cover the exact same range. This matters statistically: if a label file ends earlier than the Spotify data, then days after the label file ends should not automatically be treated as true non-vacation or non-special days.

This section adds a correction by defining explicit coverage windows for the calendar/special-date labels and the travel/break labels. The cleaner interpretation should rely on comparisons made inside the relevant coverage window.
"""
        ),
        code_cell(
            r"""
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import chi2_contingency, ttest_ind


COVERAGE_ALPHA = 0.05
BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"


def coverage_resolve_project_file(filename):
    matches = sorted(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]
    return BASE_DIR / filename


def coverage_first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def coverage_decision(p_value, alpha=COVERAGE_ALPHA):
    if pd.isna(p_value):
        return "Test not run"
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def coverage_binary(series):
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


def coverage_get_analysis_df():
    if "analysis_with_vacation_df" in globals() and isinstance(globals()["analysis_with_vacation_df"], pd.DataFrame):
        df = globals()["analysis_with_vacation_df"].copy()
    elif "analysis_df" in globals() and isinstance(globals()["analysis_df"], pd.DataFrame):
        df = globals()["analysis_df"].copy()
    else:
        raise ValueError("Daily analysis dataframe is not available. Run the preprocessing cells first.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).copy()


def coverage_get_special_df():
    if "special" in globals() and isinstance(globals()["special"], pd.DataFrame):
        df = globals()["special"].copy()
    elif "special_raw" in globals() and isinstance(globals()["special_raw"], pd.DataFrame):
        df = globals()["special_raw"].copy()
    else:
        path = coverage_resolve_project_file("special_dates.csv")
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
    return df


def coverage_get_travel_df():
    if "travel_periods_raw" in globals() and isinstance(globals()["travel_periods_raw"], pd.DataFrame):
        df = globals()["travel_periods_raw"].copy()
    else:
        path = coverage_resolve_project_file("travel_and_break_periods.csv")
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df = df.dropna(subset=["start_date", "end_date"]).copy()
    return df


analysis_coverage_base_df = coverage_get_analysis_df()
special_coverage_raw_df = coverage_get_special_df()
travel_coverage_raw_df = coverage_get_travel_df()

streaming_min_date = analysis_coverage_base_df["date"].min()
streaming_max_date = analysis_coverage_base_df["date"].max()

special_min_date = special_coverage_raw_df["date"].min() if not special_coverage_raw_df.empty and "date" in special_coverage_raw_df.columns else pd.NaT
special_max_date = special_coverage_raw_df["date"].max() if not special_coverage_raw_df.empty and "date" in special_coverage_raw_df.columns else pd.NaT

travel_min_date = travel_coverage_raw_df["start_date"].min() if not travel_coverage_raw_df.empty else pd.NaT
travel_max_date = travel_coverage_raw_df["end_date"].max() if not travel_coverage_raw_df.empty else pd.NaT

coverage_summary_df = pd.DataFrame([
    {
        "dataset": "spotify_daily_behavior",
        "coverage_start": streaming_min_date,
        "coverage_end": streaming_max_date,
        "rows_or_days": len(analysis_coverage_base_df),
        "interpretation": "Full available daily listening behavior.",
    },
    {
        "dataset": "special_dates_labels",
        "coverage_start": special_min_date,
        "coverage_end": special_max_date,
        "rows_or_days": len(special_coverage_raw_df),
        "interpretation": "Only this window should be used when treating missing exam/deadline/stress labels as 0.",
    },
    {
        "dataset": "travel_and_break_periods",
        "coverage_start": travel_min_date,
        "coverage_end": travel_max_date,
        "rows_or_days": len(travel_coverage_raw_df),
        "interpretation": "Only this window should be used when treating missing travel/break labels as 0.",
    },
])

display(coverage_summary_df)

analysis_special_coverage_df = analysis_coverage_base_df.copy()
if pd.notna(special_min_date) and pd.notna(special_max_date):
    analysis_special_coverage_df = analysis_special_coverage_df[
        analysis_special_coverage_df["date"].between(special_min_date, special_max_date)
    ].copy()

analysis_vacation_coverage_df = analysis_coverage_base_df.copy()
if pd.notna(travel_min_date) and pd.notna(travel_max_date):
    analysis_vacation_coverage_df = analysis_vacation_coverage_df[
        analysis_vacation_coverage_df["date"].between(travel_min_date, travel_max_date)
    ].copy()

print("Daily rows in full Spotify analysis:", len(analysis_coverage_base_df))
print("Daily rows inside special-date label coverage:", len(analysis_special_coverage_df))
print("Daily rows inside travel/break label coverage:", len(analysis_vacation_coverage_df))
"""
        ),
        code_cell(
            r"""
display(Markdown(
    "**Coverage insight:** The original full-period Spotify dataset is useful for general listening patterns, "
    "but context comparisons should be made only where the relevant context labels are observable. "
    "For vacation analysis, days after the travel/break label coverage ends are not reliable non-vacation controls. "
    "For exam/deadline/stress analysis, days outside the calendar label coverage should not be interpreted as confirmed normal academic days."
))
"""
        ),
        markdown_cell(
            """
## Coverage-Aware Special-Day Tests

The previous academic-context tests are repeated inside the available special-date coverage window. This avoids comparing labeled exam/deadline/stress days against years where the calendar label file may not have been available.
"""
        ),
        code_cell(
            r"""
coverage_results = []


def coverage_run_ttest(df, label_column, metric_column, hypothesis_name):
    if label_column not in df.columns or metric_column not in df.columns:
        print(f"{hypothesis_name}: skipped because {label_column} or {metric_column} is missing.")
        coverage_results.append({
            "hypothesis": hypothesis_name,
            "test": "Welch two-sample t-test",
            "n_label_1": np.nan,
            "n_label_0": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    group_1 = pd.to_numeric(df.loc[df[label_column] == 1, metric_column], errors="coerce").dropna()
    group_0 = pd.to_numeric(df.loc[df[label_column] == 0, metric_column], errors="coerce").dropna()

    print(f"\n{hypothesis_name}")
    print(f"n {label_column}=1: {len(group_1)}")
    print(f"n {label_column}=0: {len(group_0)}")

    if len(group_1) < 2 or len(group_0) < 2:
        print("Not enough observations in both groups.")
        coverage_results.append({
            "hypothesis": hypothesis_name,
            "test": "Welch two-sample t-test",
            "n_label_1": len(group_1),
            "n_label_0": len(group_0),
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "Test not run",
        })
        return None

    statistic, p_value = ttest_ind(group_1, group_0, equal_var=False, nan_policy="omit")
    decision = coverage_decision(p_value)

    print(f"t-statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Decision at alpha={COVERAGE_ALPHA}: {decision}")
    print(f"mean {label_column}=1: {group_1.mean():.3f}")
    print(f"mean {label_column}=0: {group_0.mean():.3f}")

    coverage_results.append({
        "hypothesis": hypothesis_name,
        "test": "Welch two-sample t-test",
        "n_label_1": len(group_1),
        "n_label_0": len(group_0),
        "statistic": statistic,
        "p_value": p_value,
        "decision": decision,
    })
    return statistic, p_value, decision


coverage_run_ttest(
    analysis_special_coverage_df,
    "is_exam",
    "total_minutes",
    "Coverage-aware H1: exam vs non-exam daily listening time",
)

coverage_run_ttest(
    analysis_special_coverage_df,
    "is_stress_period",
    "skip_rate",
    "Coverage-aware H2: stress vs non-stress skip rate",
)

coverage_run_ttest(
    analysis_special_coverage_df,
    "is_stress_period",
    "unique_artists",
    "Coverage-aware H3: stress vs non-stress artist diversity",
)

coverage_run_ttest(
    analysis_special_coverage_df,
    "is_deadline",
    "num_streams",
    "Coverage-aware H4: deadline vs non-deadline stream count",
)
"""
        ),
        code_cell(
            r"""
display(Markdown(
    "**Interpretation:** These coverage-aware academic tests should be treated as the more methodologically reliable version "
    "of the exam/deadline/stress comparisons. They compare special days only against days from the same observable calendar-label period, "
    "which reduces the risk of calling an unlabeled historical day a true normal day."
))
"""
        ),
        markdown_cell(
            """
## Coverage-Aware Vacation Tests

The vacation tests are repeated inside the travel/break period coverage window. This is especially important because the travel file ends before the Spotify streaming dataset ends.
"""
        ),
        code_cell(
            r"""
coverage_run_ttest(
    analysis_vacation_coverage_df,
    "is_travel_vacation",
    "total_minutes",
    "Coverage-aware H7a: vacation vs non-vacation daily listening time",
)

coverage_run_ttest(
    analysis_vacation_coverage_df,
    "is_travel_vacation",
    "num_streams",
    "Coverage-aware H7b: vacation vs non-vacation daily stream count",
)


event_vacation_coverage_df = pd.DataFrame()
if "event_vacation_df" in globals() and isinstance(globals()["event_vacation_df"], pd.DataFrame) and not event_vacation_df.empty:
    event_vacation_coverage_df = event_vacation_df.copy()
    event_vacation_coverage_df["date"] = pd.to_datetime(event_vacation_coverage_df["date"], errors="coerce")
    if pd.notna(travel_min_date) and pd.notna(travel_max_date):
        event_vacation_coverage_df = event_vacation_coverage_df[
            event_vacation_coverage_df["date"].between(travel_min_date, travel_max_date)
        ].copy()

if not event_vacation_coverage_df.empty and {"vacation_status", "daypart"}.issubset(event_vacation_coverage_df.columns):
    daypart_order = ["night", "morning", "afternoon", "evening"]
    coverage_daypart_contingency = (
        pd.crosstab(event_vacation_coverage_df["vacation_status"], event_vacation_coverage_df["daypart"])
        .reindex(columns=daypart_order, fill_value=0)
    )
    display(coverage_daypart_contingency)

    if coverage_daypart_contingency.shape[0] >= 2 and coverage_daypart_contingency.shape[1] >= 2:
        chi2_statistic, p_value, dof, expected = chi2_contingency(coverage_daypart_contingency)
        expected_df = pd.DataFrame(expected, index=coverage_daypart_contingency.index, columns=coverage_daypart_contingency.columns)
        display(expected_df.round(2))
        decision = coverage_decision(p_value)
        print("Coverage-aware H8: vacation status and listening daypart")
        print(f"chi-square statistic: {chi2_statistic:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"degrees of freedom: {dof}")
        print(f"Decision at alpha={COVERAGE_ALPHA}: {decision}")
        coverage_results.append({
            "hypothesis": "Coverage-aware H8: vacation status and listening daypart",
            "test": "Chi-square test of independence",
            "n_label_1": int((event_vacation_coverage_df["is_travel_vacation"] == 1).sum()) if "is_travel_vacation" in event_vacation_coverage_df.columns else np.nan,
            "n_label_0": int((event_vacation_coverage_df["is_travel_vacation"] == 0).sum()) if "is_travel_vacation" in event_vacation_coverage_df.columns else np.nan,
            "statistic": chi2_statistic,
            "p_value": p_value,
            "decision": decision,
        })
    else:
        print("Coverage-aware H8 skipped: contingency table does not contain both vacation statuses and dayparts.")
else:
    print("Coverage-aware H8 skipped: event-level daypart data is unavailable.")
"""
        ),
        code_cell(
            r"""
coverage_results_df = pd.DataFrame(coverage_results)
display(coverage_results_df)

display(Markdown(
    "**Updated methodological conclusion:** Yes, the notebook needs this correction because the label datasets are not equally complete over time. "
    "The full Spotify dataset remains valuable for baseline behavior, but hypothesis tests involving exams, stress, deadlines, vacations, or breaks should be interpreted within the date range where those labels are available. "
    "This makes the study clearer, avoids false control days, and makes the final conclusions more defensible."
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
    update_notebook(MAIN_NOTEBOOK)
    update_notebook(UPDATED_NOTEBOOK)


if __name__ == "__main__":
    main()
