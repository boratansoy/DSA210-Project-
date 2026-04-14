from pathlib import Path
import json


NOTEBOOK_PATH = Path.home() / "Desktop" / "DSA210 TERM PROJECT" / "advanced_behavioral_spotify_eda.ipynb"
EXT_TAG = "hypothesis_testing_extension"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [EXT_TAG]},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [EXT_TAG]},
        "outputs": [],
        "source": source.strip() + "\n",
    }


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

# Keep the existing EDA intact, but make the operation idempotent.
nb["cells"] = [
    cell
    for cell in nb.get("cells", [])
    if EXT_TAG not in cell.get("metadata", {}).get("tags", [])
]

extension_cells = [
    md(
        """
# Hypothesis Testing

This section extends the exploratory analysis with formal statistical tests. The goal is to evaluate whether the behavioral differences observed in the EDA are strong enough to count as statistical evidence against a null hypothesis.

All tests use **alpha = 0.05**. These tests support interpretation, but they do not prove causality.
"""
    ),
    md(
        """
## Statistical setup and safe data preparation

The notebook already creates a merged daily dataframe earlier. This setup cell reuses `analysis_df` if it exists. If the notebook is run from this section alone, it reconstructs the required daily dataset from `spotify_cleaned.csv` and `special_dates.csv`.
"""
    ),
    code(
        r'''
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, chi2_contingency
from IPython.display import Markdown, display

ALPHA = 0.05
BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
SPOTIFY_PATH = BASE_DIR / "spotify_cleaned.csv"
SPECIAL_DATES_PATH = BASE_DIR / "special_dates.csv"


def first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def to_binary(series):
    if series.dtype == bool:
        return series.astype(int)
    normalized = series.astype(str).str.lower().str.strip()
    mapped = normalized.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "yes": 1, "no": 0, "nan": 0, "none": 0,
    })
    return mapped.fillna(pd.to_numeric(series, errors="coerce")).fillna(0)


def reconstruct_analysis_df_if_needed():
    if "analysis_df" in globals() and isinstance(globals()["analysis_df"], pd.DataFrame):
        working = globals()["analysis_df"].copy()
        if "date" in working.columns:
            working["date"] = pd.to_datetime(working["date"], errors="coerce")
        return working

    spotify_raw = pd.read_csv(SPOTIFY_PATH)
    special_raw = pd.read_csv(SPECIAL_DATES_PATH)

    ts_col = first_existing(spotify_raw, ["ts", "timestamp", "played_at", "event_timestamp"])
    ms_col = first_existing(spotify_raw, ["ms_played", "msPlayed", "milliseconds_played"])
    artist_col = first_existing(spotify_raw, ["master_metadata_album_artist_name", "artist_name", "artist", "album_artist"])
    track_col = first_existing(spotify_raw, ["master_metadata_track_name", "track_name", "track"])
    skipped_col = first_existing(spotify_raw, ["skipped", "skip", "was_skipped"])
    shuffle_col = first_existing(spotify_raw, ["shuffle", "is_shuffle", "message_shuffle"])

    if ts_col is None or ms_col is None:
        raise ValueError("Cannot reconstruct analysis_df because timestamp or duration columns are missing.")

    spotify = spotify_raw.copy()
    spotify[ts_col] = pd.to_datetime(spotify[ts_col], errors="coerce", utc=True)
    spotify = spotify.dropna(subset=[ts_col]).copy()
    spotify["date"] = pd.to_datetime(spotify[ts_col].dt.date)
    spotify[ms_col] = pd.to_numeric(spotify[ms_col], errors="coerce").fillna(0)

    agg_dict = {ms_col: "sum"}
    if artist_col is not None:
        agg_dict[artist_col] = pd.Series.nunique
    if track_col is not None:
        agg_dict[track_col] = pd.Series.nunique

    daily = spotify.groupby("date").agg(agg_dict).reset_index()
    daily = daily.rename(columns={ms_col: "total_ms"})
    daily["total_minutes"] = daily["total_ms"] / 60000
    daily["num_streams"] = spotify.groupby("date").size().reindex(daily["date"]).values

    if artist_col is not None:
        daily = daily.rename(columns={artist_col: "unique_artists"})
    else:
        daily["unique_artists"] = pd.NA

    if track_col is not None:
        daily = daily.rename(columns={track_col: "unique_tracks"})
    else:
        daily["unique_tracks"] = pd.NA

    if skipped_col is not None:
        spotify["skipped_numeric"] = to_binary(spotify[skipped_col])
        daily = daily.merge(
            spotify.groupby("date")["skipped_numeric"].mean().reset_index(name="skip_rate"),
            on="date",
            how="left",
        )
    else:
        daily["skip_rate"] = pd.NA

    if shuffle_col is not None:
        spotify["shuffle_numeric"] = to_binary(spotify[shuffle_col])
        daily = daily.merge(
            spotify.groupby("date")["shuffle_numeric"].mean().reset_index(name="shuffle_rate"),
            on="date",
            how="left",
        )
    else:
        daily["shuffle_rate"] = pd.NA

    special = special_raw.copy()
    special["date"] = pd.to_datetime(special["date"], errors="coerce")
    special = special.dropna(subset=["date"]).copy()

    label_columns = ["is_exam", "is_deadline", "is_stress_period", "is_academic_event", "is_personal"]
    for column in label_columns:
        if column not in special.columns:
            special[column] = 0
        special[column] = pd.to_numeric(special[column], errors="coerce").fillna(0).astype(int)

    for column in ["source_events", "categories"]:
        if column not in special.columns:
            special[column] = ""
        special[column] = special[column].fillna("").astype(str)

    rebuilt = daily.merge(special, on="date", how="left")
    for column in label_columns:
        rebuilt[column] = pd.to_numeric(rebuilt[column], errors="coerce").fillna(0).astype(int)
    for column in ["source_events", "categories"]:
        if column in rebuilt.columns:
            rebuilt[column] = rebuilt[column].fillna("").astype(str)
    return rebuilt


analysis_test_df = reconstruct_analysis_df_if_needed().copy()
analysis_test_df["date"] = pd.to_datetime(analysis_test_df["date"], errors="coerce")

for label in ["is_exam", "is_deadline", "is_stress_period"]:
    if label not in analysis_test_df.columns:
        analysis_test_df[label] = 0
    analysis_test_df[label] = pd.to_numeric(analysis_test_df[label], errors="coerce").fillna(0).astype(int)

print("Hypothesis testing dataframe shape:", analysis_test_df.shape)
display(analysis_test_df.head())
'''
    ),
    md(
        """
## Test helper functions

These helper functions standardize each test output: sample sizes, group means, test statistic, p-value, alpha, and decision.
"""
    ),
    code(
        r'''
def decision_from_pvalue(p_value, alpha=ALPHA):
    if pd.isna(p_value):
        return "Test not available"
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def run_two_sample_ttest(df, label_col, value_col, label_name, h0, ha):
    if label_col not in df.columns:
        display(Markdown(f"**Skipped:** `{label_col}` is missing, so this test cannot be run."))
        return None
    if value_col not in df.columns:
        display(Markdown(f"**Skipped:** `{value_col}` is missing, so this test cannot be run."))
        return None

    group_1 = pd.to_numeric(df.loc[df[label_col] == 1, value_col], errors="coerce").dropna()
    group_0 = pd.to_numeric(df.loc[df[label_col] == 0, value_col], errors="coerce").dropna()

    if len(group_1) < 2 or len(group_0) < 2:
        display(Markdown(
            f"**Skipped:** `{label_name}` does not have enough observations for a two-sample t-test "
            f"(label=1 n={len(group_1)}, label=0 n={len(group_0)})."
        ))
        return None

    t_stat, p_value = ttest_ind(group_1, group_0, equal_var=False, nan_policy="omit")
    decision = decision_from_pvalue(p_value)

    result = {
        "test": "Welch two-sample t-test",
        "label": label_col,
        "variable": value_col,
        "n_label_1": len(group_1),
        "n_label_0": len(group_0),
        "mean_label_1": group_1.mean(),
        "mean_label_0": group_0.mean(),
        "median_label_1": group_1.median(),
        "median_label_0": group_0.median(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": ALPHA,
        "decision": decision,
    }

    print("Null hypothesis:", h0)
    print("Alternative hypothesis:", ha)
    print(f"Sample size ({label_col}=1):", len(group_1))
    print(f"Sample size ({label_col}=0):", len(group_0))
    print(f"Mean ({label_col}=1): {group_1.mean():.4f}")
    print(f"Mean ({label_col}=0): {group_0.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"alpha: {ALPHA}")
    print("Decision:", decision)

    direction = "higher" if group_1.mean() > group_0.mean() else "lower"
    if decision == "Reject H0":
        interpretation = (
            f"The p-value is below {ALPHA}, so we reject the null hypothesis. "
            f"This suggests that **{value_col} differs significantly** for {label_name}. "
            f"The labeled group has a {direction} mean than the comparison group."
        )
    else:
        interpretation = (
            f"The p-value is not below {ALPHA}, so we fail to reject the null hypothesis. "
            f"This test does not provide enough statistical evidence that **{value_col} differs** for {label_name}, "
            f"even if the descriptive means are not identical."
        )
    display(Markdown(f"**Interpretation:** {interpretation}"))
    return result


def run_paired_exam_ttest(df):
    required = {"date", "is_exam", "total_minutes"}
    missing = required - set(df.columns)
    if missing:
        display(Markdown(f"**Skipped:** Missing required columns for paired exam test: {sorted(missing)}."))
        return None

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    daily_lookup = working.set_index("date")["total_minutes"]
    exam_dates = sorted(working.loc[working["is_exam"] == 1, "date"].dropna().unique())

    pairs = []
    for exam_date in exam_dates:
        exam_date = pd.Timestamp(exam_date)
        before_date = exam_date - pd.Timedelta(days=1)
        after_date = exam_date + pd.Timedelta(days=1)
        if before_date in daily_lookup.index and after_date in daily_lookup.index:
            before_value = daily_lookup.loc[before_date]
            after_value = daily_lookup.loc[after_date]
            if pd.notna(before_value) and pd.notna(after_value):
                pairs.append({
                    "exam_date": exam_date,
                    "before_date": before_date,
                    "after_date": after_date,
                    "before_total_minutes": float(before_value),
                    "after_total_minutes": float(after_value),
                })

    pairs_df = pd.DataFrame(pairs)
    display(pairs_df)

    if len(pairs_df) < 2:
        display(Markdown(
            f"**Skipped:** Only {len(pairs_df)} valid before/after exam pairs were found. "
            "A paired t-test requires at least 2 valid pairs."
        ))
        return None

    t_stat, p_value = ttest_rel(
        pairs_df["before_total_minutes"],
        pairs_df["after_total_minutes"],
        nan_policy="omit",
    )
    decision = decision_from_pvalue(p_value)

    print("Null hypothesis: Mean listening time before exams is the same as mean listening time after exams.")
    print("Alternative hypothesis: Mean listening time before exams is different from mean listening time after exams.")
    print("Number of valid exam pairs:", len(pairs_df))
    print(f"Mean before exams: {pairs_df['before_total_minutes'].mean():.4f}")
    print(f"Mean after exams: {pairs_df['after_total_minutes'].mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"alpha: {ALPHA}")
    print("Decision:", decision)

    if decision == "Reject H0":
        interpretation = (
            f"The p-value is below {ALPHA}, so we reject the null hypothesis. "
            "This suggests a statistically detectable change in listening time from the day before exams to the day after exams."
        )
    else:
        interpretation = (
            f"The p-value is not below {ALPHA}, so we fail to reject the null hypothesis. "
            "The paired test does not provide enough evidence that listening time changes systematically before versus after exams."
        )
    display(Markdown(f"**Interpretation:** {interpretation}"))

    return {
        "test": "Paired t-test",
        "variable": "total_minutes",
        "n_pairs": len(pairs_df),
        "mean_before": pairs_df["before_total_minutes"].mean(),
        "mean_after": pairs_df["after_total_minutes"].mean(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": ALPHA,
        "decision": decision,
    }
'''
    ),
    md(
        """
## H1 — Exam days vs non-exam days: daily listening time

**H0:** Mean daily listening time is the same for exam and non-exam days.  
**HA:** Mean daily listening time is different for exam and non-exam days.

Test: two-sample t-test.
"""
    ),
    code(
        r'''
h1_result = run_two_sample_ttest(
    analysis_test_df,
    label_col="is_exam",
    value_col="total_minutes",
    label_name="exam days vs non-exam days",
    h0="Mean daily listening time is the same for exam and non-exam days.",
    ha="Mean daily listening time is different for exam and non-exam days.",
)
'''
    ),
    md(
        """
## H2 — Stress periods vs non-stress periods: skip behavior

**H0:** Mean skip rate is the same in stress and non-stress periods.  
**HA:** Mean skip rate is different in stress and non-stress periods.

Test: two-sample t-test.
"""
    ),
    code(
        r'''
if "skip_rate" not in analysis_test_df.columns or analysis_test_df["skip_rate"].isna().all():
    display(Markdown(
        "**Skipped:** `skip_rate` is not available in the merged daily dataset and could not be reconstructed. "
        "This hypothesis should be revisited if stream-level skipped information is available."
    ))
    h2_result = None
else:
    h2_result = run_two_sample_ttest(
        analysis_test_df,
        label_col="is_stress_period",
        value_col="skip_rate",
        label_name="stress periods vs non-stress periods",
        h0="Mean skip rate is the same in stress and non-stress periods.",
        ha="Mean skip rate is different in stress and non-stress periods.",
    )
'''
    ),
    md(
        """
## H3 — Stress periods vs non-stress periods: artist diversity

**H0:** Mean artist diversity is the same in stress and non-stress periods.  
**HA:** Mean artist diversity is different in stress and non-stress periods.

Test: two-sample t-test.
"""
    ),
    code(
        r'''
h3_result = run_two_sample_ttest(
    analysis_test_df,
    label_col="is_stress_period",
    value_col="unique_artists",
    label_name="stress periods vs non-stress periods",
    h0="Mean artist diversity is the same in stress and non-stress periods.",
    ha="Mean artist diversity is different in stress and non-stress periods.",
)
'''
    ),
    md(
        """
## H4 — Deadline days vs non-deadline days: listening frequency

**H0:** Mean number of streams is the same on deadline and non-deadline days.  
**HA:** Mean number of streams is different on deadline and non-deadline days.

Test: two-sample t-test.
"""
    ),
    code(
        r'''
h4_result = run_two_sample_ttest(
    analysis_test_df,
    label_col="is_deadline",
    value_col="num_streams",
    label_name="deadline days vs non-deadline days",
    h0="Mean number of streams is the same on deadline and non-deadline days.",
    ha="Mean number of streams is different on deadline and non-deadline days.",
)
'''
    ),
    md(
        """
## H5 — Before vs after exams: paired listening behavior

**H0:** Mean listening time before exams is the same as mean listening time after exams.  
**HA:** Mean listening time before exams is different from mean listening time after exams.

Test: paired t-test.
"""
    ),
    code(
        r'''
h5_result = run_paired_exam_ttest(analysis_test_df)
'''
    ),
    md(
        """
# Bonus: Chi-square Test of Independence

Create `high_listening = 1` if `total_minutes` is above the 80th percentile, and 0 otherwise.

**H0:** High listening status and exam status are independent.  
**HA:** High listening status and exam status are dependent.
"""
    ),
    code(
        r'''
if "total_minutes" not in analysis_test_df.columns or "is_exam" not in analysis_test_df.columns:
    display(Markdown("**Skipped:** `total_minutes` or `is_exam` is missing, so the chi-square test cannot be run."))
    chi_square_result = None
else:
    chi_df = analysis_test_df[["total_minutes", "is_exam"]].dropna().copy()
    threshold = chi_df["total_minutes"].quantile(0.80)
    chi_df["high_listening"] = (chi_df["total_minutes"] > threshold).astype(int)

    contingency_table = pd.crosstab(chi_df["high_listening"], chi_df["is_exam"])
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    print(f"80th percentile threshold for high listening: {threshold:.4f} minutes")
    print("Contingency table: rows=high_listening, columns=is_exam")
    display(contingency_table)

    if contingency_table.shape != (2, 2) or (contingency_table.sum(axis=1) == 0).any() or (contingency_table.sum(axis=0) == 0).any():
        display(Markdown(
            "**Skipped:** The contingency table does not contain enough variation in both categories for a valid chi-square test."
        ))
        chi_square_result = None
    else:
        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
        decision = decision_from_pvalue(chi2_p)

        print(f"Chi-square statistic: {chi2_stat:.4f}")
        print(f"p-value: {chi2_p:.6f}")
        print("Degrees of freedom:", dof)
        print("Expected counts:")
        display(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
        print(f"alpha: {ALPHA}")
        print("Decision:", decision)

        if decision == "Reject H0":
            interpretation = (
                f"The p-value is below {ALPHA}, so we reject the null hypothesis. "
                "This suggests that high-listening status and exam status are statistically associated."
            )
        else:
            interpretation = (
                f"The p-value is not below {ALPHA}, so we fail to reject the null hypothesis. "
                "This test does not provide enough evidence that high-listening status depends on exam status."
            )
        display(Markdown(f"**Interpretation:** {interpretation}"))

        chi_square_result = {
            "test": "Chi-square test of independence",
            "threshold_80th_percentile": threshold,
            "chi2_statistic": chi2_stat,
            "p_value": chi2_p,
            "degrees_of_freedom": dof,
            "alpha": ALPHA,
            "decision": decision,
        }
'''
    ),
    md(
        """
# Statistical testing summary

This final cell gathers all available hypothesis test results into one compact table. The statistical tests should be read alongside the EDA: tests indicate whether differences are unlikely under the null hypothesis, while EDA explains behavioral meaning.
"""
    ),
    code(
        r'''
results = []
for name, result in [
    ("H1_exam_vs_non_exam_total_minutes", h1_result),
    ("H2_stress_vs_non_stress_skip_rate", h2_result),
    ("H3_stress_vs_non_stress_unique_artists", h3_result),
    ("H4_deadline_vs_non_deadline_num_streams", h4_result),
    ("H5_before_vs_after_exam_total_minutes", h5_result),
    ("Bonus_high_listening_vs_exam_chi_square", chi_square_result),
]:
    if result is not None:
        row = {"hypothesis": name}
        row.update(result)
        results.append(row)

hypothesis_results_df = pd.DataFrame(results)
display(hypothesis_results_df)

if hypothesis_results_df.empty:
    display(Markdown("**Overall interpretation:** No hypothesis tests were completed because the required data was unavailable."))
else:
    rejected = hypothesis_results_df[hypothesis_results_df["decision"] == "Reject H0"]
    display(Markdown(
        f"**Overall interpretation:** {len(rejected)} out of {len(hypothesis_results_df)} completed tests reject the null hypothesis at alpha = {ALPHA}. "
        "Rejected tests provide the strongest statistical support for behavioral changes, while non-rejected tests may still be descriptively interesting but are not statistically strong enough here."
    ))
'''
    ),
]

nb["cells"].extend(extension_cells)
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Updated notebook saved to: {NOTEBOOK_PATH}")
print(f"Total cells: {len(nb['cells'])}")
