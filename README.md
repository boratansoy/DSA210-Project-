# DSA210-Project-
This repository is for DSA210 Project.


NOTE: In the .ipynb file you can also reach the detailed explanation of the study.

EDA and Hypothesis Testing Summary
In this repo in the python notebook, I analyzed whether my Spotify listening behavior changes across different personal and academic contexts, especially exam days, deadlines, stress periods, vacations, school breaks, and major entrance exam preparation periods.

1. Data Loading and Preparation
First, I loaded the cleaned Spotify streaming history dataset and the special dates dataset. The Spotify dataset contains event-level listening records, while the special dates dataset contains daily labels such as exam days, deadline days, academic events, personal events, and stress periods.

I converted timestamps into datetime format and created daily-level listening features. Each day became one observation in the main analysis dataset.

The main daily features included:

Total listening time in minutes
Number of streams
Number of unique artists
Number of unique tracks
Skip rate
Shuffle rate
After creating the daily Spotify dataset, I merged it with the special date labels using the date column.

2. Baseline Listening Behavior
I first explored my general listening behavior without considering special days. This section helped establish my normal listening patterns.

I analyzed:

Daily listening time over time
Distribution of listening time
Daily stream count
Artist diversity over time
This helped identify whether my listening behavior was stable or irregular, whether there were spikes or unusually low listening days, and how much my music habits varied over time.

3. Listening Intensity vs Frequency
Next, I compared listening intensity and listening frequency. Listening intensity was measured using total minutes, while frequency was measured using the number of streams.

I also calculated average stream length to understand whether my listening was more focused and continuous or more fragmented.

This section helped distinguish between days when I listened for a long time and days when I interacted with Spotify frequently but possibly in shorter sessions.

4. Engagement Analysis
I then analyzed engagement-related behavior using skip rate, shuffle rate, and playback-ending information when available.

Skip rate was used as a proxy for attention, satisfaction, or restlessness. Shuffle rate was used as a proxy for more passive or randomized listening behavior.

This section helped explore whether stress or special contexts were associated with more controlled listening or more fragmented listening.

5. Artist and Track Behavior
I analyzed my top artists and top tracks to understand repetition versus exploration in my listening behavior.

This section included:

Top artists by listening frequency
Top tracks by listening frequency
Artist diversity trends over time
This helped identify whether my listening was concentrated around familiar artists or spread across a wider range of music.

6. Time-Based Listening Patterns
I explored how listening changes by time context, including:

Hour of day
Weekday
Month
This section helped identify routine patterns, such as whether I listen more during certain weekdays, months, or times of day.

These patterns are important because special days may disrupt normal routines.

7. Special Day Impact Analysis
The main part of the EDA compared listening behavior on special days versus normal days.

I analyzed:

Exam days vs non-exam days
Deadline days vs non-deadline days
Stress periods vs non-stress periods
For each comparison, I examined:

Total listening minutes
Number of streams
Unique artists
I used summary tables, boxplots, and histograms to compare behavior across groups.

This section helped show whether academic pressure is associated with changes in listening intensity, frequency, or diversity.

8. Behavior Shift Analysis
I created additional time-based features to study changes in behavior over time.

These included:

Daily change in listening time
Seven-day rolling average of listening time
This helped identify sudden increases or decreases in listening behavior and smoother long-term trends.

The goal was to understand whether music listening shifts gradually or changes sharply around certain periods.

9. Event-Centered Exam Analysis
I analyzed listening behavior around exam days using a window before and after exams.

This helped examine whether listening changes before exams, on exam days, or after exams.

The purpose was to detect possible anticipation and recovery effects, where behavior may shift before a stressful event and return to normal afterward.

10. Extreme Listening Days
I identified the highest and lowest listening days in the dataset.

Then I checked whether these extreme days overlapped with special labels such as exams, deadlines, or stress periods.

This section helped determine whether unusual listening behavior could be connected to meaningful life events.

11. Combined Conditions
I examined days where multiple conditions overlapped, such as exam days that were also stress-period days.

This helped explore whether combined academic pressure was associated with stronger behavioral changes than single labels alone.

12. Initial Hypothesis Testing
After the EDA, I added statistical hypothesis tests to evaluate whether observed differences were likely to be meaningful.

The tests included:

Exam days vs non-exam days for daily listening time
Stress periods vs non-stress periods for skip rate
Stress periods vs non-stress periods for artist diversity
Deadline days vs non-deadline days for stream count
Before-exam vs after-exam listening time
A chi-square test for high-listening days and exam status
Most tests used two-sample t-tests, while before/after exam analysis used a paired t-test. The chi-square test was used for categorical high-listening status.

13. Vacation and Break Period Analysis
I then extended the analysis by adding travel_and_break_periods.csv, which contains interval-based labels for travel vacations, school breaks, public holiday blocks, and long breaks.

I expanded these interval labels into daily labels and merged them with the daily Spotify dataset.

This section analyzed whether vacation and break periods changed:

Listening time
Stream count
Time-of-day listening behavior
Artist or track preferences
For time-of-day analysis, I used event-level Spotify records and classified listening into dayparts: night, morning, afternoon, and evening.

14. Genre and Proxy Analysis During Vacations
I checked whether the Spotify dataset contained direct genre information.

Since direct genre data was not available in the cleaned dataset, I did not fabricate genre labels. Instead, I used top artists and tracks during vacation periods as a proxy analysis.

This proxy analysis was clearly separated from formal genre analysis.

15. Vacation Hypothesis Testing
I added additional hypothesis tests for vacation periods.

These included:

Whether genre distribution differs during vacations, if genre data exists
Whether daily listening time differs during vacations
Whether daily stream count differs during vacations
Whether listening time-of-day distribution differs during vacations
The genre test was skipped when genre data was unavailable. Vacation listening time and frequency were tested using two-sample t-tests. Time-of-day behavior was tested using a chi-square test.

16. Coverage-Aware Correction
A key methodological issue was that the datasets do not all cover the same time period.

The Spotify dataset covers a long period from 2018 to 2026, but the special dates and travel/break labels cover shorter or different periods.

To avoid treating unlabeled days as true normal days, I added a coverage-aware correction section.

This section restricted hypothesis tests to the date ranges where the relevant labels were available.

This made the comparisons more reliable and reduced bias from incomplete label coverage.

17. LGS and YKS Entrance Exam Period Analysis
Finally, I added long-term academic stress periods for two major entrance exams:

LGS preparation period: 2019-2020 education year
YKS preparation period: 2023-2024 education year
Since both exams take place in June, I defined the final high-stress phase as March through June.

I created daily labels for:

LGS preparation
YKS preparation
General preparation months
Final high-stress months
This allowed me to compare listening behavior during general preparation versus the final high-stress phase.

18. Entrance Exam Hypothesis Testing
I added hypothesis tests for the LGS/YKS preparation periods.

These tests examined whether the final high-stress phase differed from general preparation months in terms of:

Daily listening time
Artist diversity
Stream count
I also compared LGS preparation and YKS preparation as an exploratory analysis.

These tests helped connect long-term academic stress with changes in music listening behavior.

19. Overall Goal of the Notebook
Overall, the notebook moves from basic listening patterns to deeper behavioral analysis. It combines Spotify listening data with calendar events, vacation periods, and entrance exam preparation windows.

The main goal was to understand whether music listening changes depending on context.

The analysis suggests that music listening is not constant. It appears to vary with academic pressure, stress periods, vacations, breaks, and major exam preparation periods. This supports the idea that music listening behavior is connected to emotional, academic, and routine-based factors.



------------------------------------------------------------------------------------------------------------------------------------------------


ML PART

ML Methods part of the project focuses on modeling how Spotify listening behavior changes from day to day, both in terms of listening intensity and the character of the music being played. This notebook is designed to align closely with the main research question by building a daily-level machine learning dataset from cleaned Spotify streaming history, calendar-based special-day labels, travel and break periods, and partially enriched audio features such as energy, valence, tempo, and acousticness.

The main machine learning task in the notebook is a regression problem that predicts daily listening time (`total_minutes`) using behavioral, temporal, contextual, and audio-based features. Multiple models are compared, including linear baselines and non-linear ensemble methods, followed by model evaluation, feature importance analysis, residual analysis, and learning-curve inspection. In addition, the notebook includes an optional classification task that groups days into low, medium, and high listening-intensity categories.

Overall, the notebook shows that daily listening behavior is not random and can be modeled quite successfully. The strongest predictors are mostly related to listening structure and intensity, such as stream count, stream length, and recent listening trends, while audio features provide a smaller but still meaningful contribution as proxies for music character. This makes the notebook an important complement to the EDA and hypothesis testing stages by showing that daily listening variation can be studied not only descriptively, but also through predictive modeling.


