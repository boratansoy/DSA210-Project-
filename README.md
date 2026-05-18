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


------------------------------------------------------------------------------------------------------------------------------------------------


Final Report 

FINAL REPORT

Project Title  
How Does My Spotify Listening Behavior Change on Special Days?

1. Introduction

The main goal of this project is to understand whether my Spotify listening behavior changes on different kinds of days. I was especially interested in exam days, deadline periods, stress periods, vacations, and some personally meaningful days.

I studied this question from two sides. The first side was the behavior side, which includes listening time, number of streams, artist diversity, skip rate, and shuffle rate. The second side was the music side, which includes the character of the music I listened to, using audio features such as energy, valence, tempo, and acousticness.

The project had three main stages: data preparation and cleaning, exploratory data analysis and hypothesis testing, and machine learning methods.

2. Data Sources

2.1 Spotify Extended Streaming History

My main data source was my Spotify Extended Streaming History export. It covered my listening history from 2018 to 2026. Each row represented one listening event.

From this dataset, I used variables such as timestamp, milliseconds played, track name, artist name, reason_start, reason_end, shuffle, and skipped.

These files were cleaned and merged into one main dataset called spotify_cleaned.csv.

2.2 Calendar-Based Special Day Labels

I also created a special-day dataset from my calendar data. This included exams, quizzes, deadlines, project submissions, academic events, and some personal events.

I also created stress periods automatically by marking the 7 days before exam days and the 3 days before deadline days.

This produced a file called special_dates.csv.

2.3 Travel and Break Periods

I created another dataset for travel vacations, school breaks, public holiday blocks, and long breaks.

This produced a file called travel_and_break_periods.csv.

2.4 Audio Features

My cleaned Spotify data did not directly include audio features like danceability, energy, or valence. Because of this, I created an extra audio-feature enrichment step.

First, I used the Spotify listening data already available locally and built a track-level catalog from my listening history. Then I matched these tracks with public music-feature datasets. After that, I merged the matched audio features back into my Spotify data and aggregated them to the daily level.

The audio features I used included danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness, liveness, loudness, key, and mode.

The final coverage was limited:
- total unique tracks: 28,660
- unique tracks matched with audio features: 4,316
- unique-track coverage: about 15.1%
- event-level coverage: about 17.0%
- ms-weighted event coverage: about 18.9%

These daily audio summaries were saved in spotify_daily_audio_features.csv.

3. Data Preparation

In the data preparation step, I first merged all streaming history files into one common format. Then I converted timestamps into datetime format and assigned each listening event to a calendar date.

The cleaning process included removing rows where ms_played = 0, removing duplicate rows, handling missing track and artist names, and sorting the data by time.

After cleaning, I converted the event-level data into a daily-level dataset.

For each day, I created total listening time (total_minutes), number of streams (num_streams), number of unique artists (unique_artists), number of unique tracks (unique_tracks), skip rate (skip_rate), shuffle rate (shuffle_rate), session count, average stream length, and daypart features such as morning, afternoon, evening, and night listening share.

Then I merged this daily dataset with calendar-based special-day labels, travel and break labels, and daily audio features.

This created the main daily dataset used for analysis.

4. Exploratory Data Analysis

In the EDA stage, I first looked at my general listening behavior over time.

I examined daily listening time, daily stream count, artist diversity, day-based trends, outlier days, and weekday and month patterns.

These analyses showed that my Spotify behavior was not constant across time. There were some periods with much higher listening activity and some with lower activity.

Then I merged the daily Spotify data with special-day labels and compared exam days vs normal days, deadline days vs normal days, stress periods vs non-stress periods, vacation periods vs non-vacation periods, and LGS/YKS preparation periods vs other periods.

The EDA results showed that my listening behavior changed across these contexts. In some stressful periods, listening time increased. In other periods, listening intensity decreased. Diversity and engagement measures also changed in some cases.

Overall, the EDA stage suggested that my listening behavior is context-sensitive.

5. Hypothesis Testing

After EDA, I used hypothesis testing to examine whether some of the observed differences were statistically meaningful.

The main hypotheses included whether daily listening time differs between exam days and non-exam days, whether skip rate differs between stress and non-stress periods, whether artist diversity differs between stress and non-stress periods, whether stream count differs between deadline and non-deadline days, and whether listening behavior changes before and after exams.

Later, I also added tests for vacation periods, travel periods, and LGS and YKS preparation windows.

The general result of the hypothesis testing stage was that some contexts were clearly associated with changes in listening behavior, but not all effects were equally strong.

The strongest behavioral shifts usually appeared during academic pressure periods, stress periods, and some special-day windows.

6. Machine Learning Methods

In the machine learning stage, I used the main notebook in the project repository: ml_methods_original.ipynb.

This notebook matched my main research question better because it focused on daily listening change.

The main target was total_minutes, which means daily listening time.

So the main machine learning question became: Can daily listening time be explained using behavioral, temporal, contextual, and audio-feature-based variables?

The models used were:
- Dummy Regressor
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

I also added an optional classification task by dividing listening days into Low, Medium, and High listening-intensity groups.

6.1 Why I Used These ML Methods

I used these models because they represent different levels of complexity.

Dummy Regressor gave a simple baseline. Linear Regression and Ridge Regression showed whether daily listening patterns could be explained with mostly linear relationships. Random Forest, Extra Trees, and Gradient Boosting were used because they can model more complex and non-linear patterns.

This was important because listening behavior is probably not perfectly linear. It may depend on many small interactions between context, routine, and music character.

6.2 ML Design Choices

In the notebook, I used:
- a chronological train-test split
- scaling
- cross-validation
- hyperparameter tuning
- feature importance analysis
- residual analysis

These steps made the model comparison more careful and more realistic.

7. Machine Learning Results

7.1 Main Regression Results

The main regression task gave strong results.

The best model was Gradient Boosting.

Its results were:
- MAE ≈ 5.14
- RMSE ≈ 8.06
- R² ≈ 0.978

Other strong models were:
- Extra Trees: R² ≈ 0.974
- Random Forest: R² ≈ 0.966

The linear models were weaker:
- Linear Regression: R² ≈ 0.874
- Ridge Regression: R² ≈ 0.874

The baseline dummy model had almost no explanatory power.

At first, these regression results look extremely strong. However, they must be interpreted carefully.

7.2 Important Note About the Regression Task

A very important feedback point in this project was that some features were too close to the target variable total_minutes.

For example:
- num_streams
- avg_stream_length
- minutes_per_stream

These variables are directly related to daily listening time. Because of this, the model may have learned the structure of listening time very well, but this does not necessarily mean it has strong fully independent predictive power.

For this reason, I interpret this machine learning model mainly as a descriptive and explanatory model, not as a pure predictive model.

In other words, it is very useful for explaining how daily listening behavior is structured, but the very high R² values should be read carefully.

7.3 Optional Listening-Tier Classification

The optional classification task grouped days into Low listening, Medium listening, and High listening.

The results were:
- accuracy ≈ 0.848
- macro F1 ≈ 0.849

This shows that the model can also separate broad types of listening days, not only predict the exact number of minutes.

8. What the Models Show

The most important result from the machine learning stage is this: daily listening behavior is not random. It changes in structured ways across days.

The strongest predictors were mostly behavioral variables, such as num_streams, avg_stream_length, minutes_per_stream, delta_total_minutes, and rolling_7_total_minutes.

This means that the clearest signal comes from how I listen, such as listening intensity, session structure, and short-term routine changes.

Some audio-feature variables also appeared as useful predictors, such as audio_feature_total_ms, tempo, and liveness.

This suggests that the character of the music may also matter, but it is not the strongest driver in the current version of the model.

So the machine learning stage supports the following interpretation:
- my listening changes from day to day
- this change is strongly connected to listening behavior structure
- context matters
- music character adds a smaller but still meaningful contribution

9. Relation to the Main Research Question

My main research question was: How does my Spotify listening behavior change across days, both in terms of listening intensity and in terms of the kind of music I listen to?

The final machine learning notebook is relevant to this question because it directly models daily listening change.

It does not only try to classify special days. Instead, it tries to explain variation in listening time, listening frequency, listening structure, time-of-day behavior, and music-character proxies through audio features.

This means the notebook is closely connected to the core aim of the project.

The results support the idea that Spotify listening behavior changes across days in meaningful ways, and that these changes are connected more strongly to listening intensity and routine than to music-character features alone.

10. Limitations

This project has several important limitations.

10.1 Single-User Dataset

The project is based only on my personal Spotify data. Because of this, the results are personal and cannot be generalized to everyone.

10.2 Limited Calendar and Context Coverage

The calendar-based labels do not cover all years equally. Some special-day labels are available only for certain periods. The travel and break labels are also limited to specific windows.

This means that the context side of the dataset is not complete.

10.3 Limited Audio-Feature Coverage

The audio-feature layer is only partial.

The final coverage was:
- 15.1% of unique tracks
- 17.0% of event rows
- 18.9% of ms-weighted listening rows

This means that conclusions about music character should be interpreted carefully.

10.4 Proxy-Based Music-Type Analysis

A complete genre dataset was not available. Because of this, the type of music side of the project was analyzed using audio-feature proxies such as energy, valence, tempo, and acousticness.

This is useful, but it is not the same as having a complete genre-level dataset.

10.5 Target-Related Features

Some of the regression predictors were very close to the target variable total_minutes. This is why the regression results should be seen more as descriptive and explanatory rather than as a strong proof of independent prediction.

11. What Could Be Improved

This study could be improved in a few important ways.

First, the project would likely be more consistent if the calendar data that affects behavior had been recorded in a more detailed and systematic way. If the daily context data had included more complete and carefully tracked information, the analysis of behavior changes would probably have been stronger and more reliable.

Second, the project would benefit from a better and cleaner music-type dataset. If a more complete genre-related dataset or a more structured music-feature dataset had been available, the analysis of what kind of music I listen to on different days would likely have produced better results.

Third, audio-feature coverage could be improved. A broader and more complete audio-feature enrichment source would strengthen the music-character side of the project.

Finally, a future version of the regression model could remove features that are very directly related to total_minutes. This would make the model less descriptive and more strictly predictive.

12. Conclusion

This project shows that my Spotify listening behavior changes across days in meaningful ways.

The EDA, hypothesis testing, and machine learning stages all support the idea that listening behavior is connected to context. Exam periods, deadlines, stress periods, vacations, and other meaningful days are related to changes in my Spotify activity.

The main findings are:
- daily listening behavior is not stable
- listening intensity changes across days
- special periods are associated with behavioral shifts
- daily listening variation can be modeled successfully
- the strongest explanatory factors are behavioral intensity and listening structure
- audio features provide useful but limited extra information

Overall, this project suggests that Spotify listening behavior is closely related to daily routine, academic pressure, and personal context. A stronger future version of the study could produce more consistent and more complete results if the context data were more detailed and if the music-type dataset were more complete and better structured.

