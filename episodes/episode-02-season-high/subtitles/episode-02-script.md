# Episode 02 Recording Script

## 00:00–00:30 · Opening

When will LeBron first reach his highest score of the season? Let us study twenty-three seasons of NBA data.

An empirical distribution describes the historical timing. Correlation and linear regression help us test whether age and debut points add useful information.

Our outcome is the team's game number, including games he missed. If he ties his season high, we select the earliest game.

## 00:30–02:00 · Hand-drawn intuition

First, draw a season as a horizontal line from game one to game eighty-two. Mark the first game with the season's highest score.

Each season contributes one mark. For a small, imaginary example, place four marks at games ten, thirty, thirty, and seventy.

Now stack repeated marks. Game thirty appears twice out of four seasons, so its empirical probability is one half.

Draw bars at ten, thirty, and seventy, with heights one quarter, one half, and one quarter. This is the empirical PMF.

For the CDF, move from left to right and accumulate probability. At game thirty, three of four marks have been counted.

Draw a staircase: one quarter at ten, three quarters at thirty, and one at seventy. Between observed positions, the CDF stays flat.

Game numbers are discrete, so exact probabilities use a PMF. A PDF describes continuous density, where probability comes from area.

A density histogram can summarize grouped season positions, but it is an approximation. Its bar heights are densities, not individual game probabilities.

In Python, normalized value counts build the PMF. A cumulative sum builds the CDF. Let us replace the imaginary marks with real data.

## 02:00–06:30 · Python walkthrough

We import pandas, NumPy, and Matplotlib. Our cached NBA responses keep the recording reproducible, so we do not wait for downloads on camera.

The download script uses the third-party nba_api wrapper to request official NBA player and team game logs, limited to regular-season games.

We parse dates and sort chronologically. Then we number every team game. Missing a game does not pause the team's season counter.

Next, merge LeBron's appearances with his team's schedule using game IDs. A one-to-one validation rejects duplicate matches, and unmatched games raise an error.

For each season, compute the maximum points. Filter to games with that score, then select the earliest. This resolves ties explicitly.

Keep the team's game number and LeBron's appearance number separately. The first is our target. The second helps check the meaning of each row.

Age is measured on his first appearance that season. We also record debut field goals made, debut total points, and season-high points.

These are different quantities. The correlation uses debut FGM. The regression uses debut points. Season-high points help define the target, but never enter the predictors.

The dataset has twenty-three seasons, beginning in two thousand three. We reconcile every player's game count, points, and FGM against NBA season totals.

Only twenty seasons contain eighty-two team games. Three contain sixty-six, seventy-one, and seventy-two games, so we preserve their actual schedule lengths.

Our main analysis uses actual game numbers across all seasons. We also compare only full schedules and a separately labeled eighty-two-game equivalent.

Now select the target column. Each row contributes one first-maximum position, regardless of how many games LeBron played during that season.

Value counts with normalize set to true gives relative frequencies. Reindex from one through eighty-two to include game numbers that never occurred.

Apply cumulative sum to get the empirical CDF. By team game forty-one, 43.5 percent of these historical season highs had first occurred.

That is a descriptive frequency, not a guaranteed probability for next season. A zero-frequency game number is also possible in future data.

Next, calculate Pearson correlation. Age and debut FGM have a correlation of minus zero point zero eight five, showing little linear association in this sample.

Age and season-high points have a correlation of minus zero point two five eight. The association is negative, but it does not establish a causal effect of aging.

These observations come from one career. Age also tracks changes in teams, playing time, injuries, and role. The notebook includes rank correlation as a check.

Now build the regression matrix: one column for the intercept, one for age, and one for debut points. Our response is the first-maximum game number.

NumPy's least-squares function estimates three coefficients. The fitted equation starts at seven five point seven three, subtracting one point zero one eight times age and zero point one five four times debut points.

Each slope describes a conditional linear association while holding the other predictor fixed. We can use both inputs only after his season debut is complete.

To evaluate prediction, train on the first ten seasons and predict the next. Expand the training window, then repeat for every later season.

Each prediction uses only earlier seasons for training. Compare it with the average target from those same earlier seasons, our simple baseline.

Across thirteen held-out seasons, regression has a mean absolute error of one eight point four three games. The historical-mean baseline has an error of one eight point eight one games.

That improvement is small. Regression also has worse root mean squared error, so these predictors have not demonstrated a clear forecasting advantage.

The notebook lets us enter an age and debut score for a scenario. It labels extrapolation and keeps the raw prediction visible before rounding.

Here are the distributions, age relationships, and prediction errors. Take a brief look, then return to the main lesson about uncertainty.

## 06:30–07:00 · Closing

The PMF describes exact historical positions, and the CDF answers by which game. A density histogram offers a grouped approximation.

Correlation describes association. Regression turns selected inputs into a prediction, but its usefulness must be tested against a simple baseline.

Here, age and debut points leave substantial uncertainty. We learned how to build an honest model, and how to recognize its limits.
