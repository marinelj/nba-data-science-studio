# Episode 02 Outline: First Season-High Game

## Learning objective

Turn one event from each of LeBron James's 23 completed regular seasons into an empirical distribution, measure two age relationships, and evaluate a two-feature linear regression model without data leakage.

## Seven-minute structure

| Time | Section | Content | On screen |
|---|---|---|---|
| 00:00–00:30 | Opening | Define the prediction question, target, missed-game rule and tie rule | Title and 23-season sample |
| 00:30–02:00 | Hand-drawn intuition | Draw imaginary first-high positions, stack an empirical PMF, accumulate a CDF, and distinguish density | Drawing canvas and four-season sketch |
| 02:00–06:30 | Python walkthrough | Load, verify, define, estimate, correlate, regress, backtest and try a scenario | Notebook and step-by-step code |
| 06:30–07:00 | Closing | Recap PMF/CDF, correlation, regression and predictive limits | Results and 23-season evidence table |

## Concepts

- Empirical PMF for an exact integer game number.
- Empirical CDF for the fraction of first maxima observed by game $k$.
- Density histogram as a grouped approximation of normalized season position; its bar area represents relative frequency.
- Pearson and Spearman correlation.
- Multiple linear regression with an intercept, age and debut total points.
- Chronological expanding-window evaluation against past-mean and past-median baselines.
- Leakage prevention: training rows always precede the held-out season.
- Sensitivity to shortened regular seasons.

## Python walkthrough

- Load a verified 23-row season table from cached NBA responses.
- Explain why team game number differs from LeBron's personal appearance number.
- Reconcile detailed player logs against season GP, PTS and FGM totals.
- Estimate the PMF with normalized counts on the full support 1–82.
- Estimate the CDF with a cumulative sum.
- Calculate age correlations with debut FGM and season-high PTS.
- Fit $\hat G=b_0+b_1 age+b_2 debut\ points$.
- Test the model on 13 later seasons using only earlier seasons for training.
- Show the generated charts for approximately ten seconds.

## Closing message

An empirical distribution describes where the first season high has appeared historically. Correlation describes association. A regression equation turns chosen inputs into a number, but its value comes from out-of-sample performance. Here, age and debut points leave large uncertainty.
