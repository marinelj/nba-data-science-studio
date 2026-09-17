# Episode 01 Outline: CDF and PMF Through an NBA Question

## Learning objective

Use LeBron James's season-debut field goals made to explain the relationship between a PDF, a PMF, and a CDF, then estimate an empirical distribution in Python.

## Seven-minute structure

| Time | Section | Content | On screen |
|---|---|---|---|
| 00:00-00:30 | Opening | Introduce the 2026-27 season-debut question and the lesson goal | Title and prediction question |
| 00:30-02:00 | Concepts | Random variable, PDF, PMF, CDF, and the meaning of $P(X \leq k)$ | Simple probability sketches |
| 02:00-02:30 | NBA example | Define the 21-season sample and clarify season debut versus team opening night | Data definition card |
| 02:30-06:30 | Python walkthrough | Fetch, clean, summarize, estimate, visualize, and interpret the empirical distribution | Jupyter notebook |
| 06:30-07:00 | Closing | Summarize the distinction between PMF and CDF and state model limitations | Two-chart recap |

## Concepts addressed

- Random variable: $X=$ LeBron's field goals made in a season debut.
- PDF: density for a continuous variable; probability comes from area over an interval.
- PMF: point probability for a discrete variable such as FGM.
- CDF: cumulative probability $F(k)=P(X \leq k)$.
- Empirical distribution: probabilities estimated by relative frequency in observed data.
- Point estimate versus distribution: a mean is one summary; the PMF and CDF show uncertainty.

## Daily example

The episode asks: based on LeBron's first regular-season appearance in his first 21 NBA seasons, what does the empirical distribution suggest about his FGM in his 2026-27 season debut?

The code uses seasons 2003-04 through 2023-24. It deliberately selects LeBron's first game played in each season. A strict team-opening-night study would require the team schedule and explicit treatment of DNP records.

## Python walkthrough

- Import `pandas`, `matplotlib`, and `nba_api`.
- Generate the 21 season labels programmatically.
- Download one regular-season game log per season.
- Parse and sort dates before selecting the first appearance.
- Cache the 21-row dataset locally.
- Estimate the PMF with normalized value counts.
- Estimate the CDF with a cumulative sum of the PMF.
- Report the empirical mean, mode, and $P(X \leq 8)$.
- Plot the PMF and CDF side by side.
- Explain why this is a descriptive baseline rather than a production forecast.

## Closing message

Use the PMF to answer “exactly how many?” and the CDF to answer “at most how many?” The workflow converts 21 historical observations into an interpretable baseline while keeping its assumptions visible.
