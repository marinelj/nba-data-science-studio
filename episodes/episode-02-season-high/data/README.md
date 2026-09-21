# Episode 02 Data

## Derived tables

- `season_summary.csv`: one observation per completed season and the primary analysis input.
- `lebron_regular_season_games.csv`: chronologically ordered player appearances joined to team game number.
- `source_reconciliation.csv`: detailed-log GP, PTS and FGM totals beside NBA career-total controls.
- `empirical_distribution.csv`: PMF and CDF on team game numbers 1–82.
- `correlations.csv`: Pearson and Spearman coefficients.
- `walk_forward_predictions.csv`: chronological held-out predictions and baselines.
- `results.json`: exact reported metrics and schedule-length sensitivity.

## Raw evidence

`raw/` contains 48 JSON envelopes: player information, career totals, 23 player game logs and 23 team game logs. Each envelope contains the source request URL, retrieval time, installed `nba_api` version and the original response. `source_manifest.json` records SHA-256 checksums.

The source endpoints are official NBA statistics endpoints accessed through the third-party [`nba_api`](https://github.com/swar/nba_api) wrapper. Regeneration fails when a response is missing or a detailed log does not reconcile to season totals; it does not substitute values from another provider.
