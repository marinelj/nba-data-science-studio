# Episode 02: When does LeBron first reach his season high?

This seven-minute course analyzes the team game number when LeBron James first recorded his final highest-scoring game in each completed regular season from 2003–04 through 2025–26.

## Definitions

- Target: the team's actual regular-season game number of the first game tied for LeBron's season-high PTS.
- Tie rule: choose the earliest occurrence after chronological sorting.
- Missed games: count on the team schedule, so team game number and player appearance number remain separate.
- Age: exact elapsed days between 1984-12-30 and LeBron's first appearance that season, divided by 365.2425.
- Debut FGM: made field goals in his first appearance.
- Debut score: total PTS in his first appearance. This is the regression input requested as debut-game score.
- Season-high score: maximum single-game PTS in the completed regular season.

The sample has 23 seasons. Twenty contain 82 team games. The 2011–12, 2019–20 and 2020–21 schedules contain 66, 71 and 72 games. The main result keeps actual game numbers; the analysis also reports an 82-game-only subset and a separately labeled normalized position.

## Main results

- Empirical first-high game number: mean 41.61, median 49, mode 49.
- Empirical CDF: 43.5% by team game 41 and 82.6% by game 60.
- Pearson correlation: age versus debut FGM = −0.085.
- Pearson correlation: age versus season-high PTS = −0.258.
- OLS: predicted game # = 75.728 − 1.018 × age − 0.154 × debut PTS.
- Expanding-window evaluation over 13 later seasons: model MAE 18.43 games; past-mean baseline MAE 18.81 games.

The equation is descriptive and weakly predictive. Its small MAE improvement and worse RMSE do not establish a useful forecasting advantage.

## Run

```bash
source ../../.venv/bin/activate
python python/episode_02_season_high.py
python -m unittest python/test_episode_02.py
```

Refresh the official NBA responses only when intended:

```bash
python python/fetch_data.py --refresh
python python/episode_02_season_high.py
python python/build_assets.py
```

## Course assets

- [`episode-02-outline.md`](episode-02-outline.md)
- [`html/episode-02-recording-lab.html`](html/episode-02-recording-lab.html)
- [`python/episode_02_season_high.ipynb`](python/episode_02_season_high.ipynb)
- [`subtitles/episode-02-season-high-en.srt`](subtitles/episode-02-season-high-en.srt)
- [`subtitles/episode-02-script.md`](subtitles/episode-02-script.md)
- [`data/season_summary.csv`](data/season_summary.csv)
- [`data/source_manifest.json`](data/source_manifest.json)
