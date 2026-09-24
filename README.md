# NBA Data Science Studio

Short, reproducible lessons that use NBA questions to explain statistics, data science, and machine learning.

## Courses

| Episode | Question | Main methods | Folder |
|---|---|---|---|
| 01 | What does LeBron's season-debut FGM distribution look like? | Empirical PMF and CDF | [`episodes/episode-01-cdf-pmf`](episodes/episode-01-cdf-pmf) |
| 02 | When does LeBron first reach his highest score of the season? | Empirical PMF/CDF, correlation and linear regression | [`episodes/episode-02-season-high`](episodes/episode-02-season-high) |

Each episode is self-contained. Its `data`, `html`, `images`, `python`, `subtitles`, and `videos` folders stay together under the episode directory.

## Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Episode 01

```bash
python episodes/episode-01-cdf-pmf/python/episode_01_cdf_pmf.py
```

## Run Episode 02

The repository includes cached, auditable source responses and derived tables. The analysis runs offline:

```bash
python episodes/episode-02-season-high/python/episode_02_season_high.py
```

Refresh the NBA responses intentionally:

```bash
python episodes/episode-02-season-high/python/fetch_data.py --refresh
python episodes/episode-02-season-high/python/episode_02_season_high.py
python episodes/episode-02-season-high/python/build_assets.py
uv run episodes/episode-02-season-high/python/build_narration_audio.py
```

To present the lessons in Jupyter, open the notebook in the episode's `python` folder and run its cells from top to bottom.

## Source scope

The data comes from NBA statistics endpoints through the third-party [`nba_api`](https://github.com/swar/nba_api) wrapper. Episode 02 stores each raw response together with its request URL, retrieval time, package version, and SHA-256 checksum. The code reconciles detailed player logs against the official season totals before analysis.
