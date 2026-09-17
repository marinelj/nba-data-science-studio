"""Episode 01: empirical PMF and CDF for LeBron's season-debut FGM."""

from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
from nba_api.stats.endpoints import playergamelog


PLAYER_ID = 2544
START_YEAR = 2005
NUMBER_OF_SEASONS = 21
TARGET_SEASON = "2026-27"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "lebron_season_debut_fgm_2005_2025.csv"
FIGURE_PATH = REPO_ROOT / "videos" / "episode-01-pmf-cdf.png"


def season_label(start_year: int) -> str:
    """Convert 2005 to the NBA season label 2005-06."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


SEASONS = [
    season_label(year)
    for year in range(START_YEAR, START_YEAR + NUMBER_OF_SEASONS)
]


def fetch_season_debuts() -> pd.DataFrame:
    """Fetch LeBron's first regular-season appearance in each selected season."""
    rows: list[dict[str, object]] = []

    for season in SEASONS:
        games = playergamelog.PlayerGameLog(
            player_id=PLAYER_ID,
            season=season,
            season_type_all_star="Regular Season",
            timeout=60,
        ).get_data_frames()[0]

        if games.empty:
            raise RuntimeError(f"No regular-season games returned for {season}")

        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
        first_game = games.sort_values("GAME_DATE").iloc[0]

        rows.append(
            {
                "season": season,
                "game_date": first_game["GAME_DATE"],
                "matchup": first_game["MATCHUP"],
                "fgm": int(first_game["FGM"]),
            }
        )

        sleep(0.7)

    result = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATA_PATH, index=False)
    return result


def load_or_fetch_data() -> pd.DataFrame:
    """Use the local cache when available; otherwise query NBA game logs."""
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH, parse_dates=["game_date"])
    else:
        data = fetch_season_debuts()

    if len(data) != NUMBER_OF_SEASONS:
        raise ValueError(
            f"Expected {NUMBER_OF_SEASONS} seasons, but found {len(data)} rows"
        )

    return data


def empirical_distribution(fgm: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the empirical PMF and CDF over a complete integer support."""
    support = pd.Index(
        range(int(fgm.min()), int(fgm.max()) + 1),
        name="field_goals_made",
    )
    pmf = (
        fgm.value_counts(normalize=True)
        .sort_index()
        .reindex(support, fill_value=0.0)
    )
    cdf = pmf.cumsum()
    return pmf, cdf


def plot_distribution(pmf: pd.Series, cdf: pd.Series) -> None:
    """Plot point probabilities and cumulative probabilities side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(pmf.index, pmf.values, color="steelblue")
    axes[0].set(
        title="Empirical PMF: LeBron Season-Debut FGM",
        xlabel="Field goals made",
        ylabel="Probability",
    )

    axes[1].step(cdf.index, cdf.values, where="post", color="darkorange")
    axes[1].scatter(cdf.index, cdf.values, color="darkorange", s=24)
    axes[1].set(
        title="Empirical CDF: P(FGM ≤ k)",
        xlabel="k field goals made",
        ylabel="Cumulative probability",
        ylim=(0, 1.05),
    )

    figure.suptitle(f"Historical baseline for the {TARGET_SEASON} season debut")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.show()


def main() -> None:
    opening_games = load_or_fetch_data()
    pmf, cdf = empirical_distribution(opening_games["fgm"])

    threshold = 8
    mean_fgm = opening_games["fgm"].mean()
    mode_fgm = int(pmf.idxmax())
    probability_at_most_threshold = opening_games["fgm"].le(threshold).mean()

    print(opening_games.to_string(index=False))
    print(f"\nEmpirical mean FGM: {mean_fgm:.2f}")
    print(f"Empirical mode FGM: {mode_fgm}")
    print(
        f"P(FGM <= {threshold}) from the empirical CDF: "
        f"{probability_at_most_threshold:.1%}"
    )

    plot_distribution(pmf, cdf)


if __name__ == "__main__":
    main()
