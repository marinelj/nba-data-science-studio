"""Episode 01: empirical PMF and CDF for LeBron's season-debut FGM."""

from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
from nba_api.stats.endpoints import playergamelog


PLAYER_ID = 2544
START_YEAR = 2003
NUMBER_OF_SEASONS = 23
TARGET_SEASON = "2026-27"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "lebron_season_debut_fgm_2003_2025.csv"
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
    highlights = {9: "seagreen", 12: "crimson"}

    axes[0].bar(pmf.index, pmf.values, color="steelblue")
    for k, color in highlights.items():
        axes[0].bar(k, pmf.loc[k], color=color, edgecolor="black", zorder=3)
        axes[0].annotate(
            f"k = {k}\nP(X = {k}) = {pmf.loc[k]:.1%}",
            xy=(k, pmf.loc[k]),
            xytext=(-52 if k == 9 else 8, 26),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": color},
            color=color,
            fontsize=9,
        )
    axes[0].set(
        title="Empirical PMF: LeBron Season-Debut FGM",
        xlabel="Field goals made",
        ylabel="Probability",
    )

    axes[1].step(cdf.index, cdf.values, where="post", color="darkorange")
    axes[1].scatter(cdf.index, cdf.values, color="darkorange", s=24)
    for k, color in highlights.items():
        axes[1].axvline(k, color=color, linestyle="--", alpha=0.7)
        axes[1].scatter(k, cdf.loc[k], color=color, edgecolor="black", s=75, zorder=3)
        axes[1].annotate(
            f"k = {k}\nP(X ≤ {k}) = {cdf.loc[k]:.1%}",
            xy=(k, cdf.loc[k]),
            xytext=(-76 if k == 9 else -104, -48),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": color},
            color=color,
            fontsize=9,
        )
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

    mean_fgm = opening_games["fgm"].mean()
    median_k = int(opening_games["fgm"].median())
    mode_fgm = int(pmf.idxmax())
    probability_at_most_median = cdf.loc[median_k]
    probability_exactly_mode = pmf.loc[mode_fgm]
    probability_at_most_mode = cdf.loc[mode_fgm]

    print(opening_games.to_string(index=False))
    print(f"\nEmpirical mean FGM: {mean_fgm:.2f}")
    print(f"Empirical median FGM: {median_k}")
    print(f"Empirical mode FGM: {mode_fgm}")
    print(
        f"P(FGM <= {median_k}) from the empirical CDF: "
        f"{probability_at_most_median:.1%}"
    )
    print(f"P(FGM = {mode_fgm}) from the empirical PMF: {probability_exactly_mode:.1%}")
    print(f"P(FGM <= {mode_fgm}) from the empirical CDF: {probability_at_most_mode:.1%}")

    plot_distribution(pmf, cdf)


if __name__ == "__main__":
    main()
